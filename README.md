# Zervram (`zervram-open` v0.3.0)

**Interactive Live Console Demo:** https://zervram.netlify.app

Zervram is an open-source, pure-Rust GPU VRAM sanitization and Ed25519 attestation toolkit (`zervram-hal` + `zervram-scrub`). It compiles with **zero compile-time C/C++ GPU SDK headers**, dynamically loads native NVIDIA CUDA (`libcuda.so.1` / `nvcuda.dll`), NVIDIA NVML (`libnvidia-ml.so.1` / `nvml.dll`), and AMD ROCm HIP (`libamdhip64.so` / `amdhip64.dll`) libraries at runtime, enforces a pre-scrub active-process check, sweeps up to **99%** of free physical VRAM down a chunk ladder until OOM, runs a **100% GPU-side PTX 64-bit bitwise-OR reduction verification kernel** plus CSPRNG-random `0xAA`-poisoned canary page read-backs, and appends Ed25519-signed `ZeroizationCertificateV2` records to a file-locked persistent SHA-256 hash chain (`chain.jsonl`).

> **Runtime Driver Requirement:** Zero compile-time SDK headers are needed to build `zervram-scrub`, but **runtime execution on physical GPUs strictly requires the proprietary NVIDIA CUDA + NVML driver (`libcuda.so.1` + `libnvidia-ml.so.1` on Linux, or `nvcuda.dll` + `nvml.dll` on Windows) or AMD ROCm HIP runtime (`libamdhip64.so` / `amdhip64.dll`)**.
>
> **Backend Maturity:** `cuda-native` (NVIDIA CUDA + NVML + PTX reduction), `rocm-native` (AMD HIP + 256-page CSPRNG read-back), and `mock-gpu` (CPU simulation) are implemented. `furiosa-native` and `tt-metalium-native` are explicitly marked **`experimental-stub`** until vendor SDK FFI bindings are linked.

---

## Threat Model, Verification Physics & Compliance Positioning

### Global HBM/GDDR Persistence vs. `CVE-2023-4969` (`LeftoverLocals`)
- **What `CVE-2023-4969` (`LeftoverLocals`) actually is:** An architectural vulnerability in *local/shared* GPU memory (`__local` / threadgroup SRAM) affecting specific OpenCL, Vulkan, and Metal drivers (Apple, AMD, Qualcomm, Imagination). It does **not** directly exploit NVIDIA CUDA global HBM3.
- **What `zervram-scrub` mitigates on CUDA & ROCm datacenter GPUs:** Residual global HBM/GDDR page persistence and driver memory-pool reuse between multi-tenant containers or Slurm jobs. When GPU memory is freed without an explicit zeroization pass, residual weights, LoRA adapters, KV-cache activations, and prompt data can survive in physical memory pages later allocated to another tenant.
- **How v0.3.0 verifies 100% of swept VRAM:**
  1. **NVML Pre-Scrub Gate:** Probes `libnvidia-ml.so.1` (`nvmlDeviceGetComputeRunningProcesses_v3`) and **fails closed** if another compute process is still attached to the GPU (overridable only via `--force`, with `active_processes_before` signed into the certificate), and quarantines devices reporting uncorrectable ECC row-remapping failures.
  2. **99% Chunk Ladder Sweep:** Reserves a 16-byte device accumulator, allocates descending chunks (`256 MiB -> 64 MiB -> 16 MiB -> 4 MiB -> 1 MiB -> 64 KiB -> 4 KiB`) up to 99% of free VRAM until `CUDA_ERROR_OUT_OF_MEMORY` while holding all chunks concurrently, queues `cuMemsetD32Async` / `hipMemsetAsync`, and synchronizes once.
  3. **100% GPU PTX Reduction + `0xAA`-Poisoned Canary Read-Back:** On CUDA, dynamically loads an embedded forward-compatible PTX ISA 7.0 (`target sm_50`) kernel via `cuModuleLoadData` + `cuLaunchKernel` that scans 100% of `swept_bytes` at full HBM/GDDR bandwidth into a `0xAA`-poisoned device/host `[u64; 2]` accumulator (verifying both `or_mask == 0` and exact `words_scanned == swept_bytes / 8`), plus 64 CSPRNG-random (`ChaCha20Rng::from_entropy()`) 4 KiB canary pages copied into `[0xAA; 4096]` poisoned host buffers. If PTX JIT is unavailable, it gracefully falls back to `csprng_sampled_256_pages` and honestly records the exact `verified_bytes` and `verification_mode` inside the Ed25519 signature.
  4. **Persistent Hash Chain & Quarantine:** Appends each `ZeroizationCertificateV2` (with `schema_version`, `backend_kind`, `device_ordinal`, `vram_total_bytes`, `vram_free_bytes`, `swept_bytes`, `chunk_count`, `chunk_table_sha256`, `verification_mode`, `verified_bytes`, `sampled_pages`, `active_processes_before`, and optional `--nonce` all covered by length-prefixed canonical Ed25519 signing) to a file-locked, `sync_all()`-flushed `chain.jsonl`. Residual data or ECC faults persist the GPU to `quarantine.json` and exit with code **`42`**.

### Measured vs. Theoretical Latency
Cold CLI invocations include OS process startup, `cuCtxCreate_v2`, chunk `cuMemAlloc_v2`/`cuMemFree_v2`, verification, `cuCtxDestroy_v2`, `chain.jsonl` fsync, and Ed25519 signing â€” which adds ~300â€“450 ms of driver context overhead on top of raw `cuMemsetD32Async` bandwidth:

| Target Hardware | Sweep Scope | Raw `cuMemsetD32Async` Bound | Observed Cold CLI End-to-End |
|---|---|---:|---:|
| **16 GiB NVIDIA Tesla T4 (GDDR6, Google Colab)** | ~13.97 GiB (v0.2.0 live run) | ~55â€“70 ms (~220â€“300 GB/s) | **508.4 ms (measured)** |
| **80 GiB NVIDIA H100 SXM (HBM3)** | ~79 GiB (99% free) | ~26â€“30 ms theoretical (~2.8â€“3.1 TB/s) | ~350â€“500 ms est. (unbenchmarked cold CLI) |
| **Software `mock-gpu`** (CPU CI/dev test pool, 512 KiB backing pages) | 512 KiB host RAM | < 3 ms | **< 10 ms** |

> Note: In `mock-gpu`, the default `12..=18%` simulated dirty-page retention rate is an arbitrary synthetic test parameter for unit/CI tests, not an empirical measurement on NVIDIA HBM/GDDR hardware.

### Honest Compliance & Regulatory Positioning
`zervram-open` is designed for **multi-tenant data hygiene, defense-in-depth, and verifiable sanitization audit trails** supporting **SOC 2**, **ISO 27001**, **HIPAA**, and enterprise customer security questionnaires. While `zervram-hal::catalog` includes reference TPP calculations aligned with U.S. BIS Interim Final Rule 91 FR 74970 / 15 CFR Â§ 744.24 (effective November 19, 2026; 12,000,000 aggregate TPP ceiling and Â§ 743.10 reporting), **neither Â§ 744.24 nor Â§ 743.10 mandates GPU VRAM zeroization or Ed25519 scrub certificates**.

---

## Quick Start: `zervram-scrub` CLI

Build release binary (requires Rust 1.82+; zero compile-time CUDA/ROCm SDK or OpenSSL headers):

```bash
cargo build --release --bin zervram-scrub
```

### 1. Physical GPU Sanitization (Fail-Closed Default)
Probes `libcuda.so.1` + `libnvidia-ml.so.1` (or `libamdhip64.so`) at runtime, refuses to run if another compute process holds the GPU (unless `--force`), persists the Ed25519 key atomically (`0600` on Unix), and appends to the persistent hash chain (`~/.local/share/zervram/chain.jsonl` on Linux, `%LOCALAPPDATA%\zervram\chain.jsonl` on Windows, overridable with `--chain-file`):

```bash
zervram-scrub --device 0 --key-file /etc/zervram/node.key --nonce "$JOB_NONCE" --json --out-cert cert.json
```

### 2. Local Laptop / CI Simulation (`--allow-mock`)
Explicitly opt into `mock-gpu` when testing without datacenter GPUs:

```bash
cargo run --release --bin zervram-scrub -- \
  --device 0 --allow-mock \
  --key-file node.key \
  --chain-file chain.jsonl \
  --nonce dev-run-1 \
  --json --out-cert cert.json
```

### 3. Fail-Closed Offline Verification (`verify` & `verify-chain`)
`verify` and `verify-chain` **fail closed** unless anchored to a pinned public key via `--pubkey <hex>`, `--key-file <path>`, or `--trust-dir <dir>` (never silently trusting an attacker-controlled key embedded inside `cert.json`):

```bash
# Verify a single exported bundle against a pinned key (and optional expected nonce)
zervram-scrub verify cert.json --key-file node.key --expected-nonce dev-run-1

# Or pin directly by 64-char hex Ed25519 public key
zervram-scrub verify cert.json --pubkey <64-char-hex-pubkey>

# Audit the entire persistent hash chain from genesis to head
zervram-scrub verify-chain --chain-file chain.jsonl --key-file node.key
```

---

## Orchestrator Integrations

### Slurm Epilog (`/etc/slurm/epilog.d/99-zervram-scrub.sh`)
Exit code **`42`** indicates persistent device quarantine (residual non-zero bytes or NVML ECC row-remapping failure):

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=',' read -ra GPUS <<< "${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
for dev in "${GPUS[@]}"; do
  [ -z "$dev" ] && continue
  set +e
  /usr/local/bin/zervram-scrub \
    --device "$dev" \
    --key-file /etc/zervram/node.key \
    --chain-file /var/lib/zervram/chain.jsonl \
    --tenant-id "${SLURM_JOB_ACCOUNT:-unknown}" \
    --job-id "${SLURM_JOB_ID:-unknown}" \
    --json >> /var/log/zervram/attestations.jsonl
  rc=$?
  set -e
  if [ "$rc" -eq 42 ]; then
    scontrol update NodeName="$(hostname -s)" State=DRAIN Reason="zervram-quarantine-gpu-${dev}"
    exit 42
  elif [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
done
```

### Backend.AI / Kubernetes Node Cleanup Hook
Invoke `/usr/local/bin/zervram-scrub --device <ordinal> --key-file /etc/zervram/node.key --chain-file /var/lib/zervram/chain.jsonl --nonce <scheduler-nonce> --json` immediately after container teardown before advertising the GPU back to the scheduler; taint/drain the node on exit code `42`.

---

## Tests & Examples

```bash
cargo test --workspace
cargo check --workspace --examples
cargo run --example furiosa_rngd_sriov
```