# Zervram (`zervram-open` v0.4.0)

**Interactive Live Console Demo:** https://zervram.netlify.app

Zervram is an open-source, pure-Rust GPU VRAM sanitization daemon (`zervramd`), CLI (`zervram-scrub`), and HAL (`zervram-hal`) with Ed25519 v2 cryptographic attestation. It compiles with **zero compile-time C/C++ GPU SDK headers**, dynamically loads NVIDIA CUDA (`libcuda.so.1` / `nvcuda.dll`), NVIDIA NVML (`libnvidia-ml.so.1` / `nvml.dll`), and AMD ROCm HIP (`libamdhip64.so` / `amdhip64.dll`) at runtime, enforces a pre-scrub active-process gate, sweeps up to **99%** of free physical VRAM down a chunk ladder until OOM, verifies **100% of swept VRAM** via an embedded **PTX 64-bit bitwise-OR reduction kernel** plus `0xAA`-poisoned CSPRNG canary page read-backs, and appends signed `ZeroizationCertificateV2` records to a file-locked persistent SHA-256 hash chain (`chain.jsonl`).

> **Runtime Driver Requirement:** Zero compile-time SDK headers are needed to build `zervramd` / `zervram-scrub`, but **runtime execution on physical GPUs requires the proprietary NVIDIA CUDA + NVML driver or AMD ROCm HIP runtime**.
>
> **Backend Maturity:** `cuda-native` (CUDA + NVML + PTX reduction), `rocm-native` (HIP + 256-page CSPRNG read-back), and `mock-gpu` (CPU simulation) are production-implemented. `furiosa-native` and `tt-metalium-native` are explicitly marked **`experimental-stub`**.

---

## Verified Hardware Evidence (NVIDIA Tesla T4 Live Run)

Validated on physical **16 GiB NVIDIA Tesla T4 (GDDR6)** hardware via `zervram-scrub` + `verify-chain`:
- **Swept & Verified:** `15.37 GB` (`100%` of swept VRAM verified via `verification_mode: "gpu_ptx_full_reduction"` + 64 CSPRNG `0xAA`-poisoned canary pages).
- **Attestation Chain:** Multi-run persistent `chain.jsonl` audited end-to-end with pinned Ed25519 public key -> **`CHAIN VERIFICATION PASSED`**.
- **Warm Daemon (`zervramd`) vs. Cold CLI:**
  - **Cold CLI (`zervram-scrub` direct):** ~508 ms measured on Tesla T4 (includes process startup, driver init, chunk sweep, PTX verification, fsync, and Ed25519 signing).
  - **Warm Daemon (`zervramd` + `zervram-scrub --daemon-url http://127.0.0.1:9740`):** Keeps the initialized backend, Ed25519 key, and per-device async mutexes resident on `127.0.0.1:9740` with optional bearer-token auth (`--token-file` / `ZERVRAM_DAEMON_TOKEN`), returning HTTP `423 Locked` (CLI exit code **`42`**) if a GPU is quarantined.

---

## Threat Model & Honest Compliance Positioning

- **Global HBM/GDDR Persistence vs. `CVE-2023-4969` (`LeftoverLocals`):** `LeftoverLocals` targets local/shared SRAM in OpenCL/Vulkan/Metal drivers. `zervram-scrub` mitigates **residual global HBM/GDDR page persistence** and memory-pool reuse across multi-tenant containers and Slurm jobs by zeroizing up to 99% of free VRAM and verifying 100% of swept bytes before scheduling the next tenant.
- **Compliance Positioning:** Designed for **multi-tenant data hygiene, defense-in-depth, and tamper-evident sanitization audit trails** supporting **SOC 2**, **ISO 27001**, **HIPAA**, and enterprise security questionnaires. Reference TPP helpers in `zervram-hal::catalog` align with U.S. BIS 15 CFR 744.24 (12,000,000 TPP ceiling), though neither 744.24 nor 743.10 mandates GPU VRAM zeroization or Ed25519 scrub certificates.

---

## Quick Start (`zervramd` + `zervram-scrub`)

Build both binaries (Rust 1.82+, zero SDK headers):

```bash
cargo build --release --bin zervramd --bin zervram-scrub
```

### 1. Run Warm Daemon (`zervramd`) & Fast Client (`--daemon-url`)

```bash
# Start persistent localhost daemon (add --allow-mock on CPU-only dev laptops)
zervramd --listen 127.0.0.1:9740 \
  --state-dir /var/lib/zervram \
  --key-file /etc/zervram/node.key \
  --token-file /etc/zervram/daemon.token

# Trigger fast warm scrub (with automatic direct fallback if daemon is unreachable)
zervram-scrub --device 0 \
  --daemon-url http://127.0.0.1:9740 \
  --fallback-direct \
  --token-file /etc/zervram/daemon.token \
  --nonce "$JOB_NONCE" --json --out-cert cert.json

# Health check
curl -H "Authorization: Bearer $(cat /etc/zervram/daemon.token)" http://127.0.0.1:9740/v1/health
```

### 2. Fail-Closed Offline Verification (`verify` & `verify-chain`)

```bash
zervram-scrub verify cert.json --key-file /etc/zervram/node.key --expected-nonce "$JOB_NONCE"
zervram-scrub verify-chain --chain-file /var/lib/zervram/chain.jsonl --key-file /etc/zervram/node.key
```

---

## Turnkey Production Orchestrator Kits (`deploy/`)

| Target | Ready-to-Deploy Artifact | 1-Command Quickstart |
|---|---|---|
| **Systemd** | `deploy/systemd/zervramd.service` | `sudo install -m 0644 deploy/systemd/zervramd.service /etc/systemd/system/ && sudo systemctl enable --now zervramd` |
| **Slurm** | `deploy/slurm/epilog_zervram.sh` | `sudo install -m 0755 deploy/slurm/epilog_zervram.sh /etc/slurm/epilog.d/99-zervram.sh` (auto-drains node via `scontrol` on exit `42` or error) |
| **Backend.AI** | `deploy/backendai/clean_kernel_hook.py` | Import `scrub_kernel_devices_async` in `DockerAgent.clean_kernel` right before `alloc_map.py` frees GPU slots |
| **Docker / K8s** | `Dockerfile` + `deploy/kubernetes/daemonset.yaml` | `docker build -t zervram:0.4.0 .` / `kubectl apply -f deploy/kubernetes/daemonset.yaml` |

---

## Tests

```bash
cargo test --workspace
cargo check --workspace --bins --examples
```
