# Zervram (`zervram-open` v0.2.0)

**Interactive Live Console Demo:** https://zervram.netlify.app

Zervram is an open-source, pure-Rust cryptographic GPU VRAM sanitization and attestation toolkit (`zervram-hal` + `zervram-scrub`). It dynamically loads native NVIDIA CUDA (`libcuda.so.1` / `nvcuda.dll`) and AMD ROCm HIP (`libamdhip64.so` / `amdhip64.dll`) drivers at runtime with **zero compile-time C/C++ GPU SDK headers**, sweeps free physical GPU VRAM in safe descending chunks, reads back 4 KiB pages to verify `0x00`, and mints Ed25519 + SHA-256 hash-chained `ZeroizationCertificate` attestations using a persistent `node.key`.

---

## Threat Model & Honest Physics

### Global HBM/GDDR Persistence vs. CVE-2023-4969 (`LeftoverLocals`)
- **What `CVE-2023-4969` (`LeftoverLocals`) actually is:** An architectural vulnerability in *local/shared* GPU memory (`__local` / threadgroup SRAM) affecting specific OpenCL, Vulkan, and Metal drivers (Apple, AMD, Qualcomm, Imagination). It does **not** directly exploit NVIDIA CUDA global HBM3.
- **What `zervram-scrub` actually mitigates on CUDA & ROCm datacenter GPUs:** Uninitialized global HBM/GDDR page persistence and driver/framework memory-pool reuse between multi-tenant containers or Slurm jobs. When GPU memory is freed without an explicit zeroization pass, residual model weights, LoRA adapters, KV-cache activations, and prompt embeddings can survive in physical HBM pages allocated to subsequent workloads.
- **How `zervram-scrub` sweeps safely:** On container/job exit, `zervram-hal` queries free physical VRAM (`cuMemGetInfo_v2` / `hipMemGetInfo`), allocates ~90% of free VRAM in descending step-down chunks (`256 MiB -> 64 MiB -> 16 MiB -> 4 MiB -> 1 MiB`) while holding all chunks concurrently so the driver cannot hand back the same physical slab repeatedly, queues asynchronous hardware memsets (`cuMemsetD32Async` / `hipMemsetAsync`), synchronizes once (`cuCtxSynchronize` / `hipDeviceSynchronize`), spot-reads up to 64 stratified 4 KiB pages via `cuMemcpyDtoH_v2` / `hipMemcpy` to verify every sampled byte equals `0x00`, frees all allocations via RAII guards, and signs a `ZeroizationCertificate`.

### Realistic Hardware Bandwidth Physics
VRAM zeroization latency is strictly bounded by physical memory bandwidth (`bytes_swept / sustained_write_bandwidth`):

| Target Hardware / Slice | Swept VRAM (~90%) | Sustained Memset Bandwidth | Realistic Scrub Duration |
|---|---:|---:|---:|
| **4-6 GiB SR-IOV / MIG Slice** (e.g., FuriosaAI RNGD 1/8 slice, H100 MIG) | ~3.6 - 5.4 GiB | ~2.5 - 3.0 TB/s | **< 2 ms** |
| **80 GiB NVIDIA H100 SXM (HBM3)** | ~72 GiB | ~2.8 - 3.1 TB/s | **~25 ms** |
| **15-16 GiB GDDR6 GPU** (e.g., L4 / T4 / RTX workstation) | ~13.5 - 14.4 GiB | ~300 - 340 GB/s | **~45 ms** |
| **Software `mock-gpu`** (CPU CI/dev test pool, 512 KiB backing pages) | 512 KiB RAM | Host DRAM | **< 3 ms** |

---

## Quick Start: `zervram-scrub` CLI

Build release binary (requires only standard Rust 1.82+; zero CUDA/ROCm SDK or OpenSSL headers):

```bash
cargo build --release --bin zervram-scrub
```

### 1. Physical GPU Sanitization (Fail-Closed Default)
Probes `libcuda.so.1` / `nvcuda.dll` and `libamdhip64.so` at runtime. Refuses to fake success if no physical GPU driver is present:

```bash
zervram-scrub --device 0 --key-file /etc/zervram/node.key --json
```

### 2. Local Laptop / CI Simulation (`--allow-mock`)
Explicitly opt into `mock-gpu` when testing without datacenter GPUs:

```bash
cargo run --release --bin zervram-scrub -- --device 0 --allow-mock --key-file node.key --json --out-cert cert.json
```

### 3. Offline Cryptographic Verification
Certificates remain verifiable across reboots because the Ed25519 PKCS#8 private key is persisted to `node.key`:

```bash
zervram-scrub verify cert.json --key-file node.key
```

---

## Orchestrator Integrations

### Backend.AI (`AbstractAgent.clean_kernel`)
Invoke `zervram-scrub` as a subprocess right after stopping a GPU kernel container before returning the device slot to the scheduler:

```python
import asyncio, json

async def scrub_gpu_after_kernel(device_ordinal: int, tenant_id: str, kernel_id: str) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "/usr/local/bin/zervram-scrub",
        "--device", str(device_ordinal),
        "--key-file", "/etc/zervram/node.key",
        "--tenant-id", tenant_id,
        "--job-id", kernel_id,
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"GPU {device_ordinal} sanitization failed: {stderr.decode()}")
    return json.loads(stdout.decode())
```

### Slurm Epilog (`/etc/slurm/epilog.d/99-zervram-scrub.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=',' read -ra GPUS <<< "${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
for dev in "${GPUS[@]}"; do
  [ -z "$dev" ] && continue
  /usr/local/bin/zervram-scrub \
    --device "$dev" \
    --key-file /etc/zervram/node.key \
    --tenant-id "${SLURM_JOB_ACCOUNT:-unknown}" \
    --job-id "${SLURM_JOB_ID:-unknown}" \
    --json >> /var/log/zervram/attestations.jsonl
done
```

### Kubernetes Hook / Device Plugin Post-Stop
Run `zervram-scrub --device <ordinal> --key-file /var/lib/zervram/node.key --json` inside your node-level CRI/device-plugin cleanup hook before advertising the GPU slice back to kubelet.

---

## Examples & Tests

```bash
cargo test --workspace
cargo run --example furiosa_rngd_sriov
```