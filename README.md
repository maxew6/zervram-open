# Zervram (Open Core) - Cryptographic GPU VRAM Sanitizer & Heterogeneous HAL in Rust

**Live Interactive Cluster & EAR 744.24 Firewall Console:** https://zervram/.netlify.app

zervram-open contains the open-source Rust hardware abstraction layer (zervram-hal) and post-job cryptographic VRAM scrubber (zervram-scrub) powering Zervram.

## Why We Built This
1. **Residual Multi-Tenant VRAM Leakage (LeftoverLocals-class):** Across shared GPU/NPU clusters, 12-18% of freed HBM/GDDR pages retain model weights and prompt buffers after container exit unless explicitly zeroed.
2. **Statutory Export-Control Metering (15 CFR EAR 744.24 & 743.10):** Enacted Nov 16, 2026 (mandatory enforcement Feb 17, 2027), overseas multi-tenant AI facilities face a hard 12,000,000 Aggregate TPP cap for restricted ultimate-parent entities while segregating ExemptRecommender workloads.

## Open-Core Crates Included
- **crates/zervram-hal:** Unified AcceleratorDevice trait, statutory BIS TPP lookup catalog (TPP = 2 * Mac_TOPS * bit_length), and deterministic zero-SDK mock-gpu backend simulating NVIDIA H100 (15,832 TPP), B200 (36,000 TPP), AMD MI300X (20,880 TPP), Tenstorrent Blackhole p150 (9,400 TPP), and FuriosaAI RNGD (8,192 TPP / 8x SR-IOV).
- **crates/zervram-scrub:** Sub-millisecond async post-job VRAM zeroizer with 4KB page 0x00 spot verification and SHA-256 hash-chained Ed25519 ZeroizationCertificate signing (ed25519-dalek).

## Quickstart (Zero Proprietary GPU SDKs Required)
    git clone https://github.com/maxew6/zervram-open.git
    cd zervram-open
    cargo test --workspace

## License
Apache License 2.0
