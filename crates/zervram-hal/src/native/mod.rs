#[cfg(feature = "cuda-native")]
pub mod cuda;
#[cfg(feature = "furiosa-native")]
pub mod furiosa;
#[cfg(feature = "cuda-native")]
pub(crate) mod nvml;
#[cfg(feature = "rocm-native")]
pub mod rocm;
#[cfg(feature = "tt-metalium-native")]
pub mod tt_metalium;

use std::sync::Arc;

use crate::backend::AcceleratorDevice;
use crate::error::HalError;
use crate::model::BackendKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBackendPreference {
    Auto,
    Cuda,
    Rocm,
    TtMetalium,
    Furiosa,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicalBackendOptions {
    /// When `true`, bypasses NVML active-process rejection (`--force`).
    pub force_ignore_active_processes: bool,
    /// Optional free-VRAM sweep percentage override (`1..=100`, default `99`).
    pub sweep_free_vram_pct: Option<u8>,
}

/// Returns all native vendor backends compiled into the current binary via Cargo features.
pub fn available_native_backends() -> Vec<BackendKind> {
    #[allow(unused_mut)]
    let mut backends = Vec::new();
    #[cfg(feature = "cuda-native")]
    backends.push(BackendKind::CudaNative);
    #[cfg(feature = "rocm-native")]
    backends.push(BackendKind::RocmNative);
    #[cfg(feature = "tt-metalium-native")]
    backends.push(BackendKind::TtMetaliumNative);
    #[cfg(feature = "furiosa-native")]
    backends.push(BackendKind::FuriosaNative);
    backends
}

/// Probes real physical GPU drivers at runtime with default fail-closed options.
pub fn connect_physical_backend() -> Result<Arc<dyn AcceleratorDevice>, HalError> {
    connect_physical_backend_with_options(PhysicalBackendOptions::default())
}

/// Probes real physical GPU drivers at runtime (CUDA `libcuda.so.1`/`nvcuda.dll` first, then
/// AMD ROCm `libamdhip64.so`/`amdhip64.dll`). Fails closed if no physical GPU driver or device
/// is available; never silently falls back to `mock-gpu`.
pub fn connect_physical_backend_with_options(
    options: PhysicalBackendOptions,
) -> Result<Arc<dyn AcceleratorDevice>, HalError> {
    let _ = options;
    #[allow(unused_mut)]
    let mut diagnostics: Vec<String> = Vec::new();

    #[cfg(feature = "cuda-native")]
    {
        let mut cfg = cuda::CudaDriverConfig {
            force_ignore_active_processes: options.force_ignore_active_processes,
            ..cuda::CudaDriverConfig::default()
        };
        if let Some(pct) = options.sweep_free_vram_pct {
            cfg.sweep_free_vram_pct = pct.clamp(1, 100);
        }
        match cuda::CudaAcceleratorBackend::connect(cfg) {
            Ok(backend) => {
                if backend.device_count() > 0 {
                    return Ok(Arc::new(backend));
                }
                diagnostics.push("cuda-native: driver loaded but 0 CUDA GPUs reported".to_string());
            }
            Err(err) => diagnostics.push(format!("cuda-native: {err}")),
        }
    }

    #[cfg(feature = "rocm-native")]
    {
        let mut cfg = rocm::RocmDriverConfig::default();
        if let Some(pct) = options.sweep_free_vram_pct {
            cfg.sweep_free_vram_pct = pct.clamp(1, 100);
        }
        match rocm::RocmAcceleratorBackend::connect(cfg) {
            Ok(backend) => {
                if backend.device_count() > 0 {
                    return Ok(Arc::new(backend));
                }
                diagnostics
                    .push("rocm-native: runtime loaded but 0 HIP GPUs reported".to_string());
            }
            Err(err) => diagnostics.push(format!("rocm-native: {err}")),
        }
    }

    if diagnostics.is_empty() {
        diagnostics.push("no native GPU backend features compiled in".to_string());
    }

    Err(HalError::NativeDriverFailure {
        backend: "physical-auto",
        message: format!(
            "no physical CUDA or ROCm GPU available ({}); pass --allow-mock to run software simulation",
            diagnostics.join("; ")
        ),
    })
}

/// Safe shared helpers for physical VRAM sweeping, CSPRNG page sampling, and SKU classification.
#[cfg(any(feature = "cuda-native", feature = "rocm-native", test))]
pub(crate) mod sweep_plan {
    use std::collections::BTreeSet;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use crate::catalog::hardware_spec;
    use crate::error::HalError;
    use crate::model::{AcceleratorModel, AcceleratorVendor, VRAM_PAGE_SIZE_BYTES};

    pub const DEFAULT_SWEEP_RATIO_PCT: u8 = 99;
    pub const DEFAULT_CANARY_SAMPLE_PAGES: usize = 64;
    pub const DEFAULT_FALLBACK_SAMPLE_PAGES: usize = 256;

    /// Ordered descending chunk sizes (`256 MiB -> 64 MiB -> 16 MiB -> 4 MiB -> 1 MiB -> 64 KiB -> 4 KiB`)
    /// used to drain free physical VRAM up to OOM even under allocator fragmentation.
    pub const SWEEP_CHUNK_LADDER_BYTES: &[usize] = &[
        256 * 1024 * 1024,
        64 * 1024 * 1024,
        16 * 1024 * 1024,
        4 * 1024 * 1024,
        1024 * 1024,
        64 * 1024,
        VRAM_PAGE_SIZE_BYTES as usize,
    ];

    pub fn align_down_4k(bytes: usize) -> usize {
        let page = VRAM_PAGE_SIZE_BYTES as usize;
        bytes - (bytes % page)
    }

    pub fn compute_sweep_target_bytes(
        backend: &'static str,
        free_bytes: usize,
        sweep_ratio_pct: u8,
    ) -> Result<usize, HalError> {
        let pct = usize::from(sweep_ratio_pct.clamp(1, 100));
        let raw = free_bytes
            .checked_div(100)
            .map(|q| q.saturating_mul(pct))
            .unwrap_or(0)
            .saturating_add((free_bytes % 100).saturating_mul(pct) / 100);
        let aligned = align_down_4k(raw);
        if aligned < (VRAM_PAGE_SIZE_BYTES as usize) {
            return Err(HalError::NativeDriverFailure {
                backend,
                message: format!(
                    "insufficient free VRAM to sweep ({free_bytes} B free, target {aligned} B < 4096 B)"
                ),
            });
        }
        Ok(aligned)
    }

    /// Selects up to `max_pages` 4 KiB-aligned `(chunk_index, byte_offset_within_chunk)` coordinates
    /// covering boundary pages (first & last) plus CSPRNG-random (`ChaCha20Rng::from_entropy()`)
    /// interior pages across all live swept chunks.
    pub fn select_csprng_chunk_verification_targets(
        chunk_lengths: &[usize],
        max_pages: usize,
    ) -> Vec<(usize, usize)> {
        let page_size = VRAM_PAGE_SIZE_BYTES as usize;
        let mut total_pages: usize = 0;
        let mut chunk_page_counts = Vec::with_capacity(chunk_lengths.len());
        for &len in chunk_lengths {
            let count = len / page_size;
            chunk_page_counts.push(count);
            total_pages = total_pages.saturating_add(count);
        }
        if total_pages == 0 || max_pages == 0 {
            return Vec::new();
        }

        let sample_count = max_pages.min(total_pages);
        let mut global_indices = BTreeSet::new();

        if sample_count == total_pages {
            for idx in 0..total_pages {
                global_indices.insert(idx);
            }
        } else if sample_count == 1 {
            global_indices.insert(0);
        } else {
            global_indices.insert(0);
            global_indices.insert(total_pages.saturating_sub(1));

            let mut rng = ChaCha20Rng::from_entropy();
            let max_attempts = sample_count.saturating_mul(8).max(32);
            for _ in 0..max_attempts {
                if global_indices.len() >= sample_count {
                    break;
                }
                let candidate = rng.gen_range(1..total_pages.saturating_sub(1));
                global_indices.insert(candidate);
            }
            if global_indices.len() < sample_count {
                for fallback in 1..total_pages.saturating_sub(1) {
                    if global_indices.len() >= sample_count {
                        break;
                    }
                    global_indices.insert(fallback);
                }
            }
        }

        let mut targets = Vec::with_capacity(global_indices.len());
        for g_idx in global_indices {
            let mut rem = g_idx;
            for (c_idx, &c_pages) in chunk_page_counts.iter().enumerate() {
                if rem < c_pages {
                    targets.push((c_idx, rem.saturating_mul(page_size)));
                    break;
                }
                rem = rem.saturating_sub(c_pages);
            }
        }
        targets
    }

    fn contains_token(haystack: &str, token: &str) -> bool {
        haystack
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|part| part == token)
    }

    /// Accurately classifies real datacenter GPU names into catalog SKUs; returns
    /// `(AcceleratorModel::Unknown, None)` for unrecognized models rather than misclassifying.
    pub fn infer_model_from_name(
        name: &str,
        vendor: AcceleratorVendor,
    ) -> (AcceleratorModel, Option<u64>) {
        let lower = name.to_ascii_lowercase();
        let model = match vendor {
            AcceleratorVendor::Nvidia => {
                if contains_token(&lower, "b200") || lower.contains("blackwell") {
                    AcceleratorModel::NvidiaB200Blackwell
                } else if contains_token(&lower, "h200") {
                    AcceleratorModel::NvidiaH200Sxm
                } else if contains_token(&lower, "h100") {
                    AcceleratorModel::NvidiaH100Sxm
                } else if contains_token(&lower, "a100") {
                    AcceleratorModel::NvidiaA100
                } else if contains_token(&lower, "l40s") {
                    AcceleratorModel::NvidiaL40s
                } else if contains_token(&lower, "l4") {
                    AcceleratorModel::NvidiaL4
                } else if contains_token(&lower, "t4") {
                    AcceleratorModel::NvidiaT4
                } else {
                    AcceleratorModel::Unknown
                }
            }
            AcceleratorVendor::Amd => {
                if contains_token(&lower, "mi350x") {
                    AcceleratorModel::AmdInstinctMi350x
                } else if contains_token(&lower, "mi325x") {
                    AcceleratorModel::AmdInstinctMi325x
                } else if contains_token(&lower, "mi300x") {
                    AcceleratorModel::AmdInstinctMi300x
                } else {
                    AcceleratorModel::Unknown
                }
            }
            AcceleratorVendor::Tenstorrent => {
                if lower.contains("blackhole") || contains_token(&lower, "p150") {
                    AcceleratorModel::TenstorrentBlackholeP150
                } else {
                    AcceleratorModel::Unknown
                }
            }
            AcceleratorVendor::FuriosaAi => {
                if contains_token(&lower, "rngd") {
                    AcceleratorModel::FuriosaRngd
                } else {
                    AcceleratorModel::Unknown
                }
            }
        };
        let tpp = hardware_spec(model).tpp_per_unit;
        (model, tpp)
    }

    pub fn deterministic_device_uuid(
        vendor_tag: u8,
        ordinal: usize,
        pci_bus_id: &str,
    ) -> uuid::Uuid {
        let mut bytes = [0u8; 16];
        if let Some(slot) = bytes.first_mut() {
            *slot = 0x5A; // 'Z'
        }
        if let Some(slot) = bytes.get_mut(1) {
            *slot = vendor_tag;
        }
        let ord_bytes = u32::try_from(ordinal).unwrap_or(0).to_be_bytes();
        for (dst, &src) in bytes.iter_mut().skip(2).zip(ord_bytes.iter()) {
            *dst = src;
        }
        let mut acc: u64 = 0xCBF2_9CE4_8422_2325;
        for &b in pci_bus_id.as_bytes() {
            acc ^= u64::from(b);
            acc = acc.wrapping_mul(0x0000_0100_0000_01B3);
        }
        for (dst, &src) in bytes.iter_mut().skip(8).zip(acc.to_be_bytes().iter()) {
            *dst = src;
        }
        uuid::Builder::from_random_bytes(bytes).into_uuid()
    }
}

#[cfg(any(feature = "tt-metalium-native", feature = "furiosa-native"))]
pub(crate) mod common {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::Arc;

    use chrono::Utc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    use crate::error::HalError;
    use crate::model::{BackendKind, DeviceDescriptor, DeviceState};

    #[derive(Debug)]
    pub(crate) struct NativeBackendCore {
        pub(crate) backend_name: &'static str,
        pub(crate) backend_kind: BackendKind,
        pub(crate) driver_path: String,
        pub(crate) devices: Arc<RwLock<BTreeMap<Uuid, DeviceDescriptor>>>,
    }

    impl NativeBackendCore {
        pub(crate) fn probe_and_init(
            backend_name: &'static str,
            backend_kind: BackendKind,
            explicit_path: Option<&str>,
            candidates: &[&str],
        ) -> Result<Self, HalError> {
            let driver_path = probe_driver_library(backend_name, explicit_path, candidates)?;
            Ok(Self {
                backend_name,
                backend_kind,
                driver_path,
                devices: Arc::new(RwLock::new(BTreeMap::new())),
            })
        }

        pub(crate) fn invocation_error(&self, symbol: &'static str) -> HalError {
            HalError::NativeDriverFailure {
                backend: self.backend_name,
                message: format!(
                    "experimental-stub: vendor FFI bindings for `{symbol}` are not yet implemented (probed {})",
                    self.driver_path
                ),
            }
        }

        pub(crate) async fn discover(&self) -> Vec<DeviceDescriptor> {
            let guard = self.devices.read().await;
            guard.values().cloned().collect()
        }

        pub(crate) async fn get(&self, device_uuid: Uuid) -> Result<DeviceDescriptor, HalError> {
            let guard = self.devices.read().await;
            guard
                .get(&device_uuid)
                .cloned()
                .ok_or(HalError::DeviceNotFound { device_uuid })
        }

        pub(crate) async fn mark_available(
            &self,
            device_uuid: Uuid,
            cert_id: Uuid,
            scrub_duration_us: u64,
        ) -> Result<(), HalError> {
            let mut guard = self.devices.write().await;
            let desc = guard
                .get_mut(&device_uuid)
                .ok_or(HalError::DeviceNotFound { device_uuid })?;
            if !matches!(desc.state, DeviceState::Scrubbing { .. }) {
                return Err(HalError::InvalidDeviceState {
                    device_uuid,
                    expected: "scrubbing",
                    actual: desc.state.status_label().to_string(),
                });
            }
            desc.state = DeviceState::Available;
            desc.last_cert_id = Some(cert_id);
            desc.last_scrub_duration_us = Some(scrub_duration_us);
            Ok(())
        }

        pub(crate) async fn quarantine(
            &self,
            device_uuid: Uuid,
            reason: String,
        ) -> Result<(), HalError> {
            let mut guard = self.devices.write().await;
            let desc = guard
                .get_mut(&device_uuid)
                .ok_or(HalError::DeviceNotFound { device_uuid })?;
            desc.state = DeviceState::Quarantined {
                reason,
                quarantined_at: Utc::now(),
            };
            Ok(())
        }
    }

    pub(crate) fn probe_driver_library(
        backend: &'static str,
        explicit_path: Option<&str>,
        default_candidates: &[&str],
    ) -> Result<String, HalError> {
        if let Some(path) = explicit_path {
            if Path::new(path).exists() {
                return Ok(path.to_string());
            }
            return Err(HalError::NativeDriverFailure {
                backend,
                message: format!("configured driver library path does not exist: {path}"),
            });
        }

        for &candidate in default_candidates {
            if Path::new(candidate).exists() {
                return Ok(candidate.to_string());
            }
        }

        Err(HalError::NativeDriverFailure {
            backend,
            message: format!(
                "experimental-stub vendor runtime library not found on host (probed: {}); pass --allow-mock for software simulation",
                default_candidates.join(", ")
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::sweep_plan::*;
    use crate::error::HalError;
    use crate::model::{AcceleratorModel, AcceleratorVendor, VRAM_PAGE_SIZE_BYTES};

    #[test]
    fn sweep_target_aligns_99_and_100_percent_to_4kb_boundary() -> Result<(), HalError> {
        let free_80g: usize = 80 * 1024 * 1024 * 1024;
        let target_99 = compute_sweep_target_bytes("cuda-native", free_80g, 99)?;
        assert_eq!(target_99 % (VRAM_PAGE_SIZE_BYTES as usize), 0);
        assert!(target_99 > 79 * 1024 * 1024 * 1024);

        let target_100 = compute_sweep_target_bytes("cuda-native", free_80g, 100)?;
        assert_eq!(target_100, free_80g);

        assert!(matches!(
            compute_sweep_target_bytes("cuda-native", 2048, 99),
            Err(HalError::NativeDriverFailure { .. })
        ));
        Ok(())
    }

    #[test]
    fn csprng_chunk_sampling_covers_boundaries_and_is_4kb_aligned() {
        let chunks = [16 * 1024 * 1024, 8 * 1024 * 1024, 4 * 1024 * 1024];
        let targets = select_csprng_chunk_verification_targets(&chunks, 64);
        assert_eq!(targets.len(), 64);
        assert_eq!(targets.first().copied(), Some((0, 0)));
        assert_eq!(
            targets.last().copied(),
            Some((2, 4 * 1024 * 1024 - 4096))
        );
        for &(_c_idx, offset) in &targets {
            assert_eq!(offset % 4096, 0);
        }
    }

    #[test]
    fn infers_all_real_nvidia_and_amd_skus_and_unknown_fallback() {
        assert_eq!(
            infer_model_from_name("Tesla T4", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaT4, Some(2_080))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA L4", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaL4, Some(3_872))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA L40S", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaL40s, Some(11_728))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA A100-SXM4-80GB", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaA100, Some(9_984))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA H100 80GB HBM3", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaH100Sxm, Some(15_832))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA H200", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaH200Sxm, Some(15_832))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA B200 SXM", AcceleratorVendor::Nvidia),
            (AcceleratorModel::NvidiaB200Blackwell, Some(36_000))
        );
        assert_eq!(
            infer_model_from_name("NVIDIA GeForce RTX 4090", AcceleratorVendor::Nvidia),
            (AcceleratorModel::Unknown, None)
        );
        assert_eq!(
            infer_model_from_name("AMD Instinct MI300X", AcceleratorVendor::Amd),
            (AcceleratorModel::AmdInstinctMi300x, Some(20_880))
        );
        assert_eq!(
            infer_model_from_name("AMD Instinct MI325X", AcceleratorVendor::Amd),
            (AcceleratorModel::AmdInstinctMi325x, Some(20_880))
        );
        assert_eq!(
            infer_model_from_name("AMD Instinct MI350X", AcceleratorVendor::Amd),
            (AcceleratorModel::AmdInstinctMi350x, Some(36_800))
        );
        assert_eq!(
            infer_model_from_name("AMD Radeon RX 7900 XTX", AcceleratorVendor::Amd),
            (AcceleratorModel::Unknown, None)
        );
    }

    #[cfg(feature = "cuda-native")]
    #[test]
    fn ptx_kernel_is_nul_terminated_with_valid_isa_7_sm_50_header() {
        let ptx = super::cuda::PTX_VERIFY_ZERO_KERNEL;
        assert_eq!(ptx.last().copied(), Some(0u8));
        let text = String::from_utf8_lossy(ptx);
        assert!(text.contains(".version 7.0"));
        assert!(text.contains(".target sm_50"));
        assert!(text.contains(".address_size 64"));
        assert!(text.contains(".visible .entry zervram_verify_zero_u64"));
    }

    #[cfg(feature = "cuda-native")]
    #[test]
    fn cuda_connect_rejects_missing_explicit_library_cleanly() {
        let cfg = super::cuda::CudaDriverConfig {
            driver_library_path: Some("nonexistent_libcuda_zervram_test.so".to_string()),
            ..super::cuda::CudaDriverConfig::default()
        };
        assert!(matches!(
            super::cuda::CudaAcceleratorBackend::connect(cfg),
            Err(HalError::NativeDriverFailure {
                backend: "cuda-native",
                ..
            })
        ));
    }

    #[cfg(feature = "rocm-native")]
    #[test]
    fn rocm_connect_rejects_missing_explicit_library_cleanly() {
        let cfg = super::rocm::RocmDriverConfig {
            driver_library_path: Some("nonexistent_libamdhip64_zervram_test.so".to_string()),
            ..super::rocm::RocmDriverConfig::default()
        };
        assert!(matches!(
            super::rocm::RocmAcceleratorBackend::connect(cfg),
            Err(HalError::NativeDriverFailure {
                backend: "rocm-native",
                ..
            })
        ));
    }
}