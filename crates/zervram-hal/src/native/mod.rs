#[cfg(feature = "cuda-native")]
pub mod cuda;
#[cfg(feature = "furiosa-native")]
pub mod furiosa;
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

/// Probes real physical GPU drivers at runtime (CUDA `libcuda.so.1`/`nvcuda.dll` first, then
/// AMD ROCm `libamdhip64.so`/`amdhip64.dll`). Fails closed if no physical GPU driver or device
/// is available; never silently falls back to `mock-gpu`.
pub fn connect_physical_backend() -> Result<Arc<dyn AcceleratorDevice>, HalError> {
    #[allow(unused_mut)]
    let mut diagnostics: Vec<String> = Vec::new();

    #[cfg(feature = "cuda-native")]
    match cuda::CudaAcceleratorBackend::connect(cuda::CudaDriverConfig::default()) {
        Ok(backend) => {
            if backend.device_count() > 0 {
                return Ok(Arc::new(backend));
            }
            diagnostics.push("cuda-native: driver loaded but 0 CUDA GPUs reported".to_string());
        }
        Err(err) => diagnostics.push(format!("cuda-native: {err}")),
    }

    #[cfg(feature = "rocm-native")]
    match rocm::RocmAcceleratorBackend::connect(rocm::RocmDriverConfig::default()) {
        Ok(backend) => {
            if backend.device_count() > 0 {
                return Ok(Arc::new(backend));
            }
            diagnostics.push("rocm-native: runtime loaded but 0 HIP GPUs reported".to_string());
        }
        Err(err) => diagnostics.push(format!("rocm-native: {err}")),
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

/// Safe shared helpers for real physical VRAM sweeping and 4KB page verification planning.
#[cfg(any(feature = "cuda-native", feature = "rocm-native", test))]
pub(crate) mod sweep_plan {
    use std::collections::BTreeSet;

    use crate::catalog::hardware_spec;
    use crate::error::HalError;
    use crate::model::{AcceleratorModel, AcceleratorVendor, VRAM_PAGE_SIZE_BYTES};

    pub const DEFAULT_SWEEP_RATIO_PCT: u8 = 90;
    pub const DEFAULT_VERIFY_SAMPLE_PAGES: usize = 64;

    /// Ordered descending chunk sizes (256 MiB -> 64 MiB -> 16 MiB -> 4 MiB -> 1 MiB -> 4 KiB)
    /// used to safely cover ~90% of free physical VRAM even under allocator fragmentation.
    pub const SWEEP_CHUNK_LADDER_BYTES: &[usize] = &[
        256 * 1024 * 1024,
        64 * 1024 * 1024,
        16 * 1024 * 1024,
        4 * 1024 * 1024,
        1024 * 1024,
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
        let pct = usize::from(sweep_ratio_pct.clamp(1, 98));
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

    /// Selects up to `max_pages` 4KB-aligned `(chunk_index, byte_offset_within_chunk)` coordinates
    /// spanning boundary pages and evenly stratified interior pages across all live swept chunks.
    pub fn select_chunk_verification_targets(
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
        if sample_count == 1 {
            global_indices.insert(0);
        } else {
            let max_idx = total_pages.saturating_sub(1);
            let denom = sample_count.saturating_sub(1);
            for step in 0..sample_count {
                let idx = (step as u128)
                    .saturating_mul(max_idx as u128)
                    / (denom as u128);
                global_indices.insert(usize::try_from(idx).unwrap_or(0));
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

    pub fn infer_model_from_name(
        name: &str,
        vendor: AcceleratorVendor,
    ) -> (AcceleratorModel, u64) {
        let lower = name.to_ascii_lowercase();
        let model = match vendor {
            AcceleratorVendor::Nvidia => {
                if lower.contains("b200") || lower.contains("blackwell") {
                    AcceleratorModel::NvidiaB200Blackwell
                } else {
                    AcceleratorModel::NvidiaH100Sxm
                }
            }
            AcceleratorVendor::Amd => AcceleratorModel::AmdInstinctMi300x,
            AcceleratorVendor::Tenstorrent => AcceleratorModel::TenstorrentBlackholeP150,
            AcceleratorVendor::FuriosaAi => AcceleratorModel::FuriosaRngd,
        };
        let tpp = hardware_spec(model).tpp_per_unit;
        (model, tpp)
    }

    pub fn deterministic_device_uuid(vendor_tag: u8, ordinal: usize, pci_bus_id: &str) -> uuid::Uuid {
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
                    "native runtime session not bound for `{symbol}` via {}",
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
                "vendor runtime library not found on host (probed: {}); pass --allow-mock on systems without datacenter accelerators",
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
    fn sweep_target_aligns_90_percent_to_4kb_boundary() -> Result<(), HalError> {
        let free_80g: usize = 80 * 1024 * 1024 * 1024;
        let target = compute_sweep_target_bytes("cuda-native", free_80g, 90)?;
        assert_eq!(target % (VRAM_PAGE_SIZE_BYTES as usize), 0);
        assert_eq!(target, 72 * 1024 * 1024 * 1024);

        let odd_free: usize = 100_003;
        let odd_target = compute_sweep_target_bytes("cuda-native", odd_free, 90)?;
        assert_eq!(odd_target % 4096, 0);
        assert!(odd_target <= 90_002);

        assert!(matches!(
            compute_sweep_target_bytes("cuda-native", 2048, 90),
            Err(HalError::NativeDriverFailure { .. })
        ));
        Ok(())
    }

    #[test]
    fn stratified_chunk_sampling_covers_first_and_last_4kb_pages() {
        let chunks = [16 * 1024 * 1024, 8 * 1024 * 1024, 4 * 1024 * 1024];
        let targets = select_chunk_verification_targets(&chunks, 64);
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
    fn infers_hardware_model_from_driver_device_names() {
        let (b200, b200_tpp) =
            infer_model_from_name("NVIDIA B200 SXM", AcceleratorVendor::Nvidia);
        assert_eq!(b200, AcceleratorModel::NvidiaB200Blackwell);
        assert_eq!(b200_tpp, 36_000);

        let (h100, h100_tpp) =
            infer_model_from_name("NVIDIA H100 80GB HBM3", AcceleratorVendor::Nvidia);
        assert_eq!(h100, AcceleratorModel::NvidiaH100Sxm);
        assert_eq!(h100_tpp, 15_832);

        let (mi300x, mi300x_tpp) =
            infer_model_from_name("AMD Instinct MI300X", AcceleratorVendor::Amd);
        assert_eq!(mi300x, AcceleratorModel::AmdInstinctMi300x);
        assert_eq!(mi300x_tpp, 20_880);
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