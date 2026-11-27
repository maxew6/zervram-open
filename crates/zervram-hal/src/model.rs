use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::HalError;

pub const VRAM_PAGE_SIZE_BYTES: u32 = 4_096;

/// Non-zero poison byte written into host read-back buffers before DMA copy (`cuMemcpyDtoH_v2` /
/// `hipMemcpy`) so a broken driver or no-op shim returning success without writing cannot pass.
pub const READBACK_POISON_BYTE: u8 = 0xAA;

/// Computes a domain-separated SHA-256 hex digest over all `(device_ptr, byte_len)` swept chunks.
pub fn compute_chunk_table_sha256(chunks: &[(u64, u64)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"zervram:chunk-table:v2|");
    let count_u64 = u64::try_from(chunks.len()).unwrap_or(u64::MAX);
    hasher.update(count_u64.to_be_bytes());
    for &(ptr, len) in chunks {
        hasher.update(ptr.to_be_bytes());
        hasher.update(len.to_be_bytes());
    }
    hex::encode(hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorVendor {
    Nvidia,
    Amd,
    Tenstorrent,
    FuriosaAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorModel {
    NvidiaT4,
    NvidiaL4,
    NvidiaL40s,
    NvidiaA100,
    NvidiaH100Sxm,
    NvidiaH200Sxm,
    NvidiaB200Blackwell,
    AmdInstinctMi300x,
    AmdInstinctMi325x,
    AmdInstinctMi350x,
    TenstorrentBlackholeP150,
    FuriosaRngd,
    FuriosaRngdSriovSlice,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    MockCpu,
    CudaNative,
    RocmNative,
    TtMetaliumNative,
    FuriosaNative,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MockCpu => "mock_cpu",
            Self::CudaNative => "cuda_native",
            Self::RocmNative => "rocm_native",
            Self::TtMetaliumNative => "tt_metalium_native",
            Self::FuriosaNative => "furiosa_native",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationMode {
    /// 100% GPU-side PTX 64-bit bitwise-OR reduction across all swept chunks plus CSPRNG canary pages.
    GpuPtxFullReduction,
    /// CSPRNG-random 256-page host read-back fallback when GPU JIT kernel verification is unavailable.
    #[serde(rename = "csprng_sampled_256_pages")]
    CsprngSampled256Pages,
    /// CSPRNG-random host page read-back with custom sample count.
    CsprngSampledPages,
    /// Full host-RAM verification over all mapped pages in software `mock-gpu` mode.
    MockAllMapped,
}

impl VerificationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GpuPtxFullReduction => "gpu_ptx_full_reduction",
            Self::CsprngSampled256Pages => "csprng_sampled_256_pages",
            Self::CsprngSampledPages => "csprng_sampled_pages",
            Self::MockAllMapped => "mock_all_mapped",
        }
    }
}

/// Result of the GPU-side PTX 64-bit bitwise-OR reduction pass across 100% of swept VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuReductionVerification {
    pub verified_bytes: u64,
    pub words_scanned: u64,
    pub expected_words: u64,
    pub non_zero_or_mask: u64,
}

impl GpuReductionVerification {
    pub fn is_clean(&self) -> bool {
        self.words_scanned > 0
            && self.words_scanned == self.expected_words
            && self.non_zero_or_mask == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareSpec {
    pub model: AcceleratorModel,
    pub vendor: AcceleratorVendor,
    pub canonical_name: &'static str,
    pub tpp_per_unit: Option<u64>,
    pub mac_tops: f64,
    pub bit_length: u16,
    pub vram_total_bytes: u64,
    pub page_size_bytes: u32,
    pub max_units_under_12m_tpp: Option<u32>,
    pub sriov_slice_count: u8,
    pub k8s_resource_keys: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRange {
    pub start_addr: u64,
    pub byte_len: u64,
    pub page_size_bytes: u32,
}

impl MemoryRange {
    pub fn new(start_addr: u64, byte_len: u64) -> Result<Self, HalError> {
        let page_size = u64::from(VRAM_PAGE_SIZE_BYTES);
        if byte_len == 0
            || start_addr % page_size != 0
            || byte_len % page_size != 0
            || start_addr.checked_add(byte_len).is_none()
        {
            return Err(HalError::InvalidMemoryRange {
                start_addr,
                byte_len,
                page_size: VRAM_PAGE_SIZE_BYTES,
            });
        }
        Ok(Self {
            start_addr,
            byte_len,
            page_size_bytes: VRAM_PAGE_SIZE_BYTES,
        })
    }

    pub fn page_count(&self) -> u64 {
        let page_size = u64::from(self.page_size_bytes);
        if page_size == 0 {
            return 0;
        }
        self.byte_len / page_size
    }

    pub fn end_addr(&self) -> Option<u64> {
        self.start_addr.checked_add(self.byte_len)
    }

    pub fn contains_page(&self, page_addr: u64) -> bool {
        let page_size = u64::from(self.page_size_bytes);
        if page_size == 0 || page_addr % page_size != 0 {
            return false;
        }
        match (page_addr.checked_add(page_size), self.end_addr()) {
            (Some(page_end), Some(range_end)) => {
                page_addr >= self.start_addr && page_end <= range_end
            }
            _ => false,
        }
    }

    pub fn contains_range(&self, other: &Self) -> bool {
        if other.byte_len == 0
            || self.page_size_bytes == 0
            || other.page_size_bytes != self.page_size_bytes
        {
            return false;
        }
        let page_size = u64::from(self.page_size_bytes);
        if other.start_addr % page_size != 0 || other.byte_len % page_size != 0 {
            return false;
        }
        match (self.end_addr(), other.end_addr()) {
            (Some(self_end), Some(other_end)) => {
                other.start_addr >= self.start_addr && other_end <= self_end
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSample {
    pub page_addr: u64,
    pub data: Box<[u8; VRAM_PAGE_SIZE_BYTES as usize]>,
}

impl PageSample {
    pub fn is_all_zero(&self) -> bool {
        self.data.iter().all(|&b| b == 0x00)
    }

    pub fn non_zero_byte_count(&self) -> usize {
        self.data.iter().filter(|&&b| b != 0x00).count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DeviceState {
    Available,
    Allocated {
        allocation_id: Uuid,
        tenant_id_hash: String,
        job_id: String,
        allocated_at: DateTime<Utc>,
    },
    DirtyPendingScrub {
        last_tenant_hash: String,
        released_at: DateTime<Utc>,
        estimated_dirty_ratio_pct: u8,
    },
    Scrubbing {
        started_at: DateTime<Utc>,
    },
    Quarantined {
        reason: String,
        quarantined_at: DateTime<Utc>,
    },
}

impl DeviceState {
    pub fn status_label(&self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Allocated { .. } => "allocated",
            Self::DirtyPendingScrub { .. } => "dirty_pending_scrub",
            Self::Scrubbing { .. } => "scrubbing",
            Self::Quarantined { .. } => "quarantined",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    pub device_uuid: Uuid,
    pub device_ordinal: u32,
    pub node_name: String,
    pub pci_bus_id: String,
    pub vendor: AcceleratorVendor,
    pub model: AcceleratorModel,
    pub vendor_model_label: String,
    /// `Some(0..=7)` for FuriosaAI RNGD SR-IOV slices in simulation; `None` for whole physical cards.
    pub sriov_slice_index: Option<u8>,
    pub sriov_parent_uuid: Option<Uuid>,
    /// Reference TPP rating when the detected SKU matches a catalog entry; `None` for `Unknown`.
    pub tpp_rating: Option<u64>,
    pub vram_total_bytes: u64,
    pub state: DeviceState,
    pub last_cert_id: Option<Uuid>,
    pub last_scrub_duration_us: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZeroizeReceipt {
    pub device_uuid: Uuid,
    pub range: MemoryRange,
    pub duration_us: u64,
    pub pages_zeroed: u64,
    pub backend_kind: BackendKind,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub swept_bytes: u64,
    pub chunk_count: u32,
    pub chunk_table_sha256: String,
    pub active_processes_before: u32,
    pub verification_mode: VerificationMode,
    pub gpu_reduction: Option<GpuReductionVerification>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_range_validates_alignment_and_overflow() -> Result<(), HalError> {
        let range = MemoryRange::new(0x1_0000_0000, 4_096 * 16)?;
        assert_eq!(range.page_count(), 16);
        assert_eq!(range.end_addr(), Some(0x1_0000_0000 + 4_096 * 16));
        assert!(range.contains_page(0x1_0000_0000));
        assert!(range.contains_page(0x1_0000_0000 + 4_096 * 15));
        assert!(!range.contains_page(0x1_0000_0000 + 4_096 * 16));
        assert!(!range.contains_page(0x1_0000_0001));

        let sub = MemoryRange::new(0x1_0000_0000 + 4_096, 4_096 * 4)?;
        assert!(range.contains_range(&sub));

        assert!(matches!(
            MemoryRange::new(0x1_0000_0000, 0),
            Err(HalError::InvalidMemoryRange { .. })
        ));
        assert!(matches!(
            MemoryRange::new(0x1_0000_0001, 4_096),
            Err(HalError::InvalidMemoryRange { .. })
        ));
        assert!(matches!(
            MemoryRange::new(0x1_0000_0000, 4_097),
            Err(HalError::InvalidMemoryRange { .. })
        ));
        assert!(matches!(
            MemoryRange::new(u64::MAX - 4_095, 8_192),
            Err(HalError::InvalidMemoryRange { .. })
        ));
        Ok(())
    }

    #[test]
    fn page_sample_detects_non_zero_and_poison_bytes() {
        let mut data = Box::new([0u8; VRAM_PAGE_SIZE_BYTES as usize]);
        let clean = PageSample {
            page_addr: 0x1_0000_0000,
            data: data.clone(),
        };
        assert!(clean.is_all_zero());
        assert_eq!(clean.non_zero_byte_count(), 0);

        let poisoned = PageSample {
            page_addr: 0x1_0000_0000,
            data: Box::new([READBACK_POISON_BYTE; VRAM_PAGE_SIZE_BYTES as usize]),
        };
        assert!(!poisoned.is_all_zero());
        assert_eq!(
            poisoned.non_zero_byte_count(),
            VRAM_PAGE_SIZE_BYTES as usize
        );

        if let Some(slot) = data.get_mut(17) {
            *slot = 0xDE;
        }
        if let Some(slot) = data.get_mut(4_095) {
            *slot = 0xAD;
        }
        let dirty = PageSample {
            page_addr: 0x1_0000_0000,
            data,
        };
        assert!(!dirty.is_all_zero());
        assert_eq!(dirty.non_zero_byte_count(), 2);
    }

    #[test]
    fn chunk_table_sha256_changes_on_chunk_mutation() {
        let a = compute_chunk_table_sha256(&[(0x1000_0000, 4096), (0x2000_0000, 8192)]);
        let b = compute_chunk_table_sha256(&[(0x1000_0000, 4096), (0x2000_0000, 12288)]);
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
    }

    #[test]
    fn serde_wire_tags_match_schema() -> Result<(), Box<dyn std::error::Error>> {
        let state_json = serde_json::to_value(DeviceState::Available)?;
        assert_eq!(
            state_json.get("status"),
            Some(&serde_json::json!("available"))
        );

        let vendor_json = serde_json::to_value(AcceleratorVendor::FuriosaAi)?;
        assert_eq!(vendor_json, serde_json::json!("furiosa_ai"));

        let model_json = serde_json::to_value(AcceleratorModel::NvidiaT4)?;
        assert_eq!(model_json, serde_json::json!("nvidia_t4"));

        let mode_json = serde_json::to_value(VerificationMode::CsprngSampled256Pages)?;
        assert_eq!(mode_json, serde_json::json!("csprng_sampled_256_pages"));

        let backend_json = serde_json::to_value(BackendKind::CudaNative)?;
        assert_eq!(backend_json, serde_json::json!("cuda_native"));
        Ok(())
    }
}