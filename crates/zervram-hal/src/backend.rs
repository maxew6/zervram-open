use async_trait::async_trait;
use uuid::Uuid;

use crate::error::HalError;
use crate::model::{BackendKind, DeviceDescriptor, MemoryRange, PageSample, ZeroizeReceipt};

/// Unified asynchronous Hardware Abstraction Layer across `mock-gpu` and native vendor runtimes.
/// Object-safe (`Arc<dyn AcceleratorDevice>`) for dependency injection into `zervram-scrub` and `zervram-k8s`.
#[async_trait]
pub trait AcceleratorDevice: Send + Sync + 'static {
    fn backend_kind(&self) -> BackendKind;

    async fn discover_devices(&self) -> Result<Vec<DeviceDescriptor>, HalError>;

    async fn get_device(&self, device_uuid: Uuid) -> Result<DeviceDescriptor, HalError>;

    /// Reserves an `Available` device/slice, tainting its backing memory with tenant job context.
    async fn allocate_slice(
        &self,
        device_uuid: Uuid,
        tenant_id_hash: String,
        job_id: String,
    ) -> Result<MemoryRange, HalError>;

    /// Invoked immediately on container/job exit. Transitions device to `DirtyPendingScrub`
    /// and returns the released VRAM range requiring cryptographic zeroization.
    async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError>;

    /// Asynchronously zeroizes the target VRAM address range.
    /// - `mock-gpu`: deterministic `zeroize::Zeroize` across mapped 4KB page buffers (<3ms)
    /// - `cuda-native`: `cuMemsetD8Async` + stream sync
    /// - `rocm-native`: `hipMemsetAsync` + stream sync
    /// - `tt-metalium-native`: TT-Metalium DRAM/L1 buffer zeroing
    /// - `furiosa-native`: RNGD SR-IOV HBM3/SRAM zeroing
    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError>;

    /// Reads back requested 4KB-aligned device pages for `0x00` verification prior to attestation.
    async fn read_back_pages(
        &self,
        device_uuid: Uuid,
        page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError>;

    /// Returns all mapped 4KB page addresses for deterministic full-buffer verification in `mock-gpu`
    /// (returns `None` on native hardware where stratified random spot sampling is used).
    async fn mapped_verification_pages(
        &self,
        device_uuid: Uuid,
        range: MemoryRange,
    ) -> Result<Option<Vec<u64>>, HalError>;

    /// Marks device `Available` ONLY after `verified_zero == true` and certificate persistence.
    async fn mark_available(
        &self,
        device_uuid: Uuid,
        cert_id: Uuid,
        scrub_duration_us: u64,
    ) -> Result<(), HalError>;

    /// Permanently removes a device slice from scheduling if residual non-zero pages are detected.
    async fn quarantine_device(
        &self,
        device_uuid: Uuid,
        reason: String,
    ) -> Result<(), HalError>;
}

/// Alias preserving compatibility with `ARCHITECTURE.md` references to `AcceleratorBackend`.
pub use AcceleratorDevice as AcceleratorBackend;
