use async_trait::async_trait;
use uuid::Uuid;

use crate::backend::AcceleratorDevice;
use crate::error::HalError;
use crate::model::{BackendKind, DeviceDescriptor, MemoryRange, PageSample, ZeroizeReceipt};
use crate::native::common::NativeBackendCore;

const FURIOSA_BACKEND_NAME: &str = "furiosa-native";
const FURIOSA_DRIVER_CANDIDATES: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu/libfuriosa_hal.so",
    "/usr/local/lib/libfuriosa_hal.so",
];

#[derive(Debug, Clone)]
pub struct FuriosaDriverConfig {
    pub runtime_library_path: Option<String>,
    pub enable_sriov_8_slice_partitioning: bool,
}

impl Default for FuriosaDriverConfig {
    fn default() -> Self {
        Self {
            runtime_library_path: None,
            enable_sriov_8_slice_partitioning: true,
        }
    }
}

/// Native FuriosaAI RNGD SR-IOV HBM3/SRAM slice sanitization backend (8x 1,024-TPP slices).
#[derive(Debug)]
pub struct FuriosaAcceleratorBackend {
    pub config: FuriosaDriverConfig,
    core: NativeBackendCore,
}

impl FuriosaAcceleratorBackend {
    pub fn connect(config: FuriosaDriverConfig) -> Result<Self, HalError> {
        let core = NativeBackendCore::probe_and_init(
            FURIOSA_BACKEND_NAME,
            BackendKind::FuriosaNative,
            config.runtime_library_path.as_deref(),
            FURIOSA_DRIVER_CANDIDATES,
        )?;
        Ok(Self { config, core })
    }

    pub fn driver_path(&self) -> &str {
        &self.core.driver_path
    }
}

#[async_trait]
impl AcceleratorDevice for FuriosaAcceleratorBackend {
    fn backend_kind(&self) -> BackendKind {
        self.core.backend_kind
    }

    async fn discover_devices(&self) -> Result<Vec<DeviceDescriptor>, HalError> {
        Ok(self.core.discover().await)
    }

    async fn get_device(&self, device_uuid: Uuid) -> Result<DeviceDescriptor, HalError> {
        self.core.get(device_uuid).await
    }

    async fn allocate_slice(
        &self,
        device_uuid: Uuid,
        _tenant_id_hash: String,
        _job_id: String,
    ) -> Result<MemoryRange, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("furiosa_sriov_acquire_pe"))
    }

    async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("furiosa_sriov_drain_pe"))
    }

    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("furiosa_sriov_zeroize_hbm3"))
    }

    async fn read_back_pages(
        &self,
        device_uuid: Uuid,
        _page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("furiosa_sriov_read_pages"))
    }

    async fn mapped_verification_pages(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<Option<Vec<u64>>, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Ok(None)
    }

    async fn mark_available(
        &self,
        device_uuid: Uuid,
        cert_id: Uuid,
        scrub_duration_us: u64,
    ) -> Result<(), HalError> {
        self.core
            .mark_available(device_uuid, cert_id, scrub_duration_us)
            .await
    }

    async fn quarantine_device(
        &self,
        device_uuid: Uuid,
        reason: String,
    ) -> Result<(), HalError> {
        self.core.quarantine(device_uuid, reason).await
    }
}
