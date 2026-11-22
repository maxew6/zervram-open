use async_trait::async_trait;
use uuid::Uuid;

use crate::backend::AcceleratorDevice;
use crate::error::HalError;
use crate::model::{BackendKind, DeviceDescriptor, MemoryRange, PageSample, ZeroizeReceipt};
use crate::native::common::NativeBackendCore;

const ROCM_BACKEND_NAME: &str = "rocm-native";
const ROCM_DRIVER_CANDIDATES: &[&str] = &[
    "/opt/rocm/lib/libamdhip64.so",
    "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
    "C:\\Program Files\\AMD\\ROCm\\bin\\amdhip64.dll",
];

#[derive(Debug, Clone)]
pub struct RocmDriverConfig {
    pub driver_library_path: Option<String>,
    pub sync_stream_after_memset: bool,
}

impl Default for RocmDriverConfig {
    fn default() -> Self {
        Self {
            driver_library_path: None,
            sync_stream_after_memset: true,
        }
    }
}

/// Native AMD ROCm HIP (`hipMemsetAsync` / `hipStreamSynchronize` / `hipMemcpyDtoH`)
/// sanitization backend for AMD Instinct MI300X accelerators.
#[derive(Debug)]
pub struct RocmAcceleratorBackend {
    pub config: RocmDriverConfig,
    core: NativeBackendCore,
}

impl RocmAcceleratorBackend {
    pub fn connect(config: RocmDriverConfig) -> Result<Self, HalError> {
        let core = NativeBackendCore::probe_and_init(
            ROCM_BACKEND_NAME,
            BackendKind::RocmNative,
            config.driver_library_path.as_deref(),
            ROCM_DRIVER_CANDIDATES,
        )?;
        Ok(Self { config, core })
    }

    pub fn driver_path(&self) -> &str {
        &self.core.driver_path
    }
}

#[async_trait]
impl AcceleratorDevice for RocmAcceleratorBackend {
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
        Err(self.core.invocation_error("hipMalloc"))
    }

    async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("hipDeviceSynchronize"))
    }

    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("hipMemsetAsync"))
    }

    async fn read_back_pages(
        &self,
        device_uuid: Uuid,
        _page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("hipMemcpyDtoH"))
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
