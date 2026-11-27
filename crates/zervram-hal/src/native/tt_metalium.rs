use async_trait::async_trait;
use uuid::Uuid;

use crate::backend::AcceleratorDevice;
use crate::error::HalError;
use crate::model::{BackendKind, DeviceDescriptor, MemoryRange, PageSample, ZeroizeReceipt};
use crate::native::common::NativeBackendCore;

const TT_BACKEND_NAME: &str = "tt-metalium-native (experimental-stub)";
const TT_DRIVER_CANDIDATES: &[&str] = &[
    "/usr/local/lib/libtt_metal.so",
    "/opt/tenstorrent/lib/libtt_metal.so",
];

#[derive(Debug, Clone)]
pub struct TtMetaliumConfig {
    pub runtime_library_path: Option<String>,
    pub scrub_l1_sram_banks: bool,
}

impl Default for TtMetaliumConfig {
    fn default() -> Self {
        Self {
            runtime_library_path: None,
            scrub_l1_sram_banks: true,
        }
    }
}

/// EXPERIMENTAL STUB: Tenstorrent TT-Metalium native FFI bindings are not yet implemented.
/// All memory operations fail closed until vendor SDK FFI integration is linked.
#[derive(Debug)]
pub struct TtMetaliumBackend {
    pub config: TtMetaliumConfig,
    core: NativeBackendCore,
}

impl TtMetaliumBackend {
    pub fn connect(config: TtMetaliumConfig) -> Result<Self, HalError> {
        let core = NativeBackendCore::probe_and_init(
            TT_BACKEND_NAME,
            BackendKind::TtMetaliumNative,
            config.runtime_library_path.as_deref(),
            TT_DRIVER_CANDIDATES,
        )?;
        Ok(Self { config, core })
    }

    pub fn driver_path(&self) -> &str {
        &self.core.driver_path
    }
}

#[async_trait]
impl AcceleratorDevice for TtMetaliumBackend {
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
        Err(self.core.invocation_error("tt_metal_create_buffer"))
    }

    async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("tt_metal_finish_command_queue"))
    }

    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("tt_metal_enqueue_zero_buffer"))
    }

    async fn read_back_pages(
        &self,
        device_uuid: Uuid,
        _page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError> {
        let _ = self.core.get(device_uuid).await?;
        Err(self.core.invocation_error("tt_metal_enqueue_read_buffer"))
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