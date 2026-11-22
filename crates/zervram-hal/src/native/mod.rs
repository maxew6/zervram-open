#[cfg(feature = "cuda-native")]
pub mod cuda;
#[cfg(feature = "furiosa-native")]
pub mod furiosa;
#[cfg(feature = "rocm-native")]
pub mod rocm;
#[cfg(feature = "tt-metalium-native")]
pub mod tt_metalium;

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

#[cfg(any(
    feature = "cuda-native",
    feature = "rocm-native",
    feature = "tt-metalium-native",
    feature = "furiosa-native"
))]
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
                "vendor runtime library not found on host (probed: {}); build with `--features mock-gpu` on systems without datacenter accelerators",
                default_candidates.join(", ")
            ),
        })
    }
}

#[cfg(not(feature = "mock-gpu"))]
pub fn connect_default_native_backend(
) -> Result<std::sync::Arc<dyn crate::backend::AcceleratorDevice>, crate::error::HalError> {
    #[cfg(feature = "cuda-native")]
    {
        return Ok(std::sync::Arc::new(cuda::CudaAcceleratorBackend::connect(
            cuda::CudaDriverConfig::default(),
        )?));
    }
    #[cfg(all(not(feature = "cuda-native"), feature = "rocm-native"))]
    {
        return Ok(std::sync::Arc::new(rocm::RocmAcceleratorBackend::connect(
            rocm::RocmDriverConfig::default(),
        )?));
    }
    #[cfg(all(
        not(feature = "cuda-native"),
        not(feature = "rocm-native"),
        feature = "tt-metalium-native"
    ))]
    {
        return Ok(std::sync::Arc::new(
            tt_metalium::TtMetaliumBackend::connect(tt_metalium::TtMetaliumConfig::default())?,
        ));
    }
    #[cfg(all(
        not(feature = "cuda-native"),
        not(feature = "rocm-native"),
        not(feature = "tt-metalium-native"),
        feature = "furiosa-native"
    ))]
    {
        return Ok(std::sync::Arc::new(
            furiosa::FuriosaAcceleratorBackend::connect(furiosa::FuriosaDriverConfig::default())?,
        ));
    }
    #[cfg(not(any(
        feature = "cuda-native",
        feature = "rocm-native",
        feature = "tt-metalium-native",
        feature = "furiosa-native"
    )))]
    {
        Err(crate::error::HalError::NativeDriverFailure {
            backend: "none",
            message: "no accelerator backend feature enabled; enable `mock-gpu` or a native vendor feature".to_string(),
        })
    }
}
