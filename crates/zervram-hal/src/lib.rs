#![deny(unsafe_code)]

pub mod backend;
pub mod catalog;
pub mod error;
pub mod model;

#[cfg(feature = "mock-gpu")]
pub mod mock;

pub mod native;

pub use backend::{AcceleratorBackend, AcceleratorDevice};
pub use catalog::{
    all_hardware_specs, calculate_tpp, hardware_spec, resolve_model_from_k8s_resource,
    total_tpp_for_units, AGGREGATE_TPP_STATUTORY_CEILING, SPEC_A100, SPEC_B200_BLACKWELL,
    SPEC_BLACKHOLE_P150, SPEC_FURIOSA_RNGD, SPEC_FURIOSA_RNGD_SLICE, SPEC_H100_SXM, SPEC_H200_SXM,
    SPEC_L4, SPEC_L40S, SPEC_MI300X, SPEC_MI325X, SPEC_MI350X, SPEC_T4, SPEC_UNKNOWN,
};
pub use error::HalError;
pub use model::{
    compute_chunk_table_sha256, AcceleratorModel, AcceleratorVendor, BackendKind,
    DeviceDescriptor, DeviceState, GpuReductionVerification, HardwareSpec, MemoryRange, PageSample,
    VerificationMode, ZeroizeReceipt, READBACK_POISON_BYTE, VRAM_PAGE_SIZE_BYTES,
};
pub use native::{
    connect_physical_backend, connect_physical_backend_with_options, PhysicalBackendOptions,
};

#[cfg(feature = "mock-gpu")]
pub use mock::{MockAcceleratorBackend, MockClusterConfig, MockDeviceRecord, SimulatedVramPool};

/// Creates the default compiled-in [`AcceleratorDevice`] implementation.
///
/// Prefers `MockAcceleratorBackend` when `mock-gpu` is enabled for unit tests and local
/// development; production CLI callers should invoke [`connect_physical_backend_with_options`]
/// and only fall back to [`MockAcceleratorBackend`] when explicitly authorized via `--allow-mock`.
#[cfg(feature = "mock-gpu")]
pub fn create_default_backend() -> Result<std::sync::Arc<dyn AcceleratorDevice>, HalError> {
    Ok(std::sync::Arc::new(MockAcceleratorBackend::new(
        MockClusterConfig::default(),
    )))
}

#[cfg(not(feature = "mock-gpu"))]
pub fn create_default_backend() -> Result<std::sync::Arc<dyn AcceleratorDevice>, HalError> {
    native::connect_physical_backend()
}