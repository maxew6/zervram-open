#![cfg_attr(
    not(any(
        feature = "cuda-native",
        feature = "rocm-native",
        feature = "tt-metalium-native",
        feature = "furiosa-native"
    )),
    forbid(unsafe_code)
)]

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
    total_tpp_for_units, AGGREGATE_TPP_STATUTORY_CEILING, SPEC_B200_BLACKWELL, SPEC_BLACKHOLE_P150,
    SPEC_FURIOSA_RNGD, SPEC_FURIOSA_RNGD_SLICE, SPEC_H100_SXM, SPEC_MI300X,
};
pub use error::HalError;
pub use model::{
    AcceleratorModel, AcceleratorVendor, BackendKind, DeviceDescriptor, DeviceState, HardwareSpec,
    MemoryRange, PageSample, ZeroizeReceipt, VRAM_PAGE_SIZE_BYTES,
};

#[cfg(feature = "mock-gpu")]
pub use mock::{MockAcceleratorBackend, MockClusterConfig, MockDeviceRecord, SimulatedVramPool};

/// Creates the default compiled-in [`AcceleratorDevice`] implementation.
///
/// With the default `mock-gpu` feature enabled, initializes a deterministic heterogeneous
/// [`MockAcceleratorBackend`] simulating NVIDIA H100 SXM, NVIDIA B200 Blackwell, AMD MI300X,
/// Tenstorrent Blackhole p150, and FuriosaAI RNGD (8x SR-IOV slices) without any C/CUDA SDK.
#[cfg(feature = "mock-gpu")]
pub fn create_default_backend() -> Result<std::sync::Arc<dyn AcceleratorDevice>, HalError> {
    Ok(std::sync::Arc::new(MockAcceleratorBackend::new(
        MockClusterConfig::default(),
    )))
}

#[cfg(not(feature = "mock-gpu"))]
pub fn create_default_backend() -> Result<std::sync::Arc<dyn AcceleratorDevice>, HalError> {
    native::connect_default_native_backend()
}
