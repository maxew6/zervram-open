use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum HalError {
    #[error("accelerator device {device_uuid} not found")]
    DeviceNotFound { device_uuid: Uuid },

    #[error("invalid state transition on device {device_uuid}: expected {expected}, found {actual}")]
    InvalidDeviceState {
        device_uuid: Uuid,
        expected: &'static str,
        actual: String,
    },

    #[error("memory range [{start_addr:#x}, len={byte_len}) is invalid or unaligned to {page_size}B page boundary")]
    InvalidMemoryRange {
        start_addr: u64,
        byte_len: u64,
        page_size: u32,
    },

    #[error("page address {page_addr:#x} out of bounds for device {device_uuid}")]
    PageAddressOutOfBounds { device_uuid: Uuid, page_addr: u64 },

    #[error("invalid TPP formula parameters: mac_tops={mac_tops}, bit_length={bit_length}")]
    InvalidTppParameters { mac_tops: f64, bit_length: u16 },

    #[error("arithmetic overflow computing {context}")]
    ArithmeticOverflow { context: &'static str },

    #[error("unknown hardware model identifier: {raw}")]
    UnknownHardwareModel { raw: String },

    #[error("native backend driver failure ({backend}): {message}")]
    NativeDriverFailure { backend: &'static str, message: String },
}
