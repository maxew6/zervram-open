use thiserror::Error;
use uuid::Uuid;
use zervram_hal::HalError;

#[derive(Debug, Error)]
pub enum ScrubError {
    #[error("HAL operation failed during VRAM sanitization: {0}")]
    Hal(#[from] HalError),

    #[error("residual non-zero VRAM data detected on device {device_uuid} at page {page_addr:#x} ({non_zero_bytes} non-zero bytes); device quarantined")]
    ResidualDataDetected {
        device_uuid: Uuid,
        page_addr: u64,
        non_zero_bytes: usize,
    },

    #[error("device {device_uuid} is quarantined and blocked from scheduling: {reason}")]
    DeviceQuarantined { device_uuid: Uuid, reason: String },

    #[error("zeroization verification sampled 0 pages on device {device_uuid}; refusing vacuous verified_zero")]
    EmptyVerificationSample { device_uuid: Uuid },

    #[error("zeroization SLA exceeded on device {device_uuid}: {duration_us}us > {sla_max_us}us")]
    SlaBudgetExceeded {
        device_uuid: Uuid,
        duration_us: u64,
        sla_max_us: u64,
    },

    #[error("cryptographic signing failure: {0}")]
    SigningFailure(String),

    #[error("persistent storage failure: {0}")]
    StorageFailure(String),

    #[error("certificate verification failed: {0}")]
    VerificationFailure(String),

    #[error("invalid certificate field `{field}`: {details}")]
    InvalidCertificateField {
        field: &'static str,
        details: String,
    },

    #[error("certificate {cert_id} not found")]
    CertificateNotFound { cert_id: Uuid },

    #[error("certificate hash-chain integrity failure at cert {cert_id}: expected prev={expected}, found={actual}")]
    ChainIntegrityViolation {
        cert_id: Uuid,
        expected: String,
        actual: String,
    },

    #[error("Ed25519 signature verification failed for certificate {cert_id}")]
    InvalidSignature { cert_id: Uuid },

    #[error("invalid hex encoding in field {field}: {details}")]
    InvalidHex {
        field: &'static str,
        details: String,
    },
}