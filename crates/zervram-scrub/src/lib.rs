#![forbid(unsafe_code)]

pub mod certificate;
pub mod engine;
pub mod error;
pub mod signer;
pub mod store;
pub mod verifier;

pub use certificate::{
    hash_tenant_id, CertificateVerificationResult, ChainAuditSummary, ZeroizationCertificate,
    GENESIS_PREV_CERT_SHA256,
};
pub use engine::{ScrubConfig, ScrubRequest, VramScrubber, VramZeroEngine};
pub use error::ScrubError;
pub use signer::{AttestationSigner, Ed25519Signer};
pub use store::{CertificateStore, InMemoryCertificateStore};
pub use verifier::{PageVerificationReport, SamplingStrategy};
