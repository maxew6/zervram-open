#![forbid(unsafe_code)]

pub mod certificate;
pub mod engine;
pub mod error;
pub mod signer;
pub mod store;
pub mod verifier;

pub use certificate::{
    hash_tenant_id, parse_verifying_key, verify_bundle_json, AttestationBundle,
    CertificateVerificationResult, ChainAuditSummary, VerifiedBundleSummary, VerifyPolicy,
    ZeroizationCertificate, ZeroizationCertificateV2, CERTIFICATE_SCHEMA_V2,
    GENESIS_PREV_CERT_SHA256,
};
pub use engine::{ScrubConfig, ScrubRequest, VramScrubber, VramZeroEngine};
pub use error::ScrubError;
pub use signer::{load_verifying_key_from_file, AttestationSigner, Ed25519Signer};
pub use store::{
    default_chain_file_path, default_quarantine_file_path, default_state_dir,
    AppendOnlyChainStore, CertificateStore, InMemoryCertificateStore, QuarantineRecord,
    QuarantineStateFile, QuarantineStore, EXIT_CODE_QUARANTINED,
};
pub use verifier::{verify_zeroized_pages, PageVerificationReport, SamplingStrategy};