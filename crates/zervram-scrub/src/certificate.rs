use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::ScrubError;

pub const GENESIS_PREV_CERT_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Computes a privacy-preserving SHA-256 hex digest of the departing tenant identifier.
pub fn hash_tenant_id(tenant_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"zervram:tenant:v1|");
    hasher.update(tenant_id.as_bytes());
    hex::encode(hasher.finalize())
}

/// Exact 11-field Ed25519 + SHA-256 attestation minted after every VRAM zeroization cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZeroizationCertificate {
    pub cert_id: Uuid,
    pub device_uuid: Uuid,
    pub vendor_model: String,
    pub tenant_prev_hash: String,
    pub start_addr: u64,
    pub byte_len: u64,
    pub duration_us: u64,
    pub verified_zero: bool,
    pub timestamp_utc: DateTime<Utc>,
    pub prev_cert_sha256: String,
    pub ed25519_signature_hex: String,
}

impl ZeroizationCertificate {
    /// Deterministic canonical SHA-256 digest of unsigned certificate fields (domain-separated).
    /// This 32-byte digest is the exact message signed by Ed25519.
    pub fn canonical_signing_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"zervram:vram-zero:cert:v1|");
        hasher.update(self.cert_id.as_bytes());
        hasher.update(self.device_uuid.as_bytes());
        hasher.update(self.vendor_model.as_bytes());
        hasher.update(b"|");
        hasher.update(self.tenant_prev_hash.as_bytes());
        hasher.update(self.start_addr.to_be_bytes());
        hasher.update(self.byte_len.to_be_bytes());
        hasher.update(self.duration_us.to_be_bytes());
        hasher.update([u8::from(self.verified_zero)]);
        hasher.update(
            self.timestamp_utc
                .timestamp_nanos_opt()
                .unwrap_or(0)
                .to_be_bytes(),
        );
        hasher.update(self.prev_cert_sha256.as_bytes());
        hasher.finalize().into()
    }

    /// SHA-256 hash of the complete signed certificate, chained into the next certificate's
    /// `prev_cert_sha256` field.
    pub fn certificate_sha256_hex(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_signing_digest());
        hasher.update(self.ed25519_signature_hex.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Verifies `ed25519_signature_hex` strictly against `verifying_key`.
    pub fn verify_signature(&self, verifying_key: &VerifyingKey) -> Result<(), ScrubError> {
        let sig_bytes =
            hex::decode(&self.ed25519_signature_hex).map_err(|err| ScrubError::InvalidHex {
                field: "ed25519_signature_hex",
                details: err.to_string(),
            })?;
        let signature = Signature::from_slice(&sig_bytes).map_err(|_| ScrubError::InvalidSignature {
            cert_id: self.cert_id,
        })?;
        let digest = self.canonical_signing_digest();
        verifying_key
            .verify_strict(&digest, &signature)
            .map_err(|_| ScrubError::InvalidSignature {
                cert_id: self.cert_id,
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateVerificationResult {
    pub cert_id: Uuid,
    pub signature_valid: bool,
    pub chain_link_valid: bool,
    pub verified_zero_flag: bool,
    pub computed_cert_sha256: String,
    pub ed25519_public_key_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainAuditSummary {
    pub total_certificates: usize,
    pub head_sha256: String,
    pub all_signatures_valid: bool,
    pub chain_intact: bool,
}
