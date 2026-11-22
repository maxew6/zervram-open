use std::fmt;

use ed25519_dalek::pkcs8::DecodePrivateKey;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

use crate::error::ScrubError;

pub trait AttestationSigner: Send + Sync + 'static {
    fn sign_digest(&self, digest: &[u8; 32]) -> Result<[u8; 64], ScrubError>;
    fn verify_digest(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), ScrubError>;
    fn verifying_key_bytes(&self) -> [u8; 32];
    fn verifying_key_hex(&self) -> String;
}

/// Ed25519 signer backed by `ed25519-dalek` (`SigningKey` is zeroized on drop).
pub struct Ed25519Signer {
    signing_key: SigningKey,
}

impl fmt::Debug for Ed25519Signer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ed25519Signer")
            .field("verifying_key_hex", &self.verifying_key_hex())
            .finish()
    }
}

impl Ed25519Signer {
    pub fn from_bytes(secret_key: [u8; 32]) -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&secret_key),
        }
    }

    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        Self { signing_key }
    }

    pub fn generate_ephemeral() -> Self {
        let mut csprng = rand::rngs::OsRng;
        Self {
            signing_key: SigningKey::generate(&mut csprng),
        }
    }

    pub fn from_pkcs8_pem(pem: &str) -> Result<Self, ScrubError> {
        let signing_key = SigningKey::from_pkcs8_pem(pem)
            .map_err(|err| ScrubError::SigningFailure(format!("invalid PKCS#8 PEM key: {err}")))?;
        Ok(Self { signing_key })
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    pub fn clone_signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

impl AttestationSigner for Ed25519Signer {
    fn sign_digest(&self, digest: &[u8; 32]) -> Result<[u8; 64], ScrubError> {
        Ok(self.signing_key.sign(digest).to_bytes())
    }

    fn verify_digest(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), ScrubError> {
        let sig = Signature::from_bytes(signature);
        self.signing_key
            .verifying_key()
            .verify_strict(digest, &sig)
            .map_err(|err| ScrubError::SigningFailure(err.to_string()))
    }

    fn verifying_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    fn verifying_key_hex(&self) -> String {
        hex::encode(self.verifying_key_bytes())
    }
}
