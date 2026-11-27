#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use chrono::Utc;
use uuid::Uuid;
use zervram_scrub::{
    AttestationBundle, QuarantineRecord, QuarantineStore, EXIT_CODE_QUARANTINED,
    GENESIS_PREV_CERT_SHA256,
};

struct TempDirGuard(PathBuf);
impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_for_addr_file(
    addr_file: &Path,
    child: &mut Child,
) -> Result<String, Box<dyn std::error::Error>> {
    for _ in 0..400 {
        if let Ok(raw) = std::fs::read_to_string(addr_file) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }
        if let Some(status) = child.try_wait()? {
            return Err(format!("zervramd exited early with status {status}").into());
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    Err("timed out waiting for zervramd --addr-file".into())
}

fn http_get_raw(
    daemon_url: &str,
    path: &str,
    bearer_token: Option<&str>,
) -> Result<(u16, String), Box<dyn std::error::Error>> {
    let authority = daemon_url
        .strip_prefix("http://")
        .ok_or("invalid daemon_url")?;
    let mut stream = TcpStream::connect(authority)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let auth = match bearer_token {
        Some(t) => format!("Authorization: Bearer {t}\r\n"),
        None => String::new(),
    };
    let req = format!("GET {path} HTTP/1.1\r\nHost: {authority}\r\n{auth}Connection: close\r\n\r\n");
    stream.write_all(req.as_bytes())?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp)?;
    let (head, body) = resp
        .split_once("\r\n\r\n")
        .ok_or("malformed HTTP response")?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .ok_or("missing HTTP status")?
        .parse()?;
    Ok((status, body.to_string()))
}

#[test]
fn daemon_warm_two_scrubs_hash_chain_auth_fallback_and_quarantine(
) -> Result<(), Box<dyn std::error::Error>> {
    let daemon_bin = env!("CARGO_BIN_EXE_zervramd");
    let scrub_bin = env!("CARGO_BIN_EXE_zervram-scrub");

    let tmp_dir = std::env::temp_dir().join(format!("zervram_daemon_it_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir)?;
    let _dir_guard = TempDirGuard(tmp_dir.clone());

    let key_file = tmp_dir.join("node.key");
    let state_dir = tmp_dir.join("state");
    let chain_file = state_dir.join("chain.jsonl");
    let quarantine_file = state_dir.join("quarantine.json");
    let addr_file = tmp_dir.join("daemon.addr");
    let token_file = tmp_dir.join("daemon.token");
    let cert1_path = tmp_dir.join("cert1.json");
    let cert2_path = tmp_dir.join("cert2.json");
    let cert3_path = tmp_dir.join("cert3_fallback.json");

    let token = "zervram-test-local-token-040";
    std::fs::write(&token_file, token)?;

    let child = Command::new(daemon_bin)
        .args([
            "--listen",
            "127.0.0.1:0",
            "--allow-mock",
            "--state-dir",
            &state_dir.to_string_lossy(),
            "--key-file",
            &key_file.to_string_lossy(),
            "--token-file",
            &token_file.to_string_lossy(),
            "--addr-file",
            &addr_file.to_string_lossy(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let mut daemon = ChildGuard(child);

    let daemon_url = wait_for_addr_file(&addr_file, &mut daemon.0)?;

    // 1. /v1/health fails closed without token, succeeds with token
    let (unauth_status, _) = http_get_raw(&daemon_url, "/v1/health", None)?;
    assert_eq!(unauth_status, 401);

    let (health_status, health_body) = http_get_raw(&daemon_url, "/v1/health", Some(token))?;
    assert_eq!(health_status, 200);
    let health_json: serde_json::Value = serde_json::from_str(&health_body)?;
    assert_eq!(health_json.get("version"), Some(&serde_json::json!("0.4.0")));
    assert_eq!(
        health_json.get("backend_kind"),
        Some(&serde_json::json!("mock_cpu"))
    );
    assert_eq!(
        health_json.get("head_sha256"),
        Some(&serde_json::json!(GENESIS_PREV_CERT_SHA256))
    );

    // 2. First warm scrub via zervram-scrub --daemon-url
    let out1 = Command::new(scrub_bin)
        .args([
            "--device",
            "0",
            "--allow-mock",
            "--daemon-url",
            &daemon_url,
            "--token-file",
            &token_file.to_string_lossy(),
            "--nonce",
            "warm-nonce-1",
            "--out-cert",
            &cert1_path.to_string_lossy(),
            "--json",
        ])
        .output()?;
    assert!(
        out1.status.success(),
        "daemon scrub 1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    // 3. Second consecutive warm scrub via zervram-scrub --daemon-url
    let out2 = Command::new(scrub_bin)
        .args([
            "--device",
            "0",
            "--allow-mock",
            "--daemon-url",
            &daemon_url,
            "--token-file",
            &token_file.to_string_lossy(),
            "--nonce",
            "warm-nonce-2",
            "--out-cert",
            &cert2_path.to_string_lossy(),
            "--json",
        ])
        .output()?;
    assert!(
        out2.status.success(),
        "daemon scrub 2 failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );

    let bundle1: AttestationBundle = serde_json::from_str(&std::fs::read_to_string(&cert1_path)?)?;
    let bundle2: AttestationBundle = serde_json::from_str(&std::fs::read_to_string(&cert2_path)?)?;
    assert_eq!(bundle1.certificate.prev_cert_sha256, GENESIS_PREV_CERT_SHA256);
    assert_eq!(
        bundle2.certificate.prev_cert_sha256,
        bundle1.certificate_sha256
    );
    assert_eq!(
        bundle2.certificate.challenge_nonce.as_deref(),
        Some("warm-nonce-2")
    );

    // 4. Verify both Ed25519 signatures and full chain against pinned node.key
    for (path, nonce) in [(&cert1_path, "warm-nonce-1"), (&cert2_path, "warm-nonce-2")] {
        let v = Command::new(scrub_bin)
            .args([
                "verify",
                &path.to_string_lossy(),
                "--key-file",
                &key_file.to_string_lossy(),
                "--expected-nonce",
                nonce,
            ])
            .output()?;
        assert!(
            v.status.success(),
            "verify failed: {}",
            String::from_utf8_lossy(&v.stderr)
        );
    }

    let vc = Command::new(scrub_bin)
        .args([
            "verify-chain",
            "--chain-file",
            &chain_file.to_string_lossy(),
            "--key-file",
            &key_file.to_string_lossy(),
        ])
        .output()?;
    assert!(
        vc.status.success(),
        "verify-chain failed: {}",
        String::from_utf8_lossy(&vc.stderr)
    );

    // 5. Fail closed when client omits --allow-mock against a mock daemon
    let mock_denied = Command::new(scrub_bin)
        .args([
            "--device",
            "0",
            "--daemon-url",
            &daemon_url,
            "--token-file",
            &token_file.to_string_lossy(),
        ])
        .output()?;
    assert!(!mock_denied.status.success());

    // 6. --fallback-direct succeeds when daemon URL is unreachable
    let closed_port = {
        let tmp_listener = TcpListener::bind("127.0.0.1:0")?;
        tmp_listener.local_addr()?.port()
    };
    let dead_url = format!("http://127.0.0.1:{closed_port}");
    let fb = Command::new(scrub_bin)
        .args([
            "--device",
            "0",
            "--allow-mock",
            "--daemon-url",
            &dead_url,
            "--fallback-direct",
            "--key-file",
            &key_file.to_string_lossy(),
            "--chain-file",
            &chain_file.to_string_lossy(),
            "--out-cert",
            &cert3_path.to_string_lossy(),
            "--json",
        ])
        .output()?;
    assert!(
        fb.status.success(),
        "fallback-direct failed: {}",
        String::from_utf8_lossy(&fb.stderr)
    );
    let bundle3: AttestationBundle = serde_json::from_str(&std::fs::read_to_string(&cert3_path)?)?;
    assert_eq!(
        bundle3.certificate.prev_cert_sha256,
        bundle2.certificate_sha256
    );

    // 7. Quarantined device returns 423 Locked -> exit code 42 even with --fallback-direct
    QuarantineStore::open(&quarantine_file).record_quarantine(QuarantineRecord {
        device_uuid: bundle1.certificate.device_uuid,
        pci_bus_id: bundle1.certificate.pci_bus_id.clone(),
        vendor_model: bundle1.certificate.vendor_model.clone(),
        reason: "simulated ECC fault".to_string(),
        quarantined_at: Utc::now(),
        failed_cert_id: None,
    })?;
    let q_out = Command::new(scrub_bin)
        .args([
            "--device",
            "0",
            "--allow-mock",
            "--daemon-url",
            &daemon_url,
            "--fallback-direct",
            "--token-file",
            &token_file.to_string_lossy(),
        ])
        .output()?;
    assert_eq!(q_out.status.code(), Some(i32::from(EXIT_CODE_QUARANTINED)));

    Ok(())
}