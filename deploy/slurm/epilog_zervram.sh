#!/usr/bin/env bash
set -euo pipefail

ZERVRAM_BIN="${ZERVRAM_BIN:-/usr/local/bin/zervram-scrub}"
ZERVRAM_DAEMON_URL="${ZERVRAM_DAEMON_URL:-http://127.0.0.1:9740}"
ZERVRAM_TOKEN_FILE="${ZERVRAM_TOKEN_FILE:-/etc/zervram/daemon.token}"
ZERVRAM_KEY_FILE="${ZERVRAM_KEY_FILE:-/etc/zervram/node.key}"
ZERVRAM_CHAIN_FILE="${ZERVRAM_CHAIN_FILE:-/var/lib/zervram/chain.jsonl}"
ZERVRAM_AUDIT_LOG="${ZERVRAM_AUDIT_LOG:-/var/log/zervram/attestations.jsonl}"

RAW_GPUS="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
if [ -z "${RAW_GPUS}" ] || [ "${RAW_GPUS}" = "NoDevFiles" ]; then
  exit 0
fi

mkdir -p "$(dirname "${ZERVRAM_AUDIT_LOG}")"
IFS=',' read -ra GPUS <<< "${RAW_GPUS}"

for gpu in "${GPUS[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  [ -z "${gpu}" ] && continue
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    logger -t zervram-epilog "Skipping non-ordinal GPU identifier: ${gpu}"
    continue
  fi

  set +e
  "${ZERVRAM_BIN}" \
    --device "${gpu}" \
    --daemon-url "${ZERVRAM_DAEMON_URL}" \
    --fallback-direct \
    --token-file "${ZERVRAM_TOKEN_FILE}" \
    --key-file "${ZERVRAM_KEY_FILE}" \
    --chain-file "${ZERVRAM_CHAIN_FILE}" \
    --tenant-id "${SLURM_JOB_ACCOUNT:-unknown}" \
    --job-id "${SLURM_JOB_ID:-unknown}" \
    --nonce "slurm-${SLURM_JOB_ID:-0}-gpu-${gpu}" \
    --json >> "${ZERVRAM_AUDIT_LOG}"
  rc=$?
  set -e

  if [ "${rc}" -eq 42 ] || [ "${rc}" -ne 0 ]; then
    logger -t zervram-epilog "CRITICAL: GPU ${gpu} scrub failed (exit=${rc}) on $(hostname -s); draining node"
    scontrol update NodeName="$(hostname -s)" State=DRAIN Reason="zervram_quarantine_or_scrub_failed" || true
    exit "${rc}"
  fi
done
