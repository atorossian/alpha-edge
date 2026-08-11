#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Alpha Edge DEV Smoke Pipeline
#
# Purpose:
#   Validate the full dev environment pipeline without touching prod:
#     1) Market ingest for small sample universe
#     2) Ledger rebuild
#     3) Warehouse build from ledger
#     4) Daily report
#     5) Portfolio search
#     6) Warehouse build again to include daily report stats
#
# Usage:
#   bash scripts/dev_smoke_pipeline.sh
#
# Optional overrides:
#   SMOKE_DT=2024-01-31 bash scripts/dev_smoke_pipeline.sh
#   UNIVERSE_PATH=data/outputs/dev/dev_universe_sample.csv bash scripts/dev_smoke_pipeline.sh
# ============================================================

ENV_NAME="${ENV_NAME:-dev}"
BUCKET="${BUCKET:-alpha-edge-algo}"
REGION="${REGION:-eu-west-1}"

SMOKE_DT="${SMOKE_DT:-2024-01-31}"
INGEST_START="${INGEST_START:-2015-01-01}"
INGEST_END="${INGEST_END:-2024-02-05}"

ACCOUNT_ID="${ACCOUNT_ID:-main}"
UNIVERSE_PATH="${UNIVERSE_PATH:-data/outputs/dev/dev_universe_sample.csv}"

EQUITY0="${EQUITY0:-1000}"
GOALS="${GOALS:-1200,1500,2000}"
MAIN_GOAL="${MAIN_GOAL:-1500}"
TARGET_LEVERAGE="${TARGET_LEVERAGE:-1.0}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="data/outputs/dev/smoke_runs/${RUN_TS}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Alpha Edge DEV Smoke Pipeline"
echo "============================================================"
echo "env:           ${ENV_NAME}"
echo "bucket:        ${BUCKET}"
echo "region:        ${REGION}"
echo "smoke_dt:      ${SMOKE_DT}"
echo "ingest_start:  ${INGEST_START}"
echo "ingest_end:    ${INGEST_END}"
echo "account_id:    ${ACCOUNT_ID}"
echo "universe_path: ${UNIVERSE_PATH}"
echo "log_dir:       ${LOG_DIR}"
echo "============================================================"
echo ""

run_step() {
  local step_name="$1"
  shift

  local log_file="${LOG_DIR}/${step_name}.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "[START] ${step_name}"
  echo "------------------------------------------------------------"
  echo "log: ${log_file}"
  echo ""

  {
    echo "===== ${step_name} ====="
    echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cmd: $*"
    echo ""
    "$@"
    echo ""
    echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "${log_file}"

  echo ""
  echo "[OK] ${step_name}"
}

# ------------------------------------------------------------
# Basic preflight
# ------------------------------------------------------------
if [ ! -f "${UNIVERSE_PATH}" ]; then
  echo "[ERROR] Universe sample not found: ${UNIVERSE_PATH}"
  exit 1
fi

if ! command -v poetry >/dev/null 2>&1; then
  echo "[ERROR] poetry command not found."
  exit 1
fi

# ------------------------------------------------------------
# 1) Market ingest: small dev universe only
# ------------------------------------------------------------
run_step "01_ingest_market_data" \
  poetry run python -m alpha_edge.market.ingest_market_data \
    --env "${ENV_NAME}" \
    --universe-path "${UNIVERSE_PATH}" \
    --start "${INGEST_START}" \
    --end "${INGEST_END}" \
    --ignore-existing-state \
    --no-triage \
    --max-workers 2 \
    --yahoo-max-concurrency 1 \
    --yahoo-rate-per-sec 0.8

# ------------------------------------------------------------
# 2) Ledger rebuild
#
# Assumes dev trades/cashflows/dividends already exist under:
#   s3://alpha-edge-algo/dev/engine/v1/trades/
#   s3://alpha-edge-algo/dev/engine/v1/cashflows/
#   s3://alpha-edge-algo/dev/engine/v1/dividends/
# ------------------------------------------------------------
run_step "02_rebuild_ledger" \
  poetry run python -m alpha_edge.operations.rebuild_ledger \
    --env "${ENV_NAME}" \
    --start "${SMOKE_DT}" \
    --end "${SMOKE_DT}" \
    --as-of "${SMOKE_DT}" \
    --account-id "${ACCOUNT_ID}" \
    --prices-mode asof

# ------------------------------------------------------------
# 3) Warehouse build from ledger before report
# ------------------------------------------------------------
run_step "03_build_warehouse_pre_report" \
  poetry run python -m alpha_edge.warehouse.build_warehouse \
    --env "${ENV_NAME}" \
    --dt "${SMOKE_DT}" \
    --account-id "${ACCOUNT_ID}"

# ------------------------------------------------------------
# 4) Daily report
# ------------------------------------------------------------
run_step "04_daily_report" \
  poetry run python -m alpha_edge.jobs.run_daily_report \
    --env "${ENV_NAME}" \
    --as-of "${SMOKE_DT}"

# ------------------------------------------------------------
# 5) Portfolio search, reduced workload for smoke test
# ------------------------------------------------------------
run_step "05_portfolio_search" \
  poetry run python -m alpha_edge.jobs.run_portfolio_search \
    --env "${ENV_NAME}" \
    --as-of "${SMOKE_DT}" \
    --run-dt "${SMOKE_DT}" \
    --equity0 "${EQUITY0}" \
    --goals "${GOALS}" \
    --main-goal "${MAIN_GOAL}" \
    --override-target-leverage "${TARGET_LEVERAGE}" \
    --universe-csv "${UNIVERSE_PATH}" \
    --pop-size 20 \
    --generations 5 \
    --n-paths-init 500 \
    --n-paths-final 1000 \
    --skip-stability-rerank \
    --anneal-steps 20 \
    --anneal-n-paths-init 500 \
    --anneal-n-paths-final 1000 \
    --min-universe-size 5

# ------------------------------------------------------------
# 6) Warehouse build again after daily report exists
# ------------------------------------------------------------
run_step "06_build_warehouse_post_report" \
  poetry run python -m alpha_edge.warehouse.build_warehouse \
    --env "${ENV_NAME}" \
    --dt "${SMOKE_DT}" \
    --account-id "${ACCOUNT_ID}"

echo ""
echo "============================================================"
echo "[OK] DEV smoke pipeline completed"
echo "logs: ${LOG_DIR}"
echo "============================================================"