mkdir -p audit_env_paths

SEARCH_DIRS=()
for d in src scripts jobs; do
  [ -d "$d" ] && SEARCH_DIRS+=("$d")
done

{
  echo "===== alpha-edge-algo ====="
  grep -R --exclude="*.pyc" --exclude-dir="__pycache__" --exclude-dir=".venv" \
    "alpha-edge-algo" -n "${SEARCH_DIRS[@]}" 2>/dev/null || true

  echo ""
  echo "===== engine/v1 ====="
  grep -R --exclude="*.pyc" --exclude-dir="__pycache__" --exclude-dir=".venv" \
    "engine/v1" -n "${SEARCH_DIRS[@]}" 2>/dev/null || true

  echo ""
  echo "===== market/ ====="
  grep -R --exclude="*.pyc" --exclude-dir="__pycache__" --exclude-dir=".venv" \
    "market/" -n "${SEARCH_DIRS[@]}" 2>/dev/null || true

  echo ""
  echo "===== runtime constants ====="
  grep -R --exclude="*.pyc" --exclude-dir="__pycache__" --exclude-dir=".venv" \
    "ENGINE_BUCKET\|ENGINE_REGION\|ENGINE_ROOT\|ENGINE_ROOT_PREFIX\|OHLCV_USD_ROOT\|RETURNS_WIDE_CACHE_PATH" \
    -n "${SEARCH_DIRS[@]}" 2>/dev/null || true
} > audit_env_paths/all_hits_after_dev_smoke_clean.txt