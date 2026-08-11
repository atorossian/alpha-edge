#!/usr/bin/env bash
set -euo pipefail

poetry run alphaedge-morning \
  --env prod \
  --run-transition \
  --transition-equity0 32388.35 \
  --confirm-prod-write