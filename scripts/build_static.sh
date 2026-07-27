#!/usr/bin/env bash
# Build the static GitHub Pages bundle: wheel + pre-baked worlds + static frontend.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) build our pure-python wheel and stage it for micropip
rm -rf web/public/wheels && mkdir -p web/public/wheels
.venv/bin/pip wheel ./python --no-deps -w web/public/wheels >/dev/null
WHEEL=$(cd web/public/wheels && ls mimesis_earth-*.whl)
echo "{\"wheel\": \"${WHEEL}\"}" > web/public/wheels/manifest.json
echo "staged wheel: ${WHEEL}"

# 2) pre-bake the gallery worlds
.venv/bin/python python/scripts/prebake.py web/public/worlds

# 3) build the static frontend (gallery + pyodide worker)
( cd web && VITE_DEPLOY_TARGET=static npm run build )

echo "static bundle ready at web/dist ($(du -sh web/dist | cut -f1))"
