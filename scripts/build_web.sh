#!/usr/bin/env bash
# Build the frontend and embed it in the python package so
# `pip install` + `mimesis-earth serve` needs no Node.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d web/node_modules ] || { echo "web/node_modules missing - run: cd web && npm install" >&2; exit 1; }
(cd web && npm run build)
rm -rf python/src/mimesis_earth/webapp
cp -r web/dist python/src/mimesis_earth/webapp
echo "webapp embedded: $(du -sh python/src/mimesis_earth/webapp | cut -f1)"
