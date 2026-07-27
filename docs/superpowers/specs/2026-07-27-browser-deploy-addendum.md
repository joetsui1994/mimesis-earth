# In-Browser Generation & GitHub Pages Deploy Addendum

**Date:** 2026-07-27
**Status:** Approved design (session discussion + viability spike)
**Generator version:** the snap-before-union prerequisite changes output → bump
`GENERATOR_VERSION` to `0.5.0`.

## Goal

Deploy mimesis-earth as a **static GitHub Pages showcase** where a stranger
who lands on the URL sees a beautiful draggable globe within ~1 second, and —
after a background load they are *told about* — can generate unlimited live
worlds from their own parameters, all client-side. No backend on Pages.

## Approach: Pyodide, not a TypeScript port

The identical Python core runs in-browser via Pyodide (CPython + our
numpy/scipy/shapely/pydantic stack compiled to WASM). This was the
designed-for path — the dependency discipline was maintained from day one
specifically to keep it open. A TS reimplementation is explicitly rejected:
it would be a second codebase drifting from the Python and would break the
reproducibility guarantee.

## Spike results (evidence, run 2026-07-27, Pyodide 0.28 / Node distribution)

| Question | Result |
|---|---|
| Stack loads? | ✅ numpy 2.2.5, scipy 1.14.1, shapely 2.0.7 (GEOS 3.12.1), pydantic 2.10.6 all import & run |
| Load time | ~7.5s cold (download+init), ~1.4s warm |
| generate() runs? | ❌→✅ Fatal GEOS `TopologyException` on near-pole unions; **fixed** by snap-before-union |
| Per-world time (WASM) | ~1.7s default (20k atoms), ~0.4s small, ~1.35s rugged — barely above native's ~1s |
| Determinism | ✅ **byte-identical** native (GEOS 3.13.1) vs WASM (GEOS 3.12.1) with the fix |
| Download footprint | ~31MB uncompressed (~12–18MB gzipped); **scipy is 12.6MB** of it |

## Prerequisite (shared-code change): snap-before-union

`atoms_polygon` (geometry.py) and the parent-union in `generate.py` currently
`unary_union(geoms)` then `set_precision(result, 1e-9)`. Pyodide's older GEOS
fatally rejects degenerate near-pole rings during the union. Fix: snap each
polygon to the grid **before** unioning:
`unary_union([set_precision(g, 1e-9) for g in geoms])`. This makes overlay
robust across GEOS versions and output GEOS-independent. It slightly changes
union output vs current main, so: version bump to 0.5.0, full test-suite
validation, and re-confirm the existing export-validity/determinism invariants.

## Frontend architecture

### World source abstraction

Introduce a `WorldSource` seam so "where a world comes from" is pluggable:
- `ApiSource` — POSTs to `/api/generate` (today's behavior; used by the local
  `mimesis-earth serve` build).
- `GallerySource` — serves pre-baked worlds in sequence (instant, no compute).
- `PyodideSource` — sends a spec to the Pyodide Web Worker, awaits the world.

Build-time flag `VITE_DEPLOY_TARGET` = `server` (default) | `static`:
- **server** build (embedded in the pip package): `ApiSource` — unchanged
  local experience, no Pyodide, instant.
- **static** build (GitHub Pages): `GallerySource` at startup, upgrading to
  `PyodideSource` once the worker is ready.

### Pyodide Web Worker

Pyodide runs in a **Web Worker**, never the main thread (its ~15s boot and
per-world compute must not freeze the draggable globe). Protocol:
- Worker on init: load Pyodide core, `loadPackage([numpy,scipy,shapely,
  pydantic,micropip])`, `micropip.install` our wheel (served as a static
  asset from the same origin), post `{type:'progress', phase, pct}` through
  the phases, then `{type:'ready'}`.
- Main → worker: `{type:'generate', spec}`; worker → main:
  `{type:'world', levels}` or `{type:'error', message}`.
- Pyodide core + scientific wheels load from the official jsdelivr CDN
  (battle-tested, globally cached, keeps our deploy small); only our small
  wheel and the pre-baked worlds are self-hosted.

### Loading communication (required UX)

The visitor must always know what state they're in:
- **On load:** a pre-baked world renders immediately. A small, unobtrusive
  status near the panel reads **"browsing sample worlds · live generator
  loading…"** with a subtle activity indicator. The parameter sliders are
  visually **dimmed/disabled** to signal they aren't live yet; the hint line
  reads "space — next sample world".
- **During load:** the status reflects worker `progress` phases (e.g.
  "loading engine…", "loading scientific libraries…", "starting up…").
- **On ready:** a gentle, dismissable notification —
  **"✓ live generation ready — press space to build a world from your
  settings"** — the sliders un-dim and become active, the hint line switches
  to "space — new world", and the source swaps to `PyodideSource`. Spacebar
  now generates from the live parameters.
- **Graceful degradation:** if the worker errors or never becomes ready
  (slow/low-memory device), the app stays in gallery mode and the status
  reads "sample worlds only on this device" rather than hanging. No broken
  state, ever.

### Pre-baked worlds

A build-time script generates ~12–16 curated worlds spanning the parameter
range (calm continents → rugged archipelagos, low/high coupling, etc.) and
writes them as static GeoJSON under `web/public/worlds/`, plus an
`index.json` manifest (id, headline stats, spec). To bound size, pre-baked
worlds may use coarser coordinate quantization (they are display samples);
target well under ~2MB gzipped total. `GallerySource` fetches the manifest
and lazy-loads worlds as the visitor pages through them.

## Build & deploy

- `scripts/build_static.sh`: build the mimesis_earth wheel → run the pre-bake
  script → `VITE_DEPLOY_TARGET=static npm run build` → assemble `web/dist`
  with the wheel + `worlds/` as static assets.
- A GitHub Actions workflow builds on push to main and publishes `web/dist`
  to GitHub Pages. Vite `base` set to the repo path for Pages URLs.
- The FastAPI server and `mimesis-earth serve` are untouched — they remain
  the local dev/tool path (server build, ApiSource).

## What stays the same

Python core (except the snap fix), the server + CLI, and all existing
frontend interaction code (globe render, drag/zoom/pick, panel, inspect,
export, hover help). Only the world-source layer and the build/deploy
pipeline are added.

## Testing

- Snap-fix: full Python suite green under the new union order; export-validity
  and cross-seed determinism re-confirmed; a Pyodide-parity check (native vs
  WASM byte-identical) added to the spike harness, kept as a manual gate.
- Frontend: `WorldSource` sources unit-tested where practical; the worker
  protocol smoke-tested; a built static bundle verified to (a) show a
  pre-baked world instantly offline-of-Pyodide, (b) reach `ready` and
  generate a live world, (c) degrade to gallery on simulated worker failure.
- Determinism across the boundary is the load-bearing guarantee and gets an
  explicit check in the spike harness.

## Out of scope (future)

- Slimming scipy out of the WASM payload (biggest single asset, 12.6MB) by
  replacing the handful of scipy calls with lighter code — large effort,
  deferred.
- Mobile-specific optimization / deciding whether to attempt Pyodide at all
  on low-end phones (they may live in the gallery).
- Offline / service-worker caching of the Pyodide assets.
- Sharing a specific world by URL (spec+seed in a query param) — cheap and
  attractive as a fast follow, but not in this milestone.
