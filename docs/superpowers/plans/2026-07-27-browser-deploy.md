# In-Browser Generation & GitHub Pages Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship mimesis-earth as a static GitHub Pages showcase: an instant pre-baked globe, upgrading in the background (with clear user communication) to unlimited live in-browser world generation via Pyodide.

**Architecture:** Keep the identical Python core (with one robustness fix) and run it in-browser through Pyodide in a Web Worker. A pluggable `WorldSource` frontend seam swaps between the local API, a pre-baked gallery, and the Pyodide worker; a build-time flag selects server vs static builds. Static build + pre-baked worlds + our wheel deploy to Pages via GitHub Actions.

**Tech Stack:** Python (numpy/scipy/shapely/pydantic), Vite + TypeScript + d3-geo, Pyodide 0.28 (WASM), GitHub Actions + Pages.

**Spec:** `docs/superpowers/specs/2026-07-27-browser-deploy-addendum.md` — read it first. Spike evidence lives there; the snap-before-union fix and byte-identical cross-environment determinism are already proven.

**Baseline:** suite 107 passed; branch `feature/browser-deploy` off main. Generator version currently 0.4.0.

**Note on frontend testing:** per this project's established approach ("frontend: no automated tests, iterated visually"), frontend tasks verify via `tsc --noEmit` + `npm run build` + explicit manual browser checks, not unit tests. Only Task 1 (Python) is full TDD.

---

### Task 1: Snap-before-union robustness fix (generator 0.5.0)

**Why:** Pyodide's GEOS 3.12.1 fatally rejects degenerate near-pole rings during `unary_union`. Snapping each polygon to the precision grid *before* unioning (we currently snap only the result) fixes it and makes output GEOS-version-independent — proven byte-identical across native GEOS 3.13.1 and WASM 3.12.1 in the spike.

**Files:**
- Modify: `python/src/mimesis_earth/geometry.py`
- Modify: `python/src/mimesis_earth/generate.py`
- Modify: `python/src/mimesis_earth/spec.py`
- Modify: `python/tests/test_geometry.py`, `python/tests/test_spec.py`

- [ ] **Step 1: Write the failing test** — append to `python/tests/test_geometry.py`:

```python
def test_snap_union_matches_and_snaps_inputs():
    from mimesis_earth.geometry import PRECISION_GRID, snap_union
    import shapely
    from shapely.geometry import box

    a = box(0, 0, 1, 1)
    b = box(1, 0, 2, 1)  # shares the x=1 edge
    merged = snap_union([a, b])
    assert merged.is_valid
    # union dissolves the shared edge into one 2x1 rectangle
    assert abs(merged.area - 2.0) < 1e-9
    # inputs with sub-grid offsets get snapped, so a near-degenerate sliver
    # collapses rather than producing an invalid noding
    tiny = box(0, 0, 1, 1e-12)  # thinner than the grid
    out = snap_union([tiny])
    assert out.is_valid
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && ../.venv/bin/pytest tests/test_geometry.py::test_snap_union_matches_and_snaps_inputs -v`
Expected: FAIL with `ImportError: cannot import name 'snap_union'`.

- [ ] **Step 3: Implement snap_union in geometry.py** — near the top, after the `R_EARTH_KM` constant, add the grid constant and helper. First confirm the imports: `geometry.py` already has `import shapely` and `from shapely.ops import unary_union` (verify with `grep -n "^import shapely\|from shapely.ops" python/src/mimesis_earth/geometry.py`; if `import shapely` is missing, add it). Then add:

```python
PRECISION_GRID = 1e-9


def snap_union(geoms):
    """Union polygons after snapping each to the precision grid.

    Snapping BEFORE the union (not only after) keeps GEOS's overlay robust to
    degenerate near-pole rings; older GEOS builds — e.g. Pyodide/WASM's
    3.12.x — fatally reject them otherwise. Output is byte-identical across
    GEOS versions (verified 3.12 WASM vs 3.13 native)."""
    return unary_union([shapely.set_precision(g, PRECISION_GRID) for g in geoms])
```

- [ ] **Step 4: Use it at the two multi-cell union sites**

In `python/src/mimesis_earth/geometry.py`, `atoms_polygon`'s return (currently line ~99):

```python
    return shapely.set_precision(_polygons_only(snap_union(geoms)), PRECISION_GRID)
```

In `python/src/mimesis_earth/generate.py`: change the import `from mimesis_earth.geometry import ...` to also import `snap_union` (find the existing geometry import line and add `snap_union` to it), and change the parent-union line (currently line ~140):

```python
            geoms[lvl][i] = shapely.set_precision(snap_union(childs), PRECISION_GRID)
```

Import `PRECISION_GRID` too, or reuse the literal `1e-9` already present — prefer importing `PRECISION_GRID` from geometry for consistency. (Leave the single-cell internal unions in `_normalize_lon`/`_polygons_only` unchanged; they union pieces of one cell and never hit the multi-cell pole degeneracy.)

- [ ] **Step 5: Bump the generator version** — in `python/src/mimesis_earth/spec.py`, change `GENERATOR_VERSION = "0.4.0"` to `"0.5.0"`. Update the two pins in `python/tests/test_spec.py` (`test_new_realism_fields_defaults` and `test_border_meander_field`) from `"0.4.0"` to `"0.5.0"`.

- [ ] **Step 6: Run the new test, then the full suite**

Run: `cd python && ../.venv/bin/pytest tests/test_geometry.py::test_snap_union_matches_and_snaps_inputs -v` → PASS
Run: `cd python && ../.venv/bin/pytest -q` → expect **108 passed** (107 + 1 new). ALL existing invariants (export validity, nesting, determinism, winding) must stay green — the union-order change alters some coordinates in the last digits but preserves every invariant. If a determinism test fails, investigate (it compares same-code runs, so it must still pass); do not weaken it.

- [ ] **Step 7: Commit**

```bash
git add python && git commit -m "fix: snap polygons before union for cross-GEOS robustness; generator 0.5.0"
```

---

### Task 2: WorldSource seam + ApiSource (server build unchanged)

**Why:** Decouple "where a world comes from" so the same UI works with the local API, a pre-baked gallery, or the Pyodide worker. This task is a pure refactor — the server build must behave identically.

**Files:**
- Create: `web/src/sources/types.ts`
- Create: `web/src/sources/api.ts`
- Modify: `web/src/main.ts`
- Modify: `web/src/api.ts` (re-export only)

- [ ] **Step 1: Define the seam** — create `web/src/sources/types.ts`:

```typescript
import type { Spec, WorldData } from '../api'

export type SourceKind = 'api' | 'gallery' | 'pyodide'

// A WorldSource produces worlds for the UI. Live sources (api, pyodide) honor
// the spec; the gallery ignores it and cycles pre-baked worlds.
export interface WorldSource {
  readonly kind: SourceKind
  next(spec: Spec): Promise<WorldData>
}

export type { Spec, WorldData }
```

- [ ] **Step 2: ApiSource** — create `web/src/sources/api.ts`:

```typescript
import { generateWorld } from '../api'
import type { Spec, WorldData, WorldSource } from './types'

export class ApiSource implements WorldSource {
  readonly kind = 'api' as const
  next(spec: Spec): Promise<WorldData> {
    return generateWorld(spec)
  }
}
```

- [ ] **Step 3: Refactor main.ts to use a WorldSource** — in `web/src/main.ts`, replace the direct `generateWorld` import and call:

Change the imports block: remove `import { generateWorld } from './api'`, keep `import type { WorldData } from './api'`, and add:

```typescript
import { ApiSource } from './sources/api'
import type { WorldSource } from './sources/types'
```

After `const globe = ...`, add:

```typescript
let source: WorldSource = new ApiSource()
```

In `newWorld()`, change `world = await generateWorld(readSpec())` to:

```typescript
      world = await source.next(readSpec())
```

- [ ] **Step 4: Verify server build unchanged**

Run: `cd web && npx tsc --noEmit && npm run build`
Expected: clean build. Then run the local server and confirm identical behavior:

```bash
cd /Users/user/Documents/work/mimesis-earth
./scripts/build_web.sh
lsof -ti :8000 | xargs kill 2>/dev/null; sleep 1
.venv/bin/mimesis-earth serve --port 8000 & sleep 3
curl -s -o /dev/null -w "%{http_code}" localhost:8000/           # 200
curl -s -X POST localhost:8000/api/generate -H 'Content-Type: application/json' -d '{"seed":1,"resolution":4000}' -o /dev/null -w "%{http_code}\n"  # 200
kill %1
```

Expected: both 200; the app generates worlds exactly as before (the seam is transparent).

- [ ] **Step 5: Commit**

```bash
git add web && git commit -m "refactor: pluggable WorldSource seam with ApiSource"
```

---

### Task 3: Pre-bake pipeline + GallerySource

**Why:** Ship a set of ready-made worlds so a visitor sees a globe instantly, with no compute and before Pyodide loads.

**Files:**
- Create: `python/scripts/prebake.py`
- Create: `web/src/sources/gallery.ts`
- Create (generated, gitignored): `web/public/worlds/*.json`, `web/public/worlds/index.json`
- Modify: `.gitignore`

- [ ] **Step 1: Pre-bake script** — create `python/scripts/prebake.py`:

```python
"""Generate a curated set of worlds as static JSON for the gallery.

Each file matches the shape the API returns: {"spec": {...}, "levels": [...]}.
Coordinates keep the generator's native 6-dp rounding (already compact)."""

import json
import sys
from pathlib import Path

from mimesis_earth import WorldSpec, generate

# Curated specs spanning the parameter range, showing off variety.
CURATED = [
    {"levels": [6, 5, 6], "n_landmasses": 5, "spread": 0.7, "coast_ruggedness": 0.5,
     "border_meander": 0.8, "seed": 101},
    {"levels": [5, 4, 4], "n_landmasses": 3, "spread": 0.4, "coast_ruggedness": 0.9,
     "size_variance": 0.8, "border_meander": 1.0, "seed": 202},
    {"levels": [8, 4], "n_landmasses": 6, "spread": 1.0, "coast_ruggedness": 1.0,
     "count_coupling": 1.0, "seed": 303},
    {"levels": [4, 4, 3], "n_landmasses": 2, "spread": 0.2, "coast_ruggedness": 0.3,
     "border_meander": 0.4, "seed": 404},
    {"levels": [7, 5, 5], "n_landmasses": 4, "spread": 0.8, "coast_ruggedness": 0.7,
     "size_variance": 0.4, "seed": 505},
    {"levels": [6, 6], "n_landmasses": 8, "spread": 0.9, "coast_ruggedness": 1.0,
     "seed": 606},
    {"levels": [5, 5, 5], "n_landmasses": 3, "spread": 0.5, "coast_ruggedness": 0.6,
     "border_meander": 0.6, "count_variance": 1.0, "seed": 707},
    {"levels": [10, 4], "n_landmasses": 5, "spread": 0.6, "coast_ruggedness": 0.8,
     "count_coupling": 0.9, "seed": 808},
    {"levels": [4, 3, 3], "n_landmasses": 2, "spread": 0.3, "coast_ruggedness": 0.4,
     "seed": 909},
    {"levels": [6, 5, 4], "n_landmasses": 6, "spread": 1.0, "coast_ruggedness": 0.95,
     "size_variance": 1.2, "border_meander": 0.9, "seed": 111},
    {"levels": [5, 4], "n_landmasses": 4, "spread": 0.55, "coast_ruggedness": 0.5,
     "seed": 222},
    {"levels": [7, 4, 4], "n_landmasses": 3, "spread": 0.65, "coast_ruggedness": 0.75,
     "border_meander": 0.85, "seed": 333},
]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, kwargs in enumerate(CURATED):
        spec = WorldSpec(**kwargs)
        world = generate(spec)
        payload = {
            "spec": spec.model_dump(),
            "levels": [
                {"level": lvl, "name": name, "geojson": world.geojson_dict(lvl)}
                for lvl, name in enumerate(spec.level_names)
            ],
        }
        fname = f"world-{i:02d}.json"
        (out / fname).write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        top = world.units_at(0)
        manifest.append({
            "file": fname,
            "seed": spec.seed,
            "countries": len(top),
            "landmasses": spec.n_landmasses,
        })
    (out / "index.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"pre-baked {len(CURATED)} worlds to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web/public/worlds")
```

- [ ] **Step 2: Run it and check sizes**

```bash
cd /Users/user/Documents/work/mimesis-earth
.venv/bin/python python/scripts/prebake.py web/public/worlds
du -sh web/public/worlds
ls web/public/worlds
```

Expected: `world-00.json`…`world-11.json` + `index.json`; total on the order of a few MB (served gzipped by Pages). If total exceeds ~8MB uncompressed, reduce the count or lower `resolution` in a few specs (add `"resolution": 12000`), and note it.

- [ ] **Step 3: Gitignore the generated worlds** — append to `.gitignore`:

```
web/public/worlds/
```

- [ ] **Step 4: GallerySource** — create `web/src/sources/gallery.ts`:

```typescript
import type { Spec, WorldData, WorldSource } from './types'

interface ManifestEntry {
  file: string
}

// Serves pre-baked worlds in sequence. next() ignores the spec and advances
// to the next world (wrapping). Worlds are fetched lazily and cached.
export class GallerySource implements WorldSource {
  readonly kind = 'gallery' as const
  private files: string[] = []
  private cache = new Map<string, WorldData>()
  private index = -1
  private ready: Promise<void>

  constructor(private baseUrl: string) {
    this.ready = this.loadManifest()
  }

  private async loadManifest(): Promise<void> {
    const resp = await fetch(`${this.baseUrl}worlds/index.json`)
    const manifest = (await resp.json()) as ManifestEntry[]
    this.files = manifest.map((m) => m.file)
  }

  private async load(file: string): Promise<WorldData> {
    const hit = this.cache.get(file)
    if (hit) return hit
    const resp = await fetch(`${this.baseUrl}worlds/${file}`)
    const data = (await resp.json()) as WorldData
    this.cache.set(file, data)
    return data
  }

  async next(_spec: Spec): Promise<WorldData> {
    await this.ready
    if (this.files.length === 0) throw new Error('no pre-baked worlds')
    this.index = (this.index + 1) % this.files.length
    return this.load(this.files[this.index])
  }
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no errors. (Full wiring into main.ts happens in Task 5.)

- [ ] **Step 6: Commit**

```bash
git add python/scripts/prebake.py web/src/sources/gallery.ts .gitignore
git commit -m "feat: pre-bake pipeline and GallerySource"
```

---

### Task 4: Pyodide Web Worker + PyodideSource

**Why:** Run the real Python generator client-side without freezing the UI.

**Files:**
- Create: `web/public/pyodide-worker.js` (plain JS, not Vite-processed)
- Create: `web/src/sources/pyodide.ts`

- [ ] **Step 1: The worker** — create `web/public/pyodide-worker.js`. It is a classic worker loaded from `public/` so Vite does not transform it; it loads Pyodide from the official CDN and installs our wheel from the same origin:

```javascript
// Pyodide worker: loads the scientific stack + our wheel, then generates
// worlds off the main thread. Posts progress during load, then {type:'ready'}.
const PYODIDE_VERSION = '0.28.0'
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`
importScripts(`${CDN}pyodide.js`)

let pyodide = null
let generate = null // a JS-callable Python function

function progress(phase, pct) {
  self.postMessage({ type: 'progress', phase, pct })
}

async function init(baseUrl) {
  progress('loading engine', 10)
  pyodide = await loadPyodide({ indexURL: CDN })
  progress('loading scientific libraries', 35)
  await pyodide.loadPackage(['numpy', 'scipy', 'shapely', 'pydantic', 'micropip'])
  progress('installing generator', 75)
  // discover the wheel filename from the manifest, then micropip-install it
  const manifest = await (await fetch(`${baseUrl}wheels/manifest.json`)).json()
  const wheelUrl = new URL(`wheels/${manifest.wheel}`, baseUrl).href
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(wheelUrl)})
`)
  progress('starting up', 95)
  // build a reusable generate() that takes a spec dict and returns the payload
  generate = pyodide.runPython(`
import json
from mimesis_earth import WorldSpec, generate as _gen

def _generate(spec_json):
    spec = WorldSpec(**json.loads(spec_json))
    w = _gen(spec)
    payload = {
        "spec": spec.model_dump(),
        "levels": [
            {"level": l, "name": n, "geojson": w.geojson_dict(l)}
            for l, n in enumerate(spec.level_names)
        ],
    }
    return json.dumps(payload)
_generate
`)
  self.postMessage({ type: 'ready' })
}

self.onmessage = async (e) => {
  const msg = e.data
  try {
    if (msg.type === 'init') {
      await init(msg.baseUrl)
    } else if (msg.type === 'generate') {
      const out = generate(JSON.stringify(msg.spec))
      self.postMessage({ type: 'world', id: msg.id, payload: out })
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err) })
  }
}
```

- [ ] **Step 2: PyodideSource** — create `web/src/sources/pyodide.ts`:

```typescript
import type { Spec, WorldData, WorldSource } from './types'

type ProgressCb = (phase: string, pct: number) => void

// Owns the Pyodide worker. Boots on construction; resolves whenReady when the
// worker signals ready. next() round-trips a spec through the worker.
export class PyodideSource implements WorldSource {
  readonly kind = 'pyodide' as const
  private worker: Worker
  private seq = 0
  private pending = new Map<
    number,
    { resolve: (w: WorldData) => void; reject: (e: Error) => void }
  >()
  readonly whenReady: Promise<void>

  constructor(baseUrl: string, onProgress: ProgressCb) {
    this.worker = new Worker(`${baseUrl}pyodide-worker.js`)
    this.whenReady = new Promise<void>((resolve, reject) => {
      this.worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          onProgress(msg.phase, msg.pct)
        } else if (msg.type === 'ready') {
          resolve()
        } else if (msg.type === 'world') {
          this.pending.get(msg.id)?.resolve(JSON.parse(msg.payload) as WorldData)
          this.pending.delete(msg.id)
        } else if (msg.type === 'error') {
          const p = this.pending.get(msg.id)
          if (p) {
            p.reject(new Error(msg.message))
            this.pending.delete(msg.id)
          } else {
            reject(new Error(msg.message)) // error during init
          }
        }
      }
      this.worker.onerror = (e) => reject(new Error(e.message))
    })
    this.worker.postMessage({ type: 'init', baseUrl })
  }

  next(spec: Spec): Promise<WorldData> {
    const id = ++this.seq
    return new Promise<WorldData>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ type: 'generate', id, spec })
    })
  }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/public/pyodide-worker.js web/src/sources/pyodide.ts
git commit -m "feat: Pyodide web worker and PyodideSource"
```

---

### Task 5: Loading UX + gallery→pyodide orchestration

**Why:** Tie the sources together for the static build with the required loading communication: instant gallery, dimmed controls + "loading" status, progress, a "ready" notification, and graceful fallback.

**Files:**
- Modify: `web/index.html` (status + notification elements)
- Modify: `web/src/style.css` (dimmed panel, status, notification)
- Modify: `web/src/main.ts` (orchestration behind the build flag)
- Modify: `web/src/panel.ts` (a helper to enable/disable the sliders)

- [ ] **Step 1: DOM for status + notification** — in `web/index.html`, immediately after the closing `</aside>` of the panel, add:

```html
    <div id="load-status" hidden></div>
    <div id="ready-toast" hidden>✓ live generation ready — press space to build a world from your settings<button id="ready-dismiss" aria-label="dismiss">×</button></div>
```

- [ ] **Step 2: Styles** — append to `web/src/style.css`:

```css
#load-status {
  position: fixed;
  top: 24px;
  left: 234px;
  max-width: 320px;
  font-size: 13px;
  color: var(--dim);
  font-style: italic;
}
#panel.dimmed { opacity: 0.5; pointer-events: none; }
#panel.dimmed .panel-title::after { content: ' (loading…)'; }
#ready-toast {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--card);
  border: 1px solid var(--ink);
  padding: 10px 14px;
  font-size: 13px;
  color: var(--ink);
  display: flex;
  gap: 10px;
  align-items: center;
}
#ready-toast button {
  background: none; border: none; cursor: pointer;
  font-size: 16px; color: var(--dim); line-height: 1;
}
```

(Confirm the panel's title element carries a `panel-title` class; if not, adjust the selector to match the actual title element in `index.html`.)

- [ ] **Step 3: Panel enable/disable helper** — in `web/src/panel.ts`, add:

```typescript
export function setPanelEnabled(enabled: boolean): void {
  const panel = document.getElementById('panel')!
  panel.classList.toggle('dimmed', !enabled)
}
```

- [ ] **Step 4: Orchestrate in main.ts** — the static build starts on the gallery and upgrades to Pyodide. Gate on the build flag so the server build is untouched. In `web/src/main.ts`:

Add imports:

```typescript
import { GallerySource } from './sources/gallery'
import { PyodideSource } from './sources/pyodide'
import { setPanelEnabled } from './panel'
```

Replace the `let source: WorldSource = new ApiSource()` line with build-flag-aware setup. After the existing top-level consts, add:

```typescript
const STATIC = import.meta.env.VITE_DEPLOY_TARGET === 'static'
const BASE = import.meta.env.BASE_URL // trailing-slash base, subpath-safe
const loadStatus = document.getElementById('load-status')!
const readyToast = document.getElementById('ready-toast')!

function startPyodideUpgrade(): void {
  const py = new PyodideSource(BASE, (phase, pct) => {
    loadStatus.textContent = `${phase}… ${pct}%`
  })
  py.whenReady
    .then(() => {
      source = py
      setPanelEnabled(true)
      loadStatus.hidden = true
      readyToast.hidden = false
      updateHint(true)
    })
    .catch(() => {
      // graceful degradation: stay on the gallery
      loadStatus.textContent = 'sample worlds only on this device'
    })
}
```

And set the initial source + kick off the upgrade at the bottom of the file, replacing the current `let source: WorldSource = new ApiSource()` and the final `initParamHelp(); void newWorld()`:

```typescript
let source: WorldSource
if (STATIC) {
  source = new GallerySource(BASE)
  setPanelEnabled(false)
  loadStatus.hidden = false
  loadStatus.textContent = 'browsing sample worlds · live generator loading…'
  updateHint(false)
  startPyodideUpgrade()
} else {
  source = new ApiSource()
}
initParamHelp()
void newWorld()
```

Add the hint helper (updates the bottom-center hint line; find the hint element id in index.html — it is `hint`):

```typescript
function updateHint(live: boolean): void {
  const hint = document.getElementById('hint')
  if (!hint) return
  const first = live ? 'space — new world' : 'space — next sample world'
  hint.textContent = `${first} · drag — rotate · scroll — zoom · click — inspect`
}
```

Wire the toast dismiss button near the export handler:

```typescript
document.getElementById('ready-dismiss')?.addEventListener('click', () => {
  readyToast.hidden = true
})
```

Note: `source` is now assigned before use in `newWorld`; since `let source: WorldSource` may be used before assignment in TS's eyes, declare it with a definite-assignment note or assign in both branches (both branches above assign it, and they run before `newWorld()`), so use `let source!: WorldSource` if tsc complains about definite assignment.

- [ ] **Step 5: Verify both builds compile**

Run: `cd web && npx tsc --noEmit && npm run build` (server build) → clean.
Run: `cd web && VITE_DEPLOY_TARGET=static npx vite build` → clean (static build).

- [ ] **Step 6: Commit**

```bash
git add web && git commit -m "feat: gallery→pyodide loading orchestration and UX"
```

---

### Task 6: Static build script + Vite config + GitHub Pages deploy

**Why:** Produce the deployable static bundle and publish it.

**Files:**
- Modify: `web/vite.config.ts`
- Create: `scripts/build_static.sh`
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Vite base + env** — in `web/vite.config.ts`, set a relative base for the static build so it works under a Pages subpath. Read the current file first; add `base: './'` to the config object (relative base makes `import.meta.env.BASE_URL` resolve correctly and all runtime fetches subpath-safe). Keep the existing `/api` dev proxy. Example shape:

```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
```

- [ ] **Step 2: Static build script** — create `scripts/build_static.sh`:

```bash
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
```

```bash
chmod +x scripts/build_static.sh
./scripts/build_static.sh
```

Expected: prints the staged wheel, pre-bakes worlds, builds `web/dist`. Confirm `web/dist/wheels/manifest.json`, `web/dist/worlds/index.json`, and `web/dist/pyodide-worker.js` all exist (public/ assets are copied into dist by Vite).

- [ ] **Step 3: Local static smoke test** — serve `web/dist` as pure static files (no API) and confirm the hybrid works end to end:

```bash
cd web/dist && python3 -m http.server 8090 & sleep 1
curl -s -o /dev/null -w "%{http_code}\n" localhost:8090/            # 200 index
curl -s -o /dev/null -w "%{http_code}\n" localhost:8090/worlds/index.json  # 200
curl -s -o /dev/null -w "%{http_code}\n" localhost:8090/pyodide-worker.js  # 200
curl -s -o /dev/null -w "%{http_code}\n" localhost:8090/wheels/manifest.json # 200
kill %1
```

Expected: all 200. (Full interactive verification — instant gallery, Pyodide upgrade, ready toast — happens in the manual browser check below.)

- [ ] **Step 4: GitHub Actions Pages workflow** — create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - name: Set up venv and install package
        run: |
          python -m venv .venv
          .venv/bin/pip install -e './python[dev]'
      - name: Install web deps
        run: cd web && npm ci
      - name: Build static bundle
        run: ./scripts/build_static.sh
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web/dist
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 5: Manual browser verification** (the real acceptance test) — serve `web/dist` statically (`cd web/dist && python3 -m http.server 8090`), open `http://localhost:8090` in a browser, and confirm:
  1. A pre-baked globe appears within ~1s; panel is dimmed; status reads "browsing sample worlds · live generator loading…".
  2. Spacebar before ready cycles to another pre-baked world; hint reads "space — next sample world".
  3. Status shows progress phases; after ~10–20s the "✓ live generation ready" toast appears, the panel un-dims, hint switches to "space — new world".
  4. Adjusting sliders + spacebar now generates a *live* world (different from the pre-baked ones), and the same seed+settings reproduces identically.
  5. Console has no fatal errors.

Report the observed load time and first-live-generation time.

- [ ] **Step 6: Commit**

```bash
git add web/vite.config.ts scripts/build_static.sh .github/workflows/deploy.yml
git commit -m "feat: static build pipeline and GitHub Pages deploy workflow"
```

---

## Self-review notes (applied)

- **Spec coverage:** Pyodide-not-TS-port (whole plan); snap-before-union prerequisite + version bump (Task 1); WorldSource seam with api/gallery/pyodide + build-flag server/static (Tasks 2,3,4,5); Web Worker + protocol (Task 4); loading UX with dimming/status/progress/ready-toast/graceful-degradation (Task 5); pre-bake pipeline + manifest (Task 3); static build + CDN-for-Pyodide/self-host-wheel-and-worlds + Pages Actions (Task 6); server + CLI untouched (Task 2 verifies unchanged). Determinism-across-boundary is proven in the spike and re-checkable via the retained harness.
- **Type consistency:** `WorldSource.next(spec): Promise<WorldData>` used identically by ApiSource, GallerySource, PyodideSource, and main.ts; `PyodideSource.whenReady` and the `{type:'progress'|'ready'|'world'|'error'}` protocol match between worker and source; `Spec`/`WorldData` imported from `../api` throughout; `snap_union`/`PRECISION_GRID` defined in geometry.py and imported by generate.py.
- **Deferred (from spec, not in this plan):** slimming scipy, mobile-specific handling, offline/service-worker caching, URL-sharing a world.
- **Sharp edges flagged inline:** verify `import shapely` present in geometry.py (Task 1); confirm the `panel-title`/`hint` element ids match index.html (Task 5); `let source!: WorldSource` if tsc flags definite assignment (Task 5); pre-baked total-size guard with a resolution fallback (Task 3); relative Vite base for subpath-safe Pages URLs (Task 6).
