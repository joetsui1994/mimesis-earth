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
