import './style.css'
import type { WorldData } from './api'
import { ApiSource } from './sources/api'
import { GallerySource } from './sources/gallery'
import { PyodideSource } from './sources/pyodide'
import type { WorldSource } from './sources/types'
import { Globe } from './globe'
import { hideInspect, showInspect } from './inspect'
import { downloadWorld } from './exporter'
import { initParamHelp } from './help'
import { isTypingInPanel, maybeRandomizeSeed, readSpec, setPanelEnabled } from './panel'

const globe = new Globe(document.getElementById('globe') as HTMLCanvasElement)
const statusEl = document.getElementById('status')!
const levelsNav = document.getElementById('levels')!
const rotationToggle = document.getElementById('rotation-toggle')!
let rotationEnabled = true

rotationToggle.addEventListener('click', () => {
  rotationEnabled = !rotationEnabled
  globe.setAutoRotation(rotationEnabled)
  rotationToggle.textContent = rotationEnabled ? 'stop rotation' : 'start rotation'
})

const STATIC = import.meta.env.VITE_DEPLOY_TARGET === 'static'
const BASE = new URL(import.meta.env.BASE_URL, location.href).href // absolute, subpath-safe
const loadStatus = document.getElementById('load-status')!
const readyToast = document.getElementById('ready-toast')!

let world: WorldData | null = null
let levelIndex = 0
let busy = false
let statusTimer: number | undefined

globe.onPick = (f) => (f ? showInspect(f) : hideInspect())

function setLevel(i: number): void {
  if (!world) return
  levelIndex = Math.max(0, Math.min(i, world.levels.length - 1))
  globe.setWorld(world.levels[levelIndex].geojson)
  hideInspect()
  renderLevelNav()
}

function renderLevelNav(): void {
  if (!world) return
  levelsNav.innerHTML = ''
  world.levels.forEach((lvl, i) => {
    const btn = document.createElement('button')
    btn.textContent = lvl.name
    btn.className = i === levelIndex ? 'active' : ''
    btn.addEventListener('click', () => setLevel(i))
    levelsNav.appendChild(btn)
  })
}

async function newWorld(): Promise<void> {
  if (busy) return
  busy = true
  statusEl.textContent = 'generating…'
  try {
    clearTimeout(statusTimer)
    statusEl.hidden = false
    try {
      world = await source.next(readSpec())
      setLevel(Math.min(levelIndex, world.levels.length - 1))
    } catch (err) {
      statusEl.hidden = false
      statusEl.textContent = String(err)
      statusTimer = window.setTimeout(() => {
        statusEl.hidden = true
        statusEl.textContent = 'generating…'
      }, 4000)
      return
    }
    statusEl.hidden = true
  } finally {
    busy = false
  }
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !isTypingInPanel(e.target)) {
    e.preventDefault()
    maybeRandomizeSeed()
    void newWorld()
  }
})

document.getElementById('export')!.addEventListener('click', () => {
  if (world) downloadWorld(world)
})

document.getElementById('ready-dismiss')?.addEventListener('click', () => {
  readyToast.hidden = true
})

function updateHint(live: boolean): void {
  const hint = document.getElementById('hint')
  if (!hint) return
  const first = live ? 'space — new world' : 'space — next sample world'
  hint.textContent = `${first} · drag — rotate · scroll — zoom · click — inspect`
}

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
