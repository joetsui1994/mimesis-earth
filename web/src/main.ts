import './style.css'
import { generateWorld } from './api'
import type { WorldData } from './api'
import { Globe } from './globe'
import { hideInspect, showInspect } from './inspect'
import { downloadWorld } from './exporter'
import { initParamHelp } from './help'
import { isTypingInPanel, maybeRandomizeSeed, readSpec } from './panel'

const globe = new Globe(document.getElementById('globe') as HTMLCanvasElement)
const statusEl = document.getElementById('status')!
const levelsNav = document.getElementById('levels')!

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
      world = await generateWorld(readSpec())
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

initParamHelp()
void newWorld()
