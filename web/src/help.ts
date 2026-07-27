const HELP: Record<string, string> = {
  'p-levels':
    'Children per parent at each admin level — "6,5,6" means 6 countries, ~5 provinces each, ~6 districts each.',
  'p-islands':
    'Number of separate landmass groups. Each group is guaranteed to exist and hosts at least one country.',
  'p-spread':
    'How scattered the landmasses are: 0 clusters them in one region of the globe, 1 spreads them worldwide.',
  'p-coast':
    'Relief. Mountain amplitude in the terrain: higher values carve ragged coastlines, island arcs, and highlands.',
  'p-borders':
    'Fine-scale wiggliness of internal borders (jitter at the smallest map scale).',
  'p-sizes':
    'Size inequality between units: 0 makes them evenly sized, higher values give a few sprawling giants among many small ones.',
  'p-coupling':
    'How strongly a territory’s subdivision count follows its size — high values give big countries many provinces and small ones few.',
  'p-counts':
    'Random variation in subdivision counts between parents. Per-level totals stay exact regardless.',
  'p-meander':
    'How strongly borders follow terrain ridges, wandering at large scales instead of running straight.',
  'p-pop':
    'Total world population, distributed unevenly across districts with urban hotspots.',
  'p-res':
    'Number of tiles the sphere is built from: finer coastlines and borders, but slower generation.',
  'p-seed':
    'World number — the same seed with the same settings always reproduces the identical world. The dice toggle rerolls the seed on every spacebar press.',
}

export function initParamHelp(): void {
  const panel = document.getElementById('panel')!
  const help = document.getElementById('param-help')!
  panel.querySelectorAll('label').forEach((label) => {
    const input = label.querySelector('input')
    const text = input ? HELP[input.id] : undefined
    if (!text) return
    label.addEventListener('mouseenter', () => {
      help.textContent = text
      help.hidden = false
    })
    label.addEventListener('mouseleave', () => {
      help.hidden = true
    })
  })
}
