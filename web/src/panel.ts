import type { Spec } from './api'

const $ = (id: string) => document.getElementById(id) as HTMLInputElement

export function readSpec(): Spec {
  const levels = $('p-levels')
    .value.split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0)
  return {
    levels: levels.length ? levels : [6, 5, 6],
    n_landmasses: parseInt($('p-islands').value, 10) || 3,
    spread: parseFloat($('p-spread').value),
    coast_ruggedness: parseFloat($('p-coast').value),
    land_fraction: parseFloat($('p-land').value),
    border_roughness: parseFloat($('p-borders').value),
    size_variance: parseFloat($('p-sizes').value),
    count_coupling: parseFloat($('p-coupling').value),
    count_variance: parseFloat($('p-counts').value),
    border_meander: parseFloat($('p-meander').value),
    total_population: parseInt($('p-pop').value, 10) || 50_000_000,
    resolution: parseInt($('p-res').value, 10) || 20_000,
    seed: parseInt($('p-seed').value, 10) || 0,
  }
}

export function maybeRandomizeSeed(): void {
  if ($('p-autoseed').checked) {
    $('p-seed').value = String(Math.floor(Math.random() * 1_000_000))
  }
}

export function isTypingInPanel(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLInputElement &&
    (target.type === 'text' || target.type === 'number')
  )
}

export function setPanelEnabled(enabled: boolean): void {
  const panel = document.getElementById('panel')!
  panel.classList.toggle('dimmed', !enabled)
}
