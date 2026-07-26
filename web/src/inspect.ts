import type { Feature } from 'geojson'

const el = (id: string) => document.getElementById(id)!

export function showInspect(f: Feature): void {
  const p = f.properties ?? {}
  el('inspect-name').textContent = String(p.name ?? '?')
  el('inspect-id').textContent = String(p.id ?? '')
  el('inspect-pop').textContent = `pop   ${Number(p.population ?? 0).toLocaleString()}`
  el('inspect-area').textContent = `area  ${Math.round(Number(p.area_km2 ?? 0)).toLocaleString()} km²`
  el('inspect').hidden = false
}

export function hideInspect(): void {
  el('inspect').hidden = true
}
