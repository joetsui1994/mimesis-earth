import type { FeatureCollection } from 'geojson'

export interface Spec {
  levels: number[]
  n_landmasses: number
  spread: number
  coast_ruggedness: number
  border_roughness: number
  size_variance: number
  count_coupling: number
  count_variance: number
  border_meander: number
  total_population: number
  resolution: number
  seed: number
}

export interface LevelData {
  level: number
  name: string
  geojson: FeatureCollection
}

export interface WorldData {
  spec: Spec & { level_names: string[] }
  levels: LevelData[]
}

export async function generateWorld(spec: Spec): Promise<WorldData> {
  const resp = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  })
  if (!resp.ok) {
    const body = await resp.text()
    let message = body
    try {
      const parsed = JSON.parse(body)
      const first = parsed?.detail?.[0]
      if (first?.msg) {
        const field = Array.isArray(first.loc) ? first.loc.slice(1).join('.') : ''
        message = field ? `${field}: ${first.msg}` : first.msg
      }
    } catch {
      // keep raw body
    }
    throw new Error(`generate failed (${resp.status}): ${message}`)
  }
  return resp.json()
}
