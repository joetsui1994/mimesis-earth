import type { FeatureCollection } from 'geojson'

export interface Spec {
  levels: number[]
  n_landmasses: number
  spread: number
  coast_ruggedness: number
  border_roughness: number
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
    throw new Error(`generate failed (${resp.status}): ${body}`)
  }
  return resp.json()
}
