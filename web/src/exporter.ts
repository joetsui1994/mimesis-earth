import { strToU8, zipSync } from 'fflate'
import type { WorldData } from './api'

export function downloadWorld(world: WorldData): void {
  const files: Record<string, Uint8Array> = {}
  for (const lvl of world.levels) {
    files[`level${lvl.level}_${lvl.name}.geojson`] = strToU8(
      JSON.stringify(lvl.geojson),
    )
  }
  files['spec.json'] = strToU8(JSON.stringify(world.spec, null, 2))

  const rows = [
    'id,level,level_name,parent_id,name,population,area_km2,centroid_lon,centroid_lat,landmass',
  ]
  for (const lvl of world.levels) {
    for (const f of lvl.geojson.features) {
      const p = f.properties ?? {}
      rows.push(
        [
          p.id,
          p.level,
          p.level_name,
          p.parent_id ?? '',
          p.name,
          p.population,
          p.area_km2,
          p.centroid_lon,
          p.centroid_lat,
          p.landmass ?? '',
        ].join(','),
      )
    }
  }
  files['units.csv'] = strToU8(rows.join('\n'))

  const zip = zipSync(files)
  const blob = new Blob([zip], { type: 'application/zip' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `world-seed${world.spec.seed}.zip`
  a.click()
  URL.revokeObjectURL(a.href)
}
