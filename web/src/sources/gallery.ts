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
