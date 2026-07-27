import type { Spec, WorldData } from '../api'

export type SourceKind = 'api' | 'gallery' | 'pyodide'

// A WorldSource produces worlds for the UI. Live sources (api, pyodide) honor
// the spec; the gallery ignores it and cycles pre-baked worlds.
export interface WorldSource {
  readonly kind: SourceKind
  next(spec: Spec): Promise<WorldData>
}

export type { Spec, WorldData }
