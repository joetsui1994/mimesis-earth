import { generateWorld } from '../api'
import type { Spec, WorldData, WorldSource } from './types'

export class ApiSource implements WorldSource {
  readonly kind = 'api' as const
  next(spec: Spec): Promise<WorldData> {
    return generateWorld(spec)
  }
}
