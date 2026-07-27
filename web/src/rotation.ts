const DEGREES_PER_MILLISECOND = 2 / 1000

function wrapLongitude(longitude: number): number {
  return ((((longitude + 180) % 360) + 360) % 360) - 180
}

export class AutoRotation {
  private enabled = true
  private interacting = false
  private lastTimestamp: number | null = null

  get active(): boolean {
    return this.enabled && !this.interacting
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled
    this.lastTimestamp = null
  }

  setInteracting(interacting: boolean): void {
    this.interacting = interacting
    this.lastTimestamp = null
  }

  advance(longitude: number, timestamp: number): number {
    if (!this.active) {
      this.lastTimestamp = null
      return longitude
    }
    if (this.lastTimestamp === null) {
      this.lastTimestamp = timestamp
      return longitude
    }
    const elapsed = timestamp - this.lastTimestamp
    this.lastTimestamp = timestamp
    return wrapLongitude(longitude + elapsed * DEGREES_PER_MILLISECOND)
  }
}
