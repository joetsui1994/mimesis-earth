import { describe, expect, it } from 'vitest'
import { AutoRotation } from './rotation'

describe('AutoRotation', () => {
  it('advances longitude at two degrees per second and wraps at 180', () => {
    const rotation = new AutoRotation()

    expect(rotation.advance(179, 1000)).toBe(179)
    expect(rotation.advance(179, 2000)).toBe(-179)
  })

  it('stops and restarts without counting paused time', () => {
    const rotation = new AutoRotation()

    rotation.advance(20, 1000)
    rotation.setEnabled(false)
    expect(rotation.advance(20, 5000)).toBe(20)

    rotation.setEnabled(true)
    expect(rotation.advance(20, 9000)).toBe(20)
    expect(rotation.advance(20, 10000)).toBe(22)
  })

  it('temporarily pauses during interaction without counting paused time', () => {
    const rotation = new AutoRotation()

    rotation.advance(20, 1000)
    rotation.setInteracting(true)
    expect(rotation.advance(35, 5000)).toBe(35)

    rotation.setInteracting(false)
    expect(rotation.advance(35, 9000)).toBe(35)
    expect(rotation.advance(35, 10000)).toBe(37)
  })
})
