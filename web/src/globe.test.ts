// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { Globe } from './globe'

function pointerEvent(type: string, pointerId = 1): Event {
  const event = new Event(type)
  Object.defineProperty(event, 'pointerId', { value: pointerId })
  Object.defineProperty(event, 'clientX', { value: 400 })
  Object.defineProperty(event, 'clientY', { value: 300 })
  return event
}

function createGlobe() {
  const context = {
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    scale: vi.fn(),
    clearRect: vi.fn(),
  } as unknown as CanvasRenderingContext2D

  const canvas = document.createElement('canvas')
  vi.spyOn(canvas, 'getContext').mockReturnValue(context)
  canvas.setPointerCapture = vi.fn()

  let nextFrame = 0
  const requestFrame = vi.fn(() => ++nextFrame)
  const cancelFrame = vi.fn()
  vi.stubGlobal('requestAnimationFrame', requestFrame)
  vi.stubGlobal('cancelAnimationFrame', cancelFrame)

  new Globe(canvas)
  return { canvas, requestFrame, cancelFrame }
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Globe pointer lifecycle', () => {
  it.each(['pointercancel', 'lostpointercapture'])(
    'resumes automatic rotation after %s',
    (endEvent) => {
      const { canvas, requestFrame, cancelFrame } = createGlobe()

      canvas.dispatchEvent(pointerEvent('pointerdown'))
      expect(cancelFrame).toHaveBeenCalledWith(1)

      canvas.dispatchEvent(pointerEvent(endEvent))
      expect(requestFrame).toHaveBeenCalledTimes(2)
    },
  )
})
