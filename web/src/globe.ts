import { geoContains, geoGraticule10, geoOrthographic, geoPath } from 'd3-geo'
import type { GeoProjection } from 'd3-geo'
import type { Feature, FeatureCollection } from 'geojson'

const INK = '#2f3a45'
const SPHERE = '#f1ecdf'
const LAND = '#e5decb'
const GRID = '#cfc8b6'
const HILITE = 'rgba(214, 185, 140, 0.65)'

export function reverseWinding(fc: FeatureCollection): FeatureCollection {
  const flip = (rings: number[][][]) => rings.map((r) => [...r].reverse())
  return {
    ...fc,
    features: fc.features.map((f) => {
      const g = f.geometry
      if (!g || (g.type !== 'Polygon' && g.type !== 'MultiPolygon')) return f
      const coordinates =
        g.type === 'Polygon'
          ? flip(g.coordinates as number[][][])
          : (g.coordinates as number[][][][]).map(flip)
      return { ...f, geometry: { ...g, coordinates } as typeof g }
    }),
  }
}

export class Globe {
  private ctx: CanvasRenderingContext2D
  private projection: GeoProjection
  private rotation: [number, number] = [20, -15]
  private zoomFactor = 1
  private fc: FeatureCollection | null = null
  private hitFc: FeatureCollection | null = null
  private width = 0
  private height = 0
  private dpr = 1
  private rafPending = false
  selected: Feature | null = null
  onPick: (f: Feature | null) => void = () => {}

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!
    this.projection = geoOrthographic().clipAngle(90)
    window.addEventListener('resize', () => this.resize())
    this.bindPointer()
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault()
        this.zoomFactor = Math.min(
          60,
          Math.max(0.7, this.zoomFactor * Math.exp(-e.deltaY * 0.0015)),
        )
        this.scheduleDraw()
      },
      { passive: false },
    )
    this.resize()
  }

  setWorld(fc: FeatureCollection) {
    this.fc = fc
    this.hitFc = reverseWinding(fc)
    this.selected = null
    this.canvas.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 220 })
    this.draw()
  }

  setSelected(f: Feature | null) {
    this.selected = f
    this.draw()
  }

  private resize() {
    this.dpr = window.devicePixelRatio || 1
    this.width = window.innerWidth
    this.height = window.innerHeight
    this.canvas.width = this.width * this.dpr
    this.canvas.height = this.height * this.dpr
    this.canvas.style.width = `${this.width}px`
    this.canvas.style.height = `${this.height}px`
    this.draw()
  }

  private bindPointer() {
    let dragging = false
    let moved = false
    let last: [number, number] = [0, 0]
    this.canvas.addEventListener('pointerdown', (e) => {
      dragging = true
      moved = false
      last = [e.clientX, e.clientY]
      this.canvas.setPointerCapture(e.pointerId)
      this.canvas.classList.add('dragging')
    })
    this.canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return
      const dx = e.clientX - last[0]
      const dy = e.clientY - last[1]
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true
      last = [e.clientX, e.clientY]
      const k = 90 / this.projection.scale()
      this.rotation = [
        this.rotation[0] + dx * k,
        Math.max(-90, Math.min(90, this.rotation[1] - dy * k)),
      ]
      this.scheduleDraw()
    })
    this.canvas.addEventListener('pointerup', (e) => {
      dragging = false
      this.canvas.classList.remove('dragging')
      if (!moved) this.pick(e.clientX, e.clientY)
    })
  }

  private pick(x: number, y: number) {
    if (!this.fc || !this.hitFc) return
    const ll = this.projection.invert?.([x, y]) ?? null
    if (!ll) {
      this.setSelected(null)
      this.onPick(null)
      return
    }
    // invert() returns a point even outside the disc; verify it re-projects nearby
    const back = this.projection([ll[0], ll[1]])
    if (!back || Math.hypot(back[0] - x, back[1] - y) > 2) {
      this.setSelected(null)
      this.onPick(null)
      return
    }
    for (let i = 0; i < this.hitFc.features.length; i++) {
      if (geoContains(this.hitFc.features[i], [ll[0], ll[1]])) {
        const f = this.fc.features[i]
        this.setSelected(f)
        this.onPick(f)
        return
      }
    }
    this.setSelected(null)
    this.onPick(null)
  }

  private scheduleDraw() {
    if (this.rafPending) return
    this.rafPending = true
    requestAnimationFrame(() => {
      this.rafPending = false
      this.draw()
    })
  }

  draw() {
    const { ctx } = this
    const scale = 0.42 * Math.min(this.width, this.height) * this.zoomFactor
    this.projection
      .scale(scale)
      .translate([this.width / 2, this.height / 2])
      .rotate([this.rotation[0], this.rotation[1], 0])
    const path = geoPath(this.projection, ctx)

    ctx.save()
    ctx.scale(this.dpr, this.dpr)
    ctx.clearRect(0, 0, this.width, this.height)
    ctx.lineJoin = 'round'

    // sphere
    ctx.beginPath()
    path({ type: 'Sphere' })
    ctx.fillStyle = SPHERE
    ctx.fill()

    // graticule
    ctx.beginPath()
    path(geoGraticule10())
    ctx.strokeStyle = GRID
    ctx.lineWidth = 0.5
    ctx.stroke()

    // land units
    if (this.fc) {
      for (const f of this.fc.features) {
        ctx.beginPath()
        path(f)
        ctx.fillStyle = f === this.selected ? HILITE : LAND
        ctx.fill()
        ctx.strokeStyle = INK
        ctx.lineWidth = 0.7
        ctx.stroke()
      }
    }

    // sphere outline on top
    ctx.beginPath()
    path({ type: 'Sphere' })
    ctx.strokeStyle = INK
    ctx.lineWidth = 1.4
    ctx.stroke()
    ctx.restore()
  }
}
