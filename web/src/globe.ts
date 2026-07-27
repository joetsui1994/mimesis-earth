import { geoContains, geoGraticule10, geoOrthographic, geoPath } from 'd3-geo'
import type { GeoProjection } from 'd3-geo'
import type { Feature, FeatureCollection, MultiLineString } from 'geojson'

const INK = '#2f3a45'
const SPHERE = '#bcd9ec' // ocean: soft paper-map blue
const LAND = '#e5decb'
const GRID = '#9fbfd4'
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

// Polygons crossing lon ±180 arrive split (for GIS-valid export), so stroking
// their outlines would draw the straight cut edge along the antimeridian as if
// it were a border. Build stroke geometry that skips those seam segments
// (fills need no such care — the two halves meet seamlessly).
export function boundaryLines(f: Feature): Feature<MultiLineString> {
  const rings: number[][][] = []
  const g = f.geometry
  if (g && g.type === 'Polygon') rings.push(...(g.coordinates as number[][][]))
  if (g && g.type === 'MultiPolygon')
    for (const poly of g.coordinates as number[][][][]) rings.push(...poly)

  const onSeam = (a: number[], b: number[]) =>
    Math.abs(a[0]) === 180 && Math.abs(b[0]) === 180
  const lines: number[][][] = []
  for (const ring of rings) {
    let run: number[][] = []
    for (let i = 0; i < ring.length - 1; i++) {
      if (onSeam(ring[i], ring[i + 1])) {
        if (run.length > 1) lines.push(run)
        run = []
      } else {
        if (run.length === 0) run.push(ring[i])
        run.push(ring[i + 1])
      }
    }
    if (run.length > 1) lines.push(run)
  }
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'MultiLineString', coordinates: lines },
  }
}

export class Globe {
  private ctx: CanvasRenderingContext2D
  private projection: GeoProjection
  private rotation: [number, number] = [20, -15]
  private zoomFactor = 1
  private fc: FeatureCollection | null = null
  private hitFc: FeatureCollection | null = null
  private boundaries: Feature<MultiLineString>[] = []
  private width = 0
  private height = 0
  private dpr = 1
  private rafPending = false
  private selectedIndex = -1
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
    this.boundaries = fc.features.map(boundaryLines)
    this.selected = null
    this.selectedIndex = -1
    this.canvas.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 220 })
    this.draw()
  }

  setSelected(f: Feature | null) {
    this.selected = f
    this.selectedIndex = f && this.fc ? this.fc.features.indexOf(f) : -1
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

    // land units — draw the winding-reversed collection so d3-geo fills the
    // actual landmass rather than its complement (RFC 7946 polygons wind the
    // opposite way d3-geo expects).
    const drawFc = this.hitFc ?? this.fc
    if (drawFc) {
      drawFc.features.forEach((f, i) => {
        ctx.beginPath()
        path(f)
        ctx.fillStyle = i === this.selectedIndex ? HILITE : LAND
        ctx.fill()
      })
      // strokes come from seam-filtered boundary lines so the antimeridian
      // cut is never drawn as a border
      ctx.strokeStyle = INK
      ctx.lineWidth = 0.7
      for (const b of this.boundaries) {
        ctx.beginPath()
        path(b)
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
