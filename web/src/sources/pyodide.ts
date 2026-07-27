import type { Spec, WorldData, WorldSource } from './types'

type ProgressCb = (phase: string, pct: number) => void

// Owns the Pyodide worker. Boots on construction; resolves whenReady when the
// worker signals ready. next() round-trips a spec through the worker.
export class PyodideSource implements WorldSource {
  readonly kind = 'pyodide' as const
  private worker: Worker
  private seq = 0
  private pending = new Map<
    number,
    { resolve: (w: WorldData) => void; reject: (e: Error) => void }
  >()
  readonly whenReady: Promise<void>

  constructor(baseUrl: string, onProgress: ProgressCb) {
    this.worker = new Worker(`${baseUrl}pyodide-worker.js`)
    this.whenReady = new Promise<void>((resolve, reject) => {
      this.worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          onProgress(msg.phase, msg.pct)
        } else if (msg.type === 'ready') {
          resolve()
        } else if (msg.type === 'world') {
          this.pending.get(msg.id)?.resolve(JSON.parse(msg.payload) as WorldData)
          this.pending.delete(msg.id)
        } else if (msg.type === 'error') {
          const p = this.pending.get(msg.id)
          if (p) {
            p.reject(new Error(msg.message))
            this.pending.delete(msg.id)
          } else {
            reject(new Error(msg.message)) // error during init
          }
        }
      }
      this.worker.onerror = (e) => {
        const err = new Error(e.message || 'pyodide worker crashed')
        reject(err) // no-op if whenReady already resolved
        for (const p of this.pending.values()) p.reject(err)
        this.pending.clear()
      }
    })
    this.worker.postMessage({ type: 'init', baseUrl })
  }

  next(spec: Spec): Promise<WorldData> {
    const id = ++this.seq
    return new Promise<WorldData>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ type: 'generate', id, spec })
    })
  }
}
