import { describe, expect, it, vi } from 'vitest'
import { ThreadLiveStore } from './threadLiveStore'

const frame = (over: Partial<Parameters<ThreadLiveStore['apply']>[0]> = {}) => ({
  slot: 'member-radar',
  mid: 'm-1',
  run_id: 'run-a',
  role: 'assistant',
  content: '',
  ...over,
})

describe('ThreadLiveStore', () => {
  it('accumulates deltas of one run and clears on the terminal frame', () => {
    const store = new ThreadLiveStore()
    const seen = vi.fn()
    store.subscribe('member-radar', 'm-1', seen)
    store.apply(frame({ content: 'Five ' }))
    store.apply(frame({ content: 'are covered.' }))
    expect(store.get('member-radar', 'm-1')).toEqual({ runId: 'run-a', text: 'Five are covered.' })
    store.apply(frame({ content: 'Five are covered.', final: true }))
    expect(store.get('member-radar', 'm-1')).toBeUndefined()
    expect(seen).toHaveBeenCalledTimes(3)
  })

  it('a frame from another run replaces, never appends', () => {
    const store = new ThreadLiveStore()
    store.apply(frame({ content: 'old ' }))
    store.apply(frame({ run_id: 'run-b', content: 'new' }))
    expect(store.get('member-radar', 'm-1')?.text).toBe('new')
  })

  it('keeps a terminal error until the next reply clears it', () => {
    const store = new ThreadLiveStore()
    store.apply(frame({ content: "The reply didn't go through.", final: true, is_error: true }))
    expect(store.get('member-radar', 'm-1')?.error).toBe("The reply didn't go through.")
    store.clearError('member-radar', 'm-1')
    expect(store.get('member-radar', 'm-1')).toBeUndefined()
  })

  it('ignores user frames and keeps threads apart', () => {
    const store = new ThreadLiveStore()
    store.apply(frame({ role: 'user', content: 'hello' }))
    expect(store.get('member-radar', 'm-1')).toBeUndefined()
    store.apply(frame({ mid: 'm-2', content: 'x' }))
    expect(store.get('member-radar', 'm-1')).toBeUndefined()
    expect(store.get('member-radar', 'm-2')?.text).toBe('x')
  })

  it('unsubscribe stops notifications', () => {
    const store = new ThreadLiveStore()
    const seen = vi.fn()
    const off = store.subscribe('member-radar', 'm-1', seen)
    off()
    store.apply(frame({ content: 'x' }))
    expect(seen).not.toHaveBeenCalled()
  })
})
