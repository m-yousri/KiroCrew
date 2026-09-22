/**
 * The crewmate's reply as it streams into ONE thread.
 *
 * Framework-free, `useSyncExternalStore`-shaped, keyed by `slot + mid`. The
 * stored replies live in React Query (`threadsApi`); this store holds only the
 * text of the reply that is still being written, fed by `chat.thread_reply`
 * frames. The terminal frame (`final`) clears the row: the panel then reads the
 * stored reply from the refetched query, so streamed text is never shown twice.
 *
 * Deltas are grouped by `run_id`: a frame for another run than the one held
 * replaces the row rather than appending, so a late chunk from an earlier
 * reply cannot be glued onto a newer one.
 */

export interface ThreadLive {
  runId: string
  text: string
  /** A plain-language failure from the terminal frame; the row is kept so the
   *  panel can show it until the next reply is sent. */
  error?: string
}

export interface ThreadReplyFrame {
  slot: string
  mid: string
  run_id: string
  role: 'user' | 'assistant' | string
  content: string
  final?: boolean
  is_error?: boolean
}

const EMPTY_LISTENERS: ReadonlySet<() => void> = new Set()

export class ThreadLiveStore {
  private readonly rows = new Map<string, ThreadLive>()
  private readonly listeners = new Map<string, Set<() => void>>()

  static key(slot: string, mid: string): string {
    return slot + '\u0000' + mid
  }

  private notify(key: string): void {
    for (const fn of this.listeners.get(key) ?? EMPTY_LISTENERS) fn()
  }

  /** Apply one wire frame. User frames carry no live text and are ignored here
   *  (the stored reply arrives through the query refetch). */
  apply(frame: ThreadReplyFrame): void {
    if (frame.role !== 'assistant') return
    const key = ThreadLiveStore.key(frame.slot, frame.mid)
    if (frame.final) {
      if (frame.is_error) {
        this.rows.set(key, { runId: frame.run_id, text: '', error: frame.content })
      } else {
        this.rows.delete(key)
      }
      this.notify(key)
      return
    }
    const held = this.rows.get(key)
    const text = held && held.runId === frame.run_id && !held.error ? held.text + frame.content : frame.content
    this.rows.set(key, { runId: frame.run_id, text })
    this.notify(key)
  }

  /** A new reply was just sent: drop a stale error so the panel shows the pending state. */
  clearError(slot: string, mid: string): void {
    const key = ThreadLiveStore.key(slot, mid)
    const held = this.rows.get(key)
    if (held?.error) {
      this.rows.delete(key)
      this.notify(key)
    }
  }

  subscribe(slot: string, mid: string, listener: () => void): () => void {
    const key = ThreadLiveStore.key(slot, mid)
    let set = this.listeners.get(key)
    if (!set) {
      set = new Set()
      this.listeners.set(key, set)
    }
    set.add(listener)
    return () => {
      set!.delete(listener)
      if (set!.size === 0) this.listeners.delete(key)
    }
  }

  get(slot: string, mid: string): ThreadLive | undefined {
    return this.rows.get(ThreadLiveStore.key(slot, mid))
  }
}

export const threadLiveStore = new ThreadLiveStore()
