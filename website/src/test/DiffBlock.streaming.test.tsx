import { describe, it, vi, beforeEach, expect } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import DiffBlock from '../components/DiffBlock'

beforeEach(() => {
  globalThis.fetch = vi.fn(() => Promise.resolve({ ok: true })) as unknown as typeof fetch
})

const fullPatch = `--- /home/user/example/src/greet.py
+++ /home/user/example/src/greet.py
@@ -1,5 +1,7 @@
 def greet(name):
-    print("Hello " + name)
+    if not name:
+        raise ValueError("name is required")
+    print(f"Hello {name}")
 
 
-greet("world")
+greet("Krish")
`

/** Every streaming prefix of a chat diff block must render without throwing.
 *  Pierre's PatchDiff itself asserts exactly-one-file-diff and THROWS on the
 *  partial frames a streaming fence produces (bare header lines, no hunk yet)
 *  — the wrapper must absorb those states rather than crash-looping the
 *  per-message error boundary.
 *
 *  Reaching that throw requires the lazy chunk to be RESOLVED: a render that
 *  is unmounted in the same tick only ever shows the Suspense fallback, so
 *  Pierre's parser is never entered and the suite proves nothing about the
 *  claim above. `warmPierre()` resolves the chunk once (module registry keeps
 *  it resolved for the rest of the file), and each prefix then flushes so
 *  PatchImpl actually mounts and parses before being torn down. */
async function warmPierre() {
  const { unmount } = render(<DiffBlock code={fullPatch} complete />)
  // The lazy `import('./PierreImpl')` chunk (src/pierre/index.tsx) races the
  // default findBy timeout (1000ms) under a loaded, concurrent run -- widen
  // it rather than assume the dynamic import always resolves within 1s
  // (same class as touchHoverActions.test.tsx).
  await screen.findByTitle('Copy patch', {}, { timeout: 5000 })
  unmount()
}

/** One macrotask: long enough for the resolved lazy child to commit and for
 *  Pierre's synchronous parse to run during that commit. */
const flush = () => act(() => new Promise<void>(r => setTimeout(r, 0)))

describe('DiffBlock streaming', () => {
  it('renders every streamed prefix without throwing', async () => {
    await warmPierre()
    for (let end = 1; end <= fullPatch.length; end += 7) {
      const partial = fullPatch.slice(0, end)
      const { unmount } = render(
        <DiffBlock code={partial} complete={false} />,
      )
      await flush()
      unmount()
    }
  })

  it('renders a multi-file patch without throwing', async () => {
    await warmPierre()
    const multi = fullPatch + '\n' + fullPatch.replace(/greet\.py/g, 'other.py')
    render(<DiffBlock code={multi} complete />)
    await flush()
  })

  it('renders empty and header-only content without throwing', async () => {
    await warmPierre()
    render(<DiffBlock code="" complete={false} />)
    await flush()
    render(<DiffBlock code={'--- /a/b.py\n+++ /a/b.py'} complete={false} />)
    await flush()
  })
  /** The GATE: an unfinished block must not reach Pierre at all. Pierre re-parses
   *  and re-tokenizes the WHOLE patch per frame (its cache key is content-derived,
   *  so every frame misses), which a streamed diff would otherwise pay once per
   *  chunk. `pre.pierre-plain` is the app-owned stand-in, and the header path it
   *  prints is the untouched one -- the Pierre path shortens headers, so the full
   *  path doubles as proof of WHICH path rendered. */
  it('renders a streaming patch as plain text, keeping the original header paths', async () => {
    await warmPierre()
    const { container } = render(<DiffBlock code={fullPatch} complete={false} streaming />)
    await flush()
    // Scoped to this render: the cases above deliberately leave their trees
    // mounted, so a document-wide query can read THEIR stand-in instead.
    const plain = container.querySelector('pre.pierre-plain')
    expect(plain).not.toBeNull()
    // The full path survives: a basename-shortened header would not apply if copied.
    expect(plain?.textContent).toContain('--- /home/user/example/src/greet.py')
  })

  /** jsdom has no `Worker`, so Pierre's real surface never mounts here -- the pool
   *  reports `unsupported` and Pierre's own wrapper falls back to the SAME plain
   *  stand-in. So "the stand-in is gone" cannot tell the two paths apart. The full
   *  header path can: only the Pierre path shortens headers
   *  (`basenamePatchHeaders`), so the untouched path is present while frames arrive
   *  and absent once the block is handed to Pierre. Holds either way, and deleting
   *  the gate flips the first assertion. */
  it('hands the patch to Pierre only once the block is complete', async () => {
    await warmPierre()
    const { container, rerender } = render(<DiffBlock code={fullPatch} complete={false} streaming />)
    await flush()
    expect(container.textContent).toContain('--- /home/user/example/src/greet.py')
    rerender(<DiffBlock code={fullPatch} complete />)
    await flush()
    expect(container.textContent).not.toContain('--- /home/user/example/src/greet.py')
  })
})
