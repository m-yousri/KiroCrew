import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ApiError } from '../../api/apiError'
import { threadLiveStore } from '../../state/threadLiveStore'

const mockDetail = vi.fn()
const mockReply = vi.fn()

vi.mock('../../api/threads', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/threads')>()
  return {
    ...actual,
    threadsApi: {
      summary: vi.fn(),
      detail: (...args: unknown[]) => mockDetail(...args),
      reply: (...args: unknown[]) => mockReply(...args),
    },
  }
})

// The markdown renderer pulls in the whole highlight/mermaid stack; the thread
// panel's contract is the rows and the composer, not markdown rendering.
vi.mock('../../components/MarkdownRenderer', () => ({
  default: ({ content }: { content: string }) => <span>{content}</span>,
}))

import ThreadPanel from './ThreadPanel'

const PARENT = { mid: 'm-1', role: 'assistant', content: 'Overnight triage: 9 new issues.', ts: '2026-09-22T07:02:00Z' }
const REPLIES = [
  { id: 'r1', role: 'user', content: 'What about the other 8?', ts: '2026-09-22T07:41:00Z' },
  { id: 'r2', role: 'assistant', content: '5 are covered by open PRs.', ts: '2026-09-22T07:41:30Z' },
  { id: 'r3', role: 'assistant', content: '3 are queued for Fixer.', ts: '2026-09-22T07:41:40Z' },
]

let qc: QueryClient
const renderPanel = (onClose = vi.fn()) =>
  render(
    <QueryClientProvider client={qc}>
      <ThreadPanel slot="member-radar" mid="m-1" crewmateName="Radar" onClose={onClose} />
    </QueryClientProvider>,
  )

beforeEach(() => {
  vi.clearAllMocks()
  qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  mockDetail.mockResolvedValue({ parent: PARENT, replies: REPLIES, in_flight: false })
  mockReply.mockResolvedValue({ reply: { id: 'r4', role: 'user', content: 'ok', ts: '2026-09-22T07:44:00Z' }, run_id: 'run-1' })
  threadLiveStore.clearError('member-radar', 'm-1')
})

describe('ThreadPanel', () => {
  it('quotes the parent, counts the replies and groups the crewmate run', async () => {
    renderPanel()
    expect(screen.getByRole('complementary', { name: 'Thread' })).toBeInTheDocument()
    await screen.findByTestId('thread-parent')
    expect(screen.getByTestId('thread-parent-bubble')).toHaveTextContent('Overnight triage')
    expect(screen.getByTestId('thread-reply-count')).toHaveTextContent('3 replies')
    const rows = screen.getAllByTestId('thread-reply')
    expect(rows.map((r) => r.getAttribute('data-reply-from'))).toEqual(['user', 'assistant', 'assistant'])
    // Two consecutive crewmate replies within the gap share one run: start, end.
    const bubbles = rows.map((r) => r.querySelector('[data-bubble-pos]')?.getAttribute('data-bubble-pos'))
    expect(bubbles).toEqual(['single', 'start', 'end'])
    // The crewmate's name heads its run once; the user's row says "You".
    expect(screen.getAllByText('Radar').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('You')).toBeInTheDocument()
  })

  it('sends a reply on Enter, clears the draft and disables send while the crewmate replies', async () => {
    renderPanel()
    await screen.findByTestId('thread-parent')
    const box = screen.getByTestId('thread-composer') as HTMLTextAreaElement
    fireEvent.change(box, { target: { value: '  Is #4198 among them?  ' } })
    fireEvent.keyDown(box, { key: 'Enter' })
    await waitFor(() => expect(mockReply).toHaveBeenCalledWith('member-radar', 'm-1', 'Is #4198 among them?'))
    await waitFor(() => expect(box.value).toBe(''))
    // The server now reports the crewmate replying: typing row, send disabled.
    mockDetail.mockResolvedValue({ parent: PARENT, replies: REPLIES, in_flight: true })
    await act(async () => { await qc.invalidateQueries() })
    await screen.findByTestId('thread-replying')
    fireEvent.change(box, { target: { value: 'again' } })
    expect(screen.getByRole('button', { name: 'Send reply' })).toBeDisabled()
  })

  it('streams the crewmate reply from the live store', async () => {
    renderPanel()
    await screen.findByTestId('thread-parent')
    act(() => {
      threadLiveStore.apply({ slot: 'member-radar', mid: 'm-1', run_id: 'run-9', role: 'assistant', content: 'Yes, #4198 ' })
      threadLiveStore.apply({ slot: 'member-radar', mid: 'm-1', run_id: 'run-9', role: 'assistant', content: 'is covered.' })
    })
    expect(screen.getByTestId('thread-reply-live')).toHaveTextContent('Yes, #4198 is covered.')
    act(() => {
      threadLiveStore.apply({ slot: 'member-radar', mid: 'm-1', run_id: 'run-9', role: 'assistant', content: 'Yes, #4198 is covered.', final: true })
    })
    expect(screen.queryByTestId('thread-reply-live')).toBeNull()
  })

  it('names the failure in plain words and keeps the draft', async () => {
    mockReply.mockRejectedValue(new ApiError(409, 'x', JSON.stringify({ error: 'x', code: 'thread_turn_in_flight' })))
    renderPanel()
    await screen.findByTestId('thread-parent')
    const box = screen.getByTestId('thread-composer') as HTMLTextAreaElement
    fireEvent.change(box, { target: { value: 'one more' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send reply' }))
    await screen.findByTestId('thread-send-error')
    expect(screen.getByTestId('thread-send-error')).toHaveTextContent('Radar is still replying. Wait for that reply.')
    expect(box.value).toBe('one more')
  })

  it('close hands the panel back', async () => {
    const onClose = vi.fn()
    renderPanel(onClose)
    fireEvent.click(screen.getByRole('button', { name: 'Close thread' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
