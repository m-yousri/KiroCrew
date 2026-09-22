/**
 * A reply thread on one message of a crewmate's chat, opened in the right side
 * panel (CrewMates launch, screen 07). The main chat stays visible beside it.
 *
 * Top to bottom: a "Thread" header with the crewmate's name and a close
 * button; the parent message quoted as a single bubble (avatar, name, time);
 * a hairline "N replies"; the replies as small bubbles on the grouped-corner
 * rule (the crewmate's on the left under a small avatar, the user's on the
 * right, mirrored); and a one-line "Reply…" composer with the real SendBtn.
 *
 * Stored replies come from `threadsApi.detail` (React Query); the crewmate's
 * reply-in-progress streams through `threadLiveStore`, fed by
 * `chat.thread_reply` frames. While the crewmate is replying the composer is
 * disabled (the backend refuses a second reply in that thread until then), and
 * a typing row stands in as an ordinary item, not a notice.
 */
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowUp, X } from 'lucide-react'
import CrewAvatar from '../../components/CrewAvatar'
import ErrorNotice from '../../components/ErrorNotice'
import MarkdownRenderer from '../../components/MarkdownRenderer'
import MessageErrorBoundary from '../../components/MessageErrorBoundary'
import { SendBtn } from '../../components/ui'
import { useImeGuard } from '../../hooks/useImeGuard'
import { ApiError } from '../../api/apiError'
import { threadQueryKey, threadsApi, threadsQueryKey, type ThreadReply } from '../../api/threads'
import { fmtMessageTime } from '../chat/messageTime'
import { threadLiveStore } from '../../state/threadLiveStore'
import { parseErrorCode } from '../../utils/errorReport'
import { bubbleRadiusStyle, runPositions, type BubblePos } from './threadBubbles'

const AVATAR_PX = 22
const MAX_REPLY_CHARS = 32_000

/** One small bubble on the corner rule. `side` is where its run sits. */
function Bubble({ pos, side, children, testId }: { pos: BubblePos; side: 'left' | 'right'; children: React.ReactNode; testId?: string }) {
  return (
    <div
      data-testid={testId}
      data-bubble-pos={pos}
      className={`border border-border bg-card text-card-fg px-3 py-1.5 text-[13px] leading-[1.45] ${side === 'right' ? 'max-w-[85%]' : 'max-w-[92%]'}`}
      style={{ overflowWrap: 'anywhere', ...bubbleRadiusStyle(pos, side) }}
    >
      {children}
    </div>
  )
}

/** Author line above the first bubble of a run: name + time. */
function AuthorLine({ name, ts, align }: { name: string; ts: string; align: 'left' | 'right' }) {
  return (
    <span className={`text-[11px] leading-4 text-muted tabular-nums mb-1 ${align === 'left' ? 'ml-1' : 'mr-1'}`}>
      <span className="font-semibold text-text">{name}</span>
      {ts && <> · {fmtMessageTime(ts)}</>}
    </span>
  )
}

function ReplyRow({ reply, pos, crewmateName, youLabel }: { reply: ThreadReply; pos: BubblePos; crewmateName: string; youLabel: string }) {
  const opens = pos === 'start' || pos === 'single'
  if (reply.role === 'user') {
    return (
      <li className={`flex flex-col items-end ${opens ? 'mt-3' : 'mt-1'}`} data-testid="thread-reply" data-reply-from="user">
        {opens && <AuthorLine name={youLabel} ts={reply.ts} align="right" />}
        <Bubble pos={pos} side="right">{reply.content}</Bubble>
      </li>
    )
  }
  return (
    <li className={`flex gap-2 ${opens ? 'mt-3' : 'mt-1'}`} data-testid="thread-reply" data-reply-from="assistant">
      <div className="shrink-0" style={{ width: AVATAR_PX }}>{opens && <CrewAvatar seed={crewmateName} size={AVATAR_PX} />}</div>
      <div className="min-w-0 flex-1 flex flex-col items-start">
        {opens && <AuthorLine name={crewmateName} ts={reply.ts} align="left" />}
        <Bubble pos={pos} side="left">
          <MessageErrorBoundary rawContent={reply.content}><MarkdownRenderer content={reply.content} softBreaks /></MessageErrorBoundary>
        </Bubble>
      </div>
    </li>
  )
}

export default function ThreadPanel({ slot, mid, crewmateName, onClose }: {
  slot: string
  mid: string
  crewmateName: string
  onClose: () => void
}) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const detail = useQuery({
    queryKey: threadQueryKey(slot, mid),
    queryFn: () => threadsApi.detail(slot, mid),
  })
  const live = useSyncExternalStore(
    useCallback((cb: () => void) => threadLiveStore.subscribe(slot, mid, cb), [slot, mid]),
    () => threadLiveStore.get(slot, mid),
  )
  const [draft, setDraft] = useState('')
  const ime = useImeGuard()
  const scrollerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const send = useMutation({
    mutationFn: (text: string) => threadsApi.reply(slot, mid, text),
    onMutate: () => threadLiveStore.clearError(slot, mid),
    onSuccess: () => {
      setDraft('')
      void qc.invalidateQueries({ queryKey: threadQueryKey(slot, mid) })
      void qc.invalidateQueries({ queryKey: threadsQueryKey(slot) })
    },
  })

  const storedReplies = detail.data?.replies
  const replies = useMemo(() => storedReplies ?? [], [storedReplies])
  const positions = useMemo(
    () => runPositions(replies.map((r) => ({ author: r.role, ts: r.ts }))),
    [replies],
  )
  // The crewmate is writing: the server says so, or a streamed delta is in hand.
  const replying = !!detail.data?.in_flight || (!!live && !live.error)
  const youLabel = t('pages.chat.thread.you')

  // Follow the tail as replies and streamed text land.
  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [replies.length, live?.text])
  useEffect(() => { textareaRef.current?.focus() }, [mid])

  const submit = () => {
    const text = draft.trim()
    if (!text || send.isPending || replying) return
    send.mutate(text)
  }

  // One plain sentence per failure; the backend's code picks the sentence and a
  // failed send keeps the draft, so nothing typed is lost.
  const sendError = (() => {
    const err = send.error
    if (!err) return ''
    const code = err instanceof ApiError ? parseErrorCode(err.body) : undefined
    if (code === 'thread_turn_in_flight') return t('pages.chat.thread.err_replying', { name: crewmateName })
    if (code === 'parent_not_found') return t('pages.chat.thread.err_parent_gone')
    if (code === 'reply_too_long') return t('pages.chat.thread.err_too_long')
    if (code === 'thread_full') return t('pages.chat.thread.err_full')
    return t('pages.chat.thread.err_send_failed')
  })()
  const shownError = sendError || live?.error || ''

  const parent = detail.data?.parent
  const parentIsUser = parent?.role === 'user'

  return (
    <div
      data-testid="thread-panel"
      className="absolute inset-0 z-20 flex flex-col bg-bg"
      role="complementary"
      aria-label={t('pages.chat.thread.title')}
    >
      <div className="shrink-0 flex items-center gap-2 px-3 min-h-10 rounded-tl-xl bg-bg-elevated border-b border-border">
        <h2 className="text-[13px] font-semibold m-0 leading-none">{t('pages.chat.thread.title')}</h2>
        <span className="text-[12px] text-muted truncate">{crewmateName}</span>
        <button
          type="button"
          onClick={onClose}
          className="ml-auto inline-flex items-center justify-center w-7 h-7 rounded-md text-muted hover:text-text hover:bg-bg-hover cursor-pointer"
          aria-label={t('pages.chat.thread.close')}
          title={t('pages.chat.thread.close')}
        >
          <X className="lucide-inline" style={{ width: 15, height: 15 }} />
        </button>
      </div>
      <div ref={scrollerRef} className="flex-1 min-h-0 overflow-y-auto px-3 pt-3">
        {detail.isError && (
          /* No hand-off: the reply draft in the composer below is unsaved local
             state; a navigation would discard it. The read retries on the next
             chat.thread_reply frame or reopen. */
          <ErrorNotice
            variant="inline"
            message={t('pages.chat.thread.err_load_failed')}
            testId="thread-load-error"
          />
        )}
        {parent && (
          parentIsUser ? (
            <div className="flex flex-col items-end" data-testid="thread-parent">
              <AuthorLine name={youLabel} ts={parent.ts} align="right" />
              <Bubble pos="single" side="right" testId="thread-parent-bubble">{parent.content}</Bubble>
            </div>
          ) : (
            <div className="flex gap-2" data-testid="thread-parent">
              <div className="shrink-0" style={{ width: AVATAR_PX }}><CrewAvatar seed={crewmateName} size={AVATAR_PX} /></div>
              <div className="min-w-0 flex-1 flex flex-col items-start">
                <AuthorLine name={crewmateName} ts={parent.ts} align="left" />
                <Bubble pos="single" side="left" testId="thread-parent-bubble">
                  <MessageErrorBoundary rawContent={parent.content}><MarkdownRenderer content={parent.content} softBreaks /></MessageErrorBoundary>
                </Bubble>
              </div>
            </div>
          )
        )}
        {parent && (
          <div className="flex items-center gap-2 my-3 text-[11px] text-muted" data-testid="thread-reply-count">
            <span className="shrink-0">
              {replies.length > 0
                ? t('pages.chat.thread.replies_count', { count: replies.length })
                : t('pages.chat.thread.no_replies_yet')}
            </span>
            <span className="flex-1 h-px bg-border" aria-hidden="true" />
          </div>
        )}
        <ul className="list-none m-0 p-0" data-testid="thread-replies">
          {replies.map((r, i) => (
            <ReplyRow key={r.id} reply={r} pos={positions[i] ?? 'single'} crewmateName={crewmateName} youLabel={youLabel} />
          ))}
          {live && !live.error && (
            // The reply as it streams in: one crewmate bubble that grows, then
            // gives way to the stored row the terminal frame refetches.
            <li className="flex gap-2 mt-3" data-testid="thread-reply-live" aria-live="polite">
              <div className="shrink-0" style={{ width: AVATAR_PX }}><CrewAvatar seed={crewmateName} size={AVATAR_PX} working="subtle" /></div>
              <div className="min-w-0 flex-1 flex flex-col items-start">
                <AuthorLine name={crewmateName} ts="" align="left" />
                <Bubble pos="single" side="left">
                  <MessageErrorBoundary rawContent={live.text}><MarkdownRenderer content={live.text} streaming softBreaks /></MessageErrorBoundary>
                </Bubble>
              </div>
            </li>
          )}
          {replying && !live?.text && (
            <li className="flex items-center gap-1.5 mt-3 px-1 text-muted" data-testid="thread-replying" role="status">
              <span className="flex gap-0.5" aria-hidden="true">
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" style={{ animationDelay: '300ms' }} />
              </span>
              <span className="text-[12px]">{t('pages.chat.thread.replying', { name: crewmateName })}</span>
            </li>
          )}
        </ul>
      </div>
      {shownError && (
        <div className="px-3 pt-1">
          {/* No hand-off: the draft below is unsaved; a failed send keeps it. */}
          <ErrorNotice variant="inline" message={shownError} testId="thread-send-error" />
        </div>
      )}
      <div className="shrink-0 px-3 pb-3 pt-2">
        <div className="flex items-center gap-2 rounded-xl border border-border focus-within:border-accent bg-card px-3 py-2 transition-colors">
          <textarea
            ref={textareaRef}
            rows={1}
            value={draft}
            maxLength={MAX_REPLY_CHARS}
            onChange={(e) => setDraft(e.target.value)}
            {...ime.bindComposition()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { if (ime.claimEnter(e)) submit() }
            }}
            className={/* focus-cue-ok: the cue is the composer card's focus-within border-accent, the same cue the main composer shell paints; a second ring on the textarea would double-paint one control. */ 'flex-1 min-w-0 resize-none bg-transparent text-[13px] leading-6 text-card-fg outline-hidden placeholder:text-muted'}
            placeholder={t('pages.chat.thread.reply_placeholder')}
            aria-label={t('pages.chat.thread.reply_in_thread')}
            data-testid="thread-composer"
          />
          <SendBtn
            className="px-0 min-h-0 w-8 h-8 rounded-full inline-flex items-center justify-center shrink-0"
            aria-label={t('pages.chat.thread.send_reply')}
            disabled={!draft.trim() || send.isPending || replying}
            onClick={submit}
          >
            <ArrowUp className="lucide-inline" style={{ width: 18, height: 18 }} />
          </SendBtn>
        </div>
      </div>
    </div>
  )
}
