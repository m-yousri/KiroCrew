/**
 * CrewBoardPage — the conductor's work items, as a board a human can scan.
 *
 * Implements Phase 4 ("the surfaces", Crew page half) of
 * `docs/request-for-change/rfc-conductor-work-ledger.md`. It reads the masked
 * projection at `GET /api/crew-board`, never the conductor's own MCP route: no
 * row here carries `worker_session_key`, and the text of a `bind` event arrives
 * blanked, so there is nothing on this page that could address a worker session.
 *
 * ## Why the bands, and why this order
 *
 * A conductor's board is read for ONE reason: to find the item that cannot move
 * without a human. So items waiting on a ruling are lifted out of document order
 * into a band at the top — if finding a blocked worker needs a scroll, the board
 * has failed at the only job it has. Everything still open follows. Terminal items
 * collapse behind an expander, because a finished item is evidence rather than
 * work and a board that grows forever stops being scannable.
 *
 * ## Zero model turns
 *
 * The browser polls every 10 s. No agent wakes to render this, no turn is spent,
 * and nothing is pushed: the RFC's push half is PR A (the wake hook), which lands
 * separately. `channels_available` arrives `false` until the RFC's Phase 5 ships
 * the channel records, so that band turns on server-side with no edit here.
 *
 * ## Chips render the store's own tokens
 *
 * `status`, `state`, `verdict` and `alive` are shown verbatim as the store spells
 * them. They are a technical vocabulary shared by the MCP tools, the event log and
 * this page, and a translated synonym would mean the page and the tool a conductor
 * just ran disagree about what an item is. Only prose is localized.
 */

import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, ChevronDown, ChevronRight, Inbox } from 'lucide-react'

import { Card, EmptyState } from '../components/ui'
import { api } from '../api/client'
import { i18nT } from '../i18n/t'
import { fmtRelative } from '../i18n/format'
import { ApiError } from '../api/apiError'
import {
  CREW_BOARD_POLL_MS,
  crewBoardQueryKey,
  type CrewBoardAction,
  type WorkBoardItem,
  type WorkBoardResponse,
} from '../api/crewBoard'
import { artifactEntries, partitionBoardRows, rowKindLabelKey } from './crewBoardRows'

/** Theme tokens only. A literal hex or a palette class fails `no-raw-colors`,
 *  and more to the point a fixed palette would ignore the user's theme. */
const C = {
  text: 'var(--text)',
  dim: 'var(--text-dim)',
  border: 'var(--border)',
  cardHl: 'var(--card-hl)',
  accent: 'var(--accent)',
  warn: 'var(--warn)',
} as const

/** The alive dot. Filled while running, hollow once idle, faint when closed —
 *  three states a scan can tell apart without reading the word beside it. */
function AliveDot({ alive }: { alive: WorkBoardItem['alive'] }) {
  const style =
    alive === 'running'
      ? { background: C.accent, borderColor: C.accent }
      : alive === 'idle'
        ? { background: 'transparent', borderColor: C.text }
        : { background: 'transparent', borderColor: C.dim }
  return (
    <span
      aria-hidden
      className="mt-[6px] size-[7px] shrink-0 rounded-full border"
      style={style}
    />
  )
}

/** One small token chip. No background and no box — a hairline and the word.
 *  Raymond rejects visible chrome on dense rows; the weight belongs on the text. */
function Chip({ children, tone }: { children: React.ReactNode; tone?: 'warn' }) {
  return (
    <span
      className="shrink-0 rounded-sm border px-1 py-px text-[11px] leading-[14px] tabular-nums"
      style={{ borderColor: tone === 'warn' ? C.warn : C.border, color: tone === 'warn' ? C.warn : C.dim }}
    >
      {children}
    </span>
  )
}

/** A band heading: the label, then the count, then a hairline across the rest.
 *  The rule carries the eye without drawing a container around the rows. */
function BandHeading({ label, count }: { label: string; count: number }) {
  return (
    <div className="flex items-center gap-2 pb-1 pt-3">
      <span className="text-[11px] uppercase tracking-wide" style={{ color: C.dim }}>
        {label}
      </span>
      <span className="text-[11px] tabular-nums" style={{ color: C.dim }}>
        {count}
      </span>
      <span className="h-px flex-1" style={{ background: C.border }} />
    </div>
  )
}

/** What this gateway can actually do to an orphaned item, straight from the read. */
interface BoardCaps {
  stopAvailable: boolean
  takeOverAvailable: boolean
  takeOverUnavailableCode: string
}

/**
 * The take-over and stop affordances Phase 4 names, on an orphaned row only.
 *
 * Both are keyed by `item_id`, never by a session: the masked read gives the page
 * no worker key, so the server resolves one from the store and never returns it.
 * That is the whole reason an action route exists rather than the page reusing the
 * ordinary Stop button.
 *
 * Take-over renders DISABLED on main. Nothing on the gateway performs one —
 * `session_control` has no re-own verb and `CONDUCTOR_ACTIONS` has no transfer —
 * so the button states why instead of being wired to something invented here. The
 * reason comes from the server as a CODE which this maps to a translated string,
 * so the page and the route cannot drift into disagreeing about what is possible.
 */
function RowActions({
  item,
  conductor,
  caps,
}: {
  item: WorkBoardItem
  conductor: string
  caps: BoardCaps
}) {
  const queryClient = useQueryClient()
  const [failure, setFailure] = useState('')

  const act = useMutation({
    mutationFn: (action: CrewBoardAction) => api.crewBoardAction(conductor, item.item_id, action),
    onSuccess: () => {
      setFailure('')
      void queryClient.invalidateQueries({ queryKey: crewBoardQueryKey(conductor) })
    },
    onError: (err) => {
      // 409 is the one failure worth wording differently: it means the board this
      // click was made from is stale, not that the action is broken. Re-reading is
      // the remedy, so it fires one.
      const stale = err instanceof ApiError && err.status === 409
      setFailure(
        i18nT(stale ? 'pages.crewBoard.action_stale_view' : 'pages.crewBoard.action_failed'),
      )
      if (stale) void queryClient.invalidateQueries({ queryKey: crewBoardQueryKey(conductor) })
    },
  })

  const workerGone = item.alive === 'closed'
  const stopDisabled = !caps.stopAvailable || workerGone || act.isPending
  const stopReason = workerGone ? i18nT('pages.crewBoard.stop_unavailable_closed') : ''
  const takeOverReason = caps.takeOverAvailable
    ? ''
    : i18nT('pages.crewBoard.take_over_unavailable')

  // Shown as TEXT, not only as a `title`. A tooltip on a disabled button is not
  // reachable by keyboard and is unreliably surfaced by browsers, so a reason that
  // lives only there is a reason nobody reads.
  //
  // Only the STOP reason is inline, because only it is a property of this row --
  // whether this item's own worker session is still open. Take-over's
  // unavailability belongs to the gateway, so repeating it under every row would
  // be the same sentence four times on a board whose whole job is scanning; it is
  // stated once at board level instead.
  const reasons = [stopReason].filter(Boolean)

  return (
    <div className="ml-[15px] mt-1 flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={stopDisabled}
          title={stopReason || undefined}
          onClick={() => act.mutate('stop')}
          className="rounded border px-1.5 py-0.5 text-[11px] disabled:opacity-40"
          style={{ borderColor: C.border, color: stopDisabled ? C.dim : C.text }}
        >
          {i18nT('pages.crewBoard.action_stop')}
        </button>

        <button
          type="button"
          disabled={!caps.takeOverAvailable}
          title={takeOverReason || undefined}
          onClick={() => act.mutate('take_over')}
          className="rounded border px-1.5 py-0.5 text-[11px] disabled:opacity-40"
          style={{ borderColor: C.border, color: C.dim }}
        >
          {i18nT('pages.crewBoard.action_take_over')}
        </button>

        {failure ? (
          <span className="text-[11px]" style={{ color: C.warn }}>
            {failure}
          </span>
        ) : null}
      </div>

      {reasons.length > 0 ? (
        <span className="text-[11px]" style={{ color: C.dim }}>
          {reasons.join(' ')}
        </span>
      ) : null}
    </div>
  )
}

/** One item row: two lines. Identity and state above, the worker's own account
 *  below. The kind label is right-aligned so the right edge reads as a column. */
function ItemRow({
  item,
  conductor,
  caps,
}: {
  item: WorkBoardItem
  conductor: string
  caps: BoardCaps
}) {
  const [open, setOpen] = useState(false)
  const kindKey = rowKindLabelKey(item)
  const artifacts = artifactEntries(item)

  return (
    <div className="border-b py-2" style={{ borderColor: C.border }}>
      <div className="flex items-start gap-2">
        <AliveDot alive={item.alive} />

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="text-[12px] font-medium" style={{ color: C.text }}>
              {item.title || item.item_id}
            </span>

            {/* status (worker) and state (conductor) stay TWO chips. They are
                different parties' claims about the same item and can legitimately
                disagree — a worker reporting `done` on an item the conductor has
                not accepted is the normal case, and one merged chip would have to
                pick a side and would hide exactly that. */}
            {item.status ? <Chip>{item.status}</Chip> : <Chip>{i18nT('pages.crewBoard.no_report')}</Chip>}
            <Chip>{item.state}</Chip>
            {item.verdict ? <Chip>{item.verdict}</Chip> : null}
            {item.stale ? <Chip tone="warn">{i18nT('pages.crewBoard.kind_stale')}</Chip> : null}
            {!item.acceptance_concrete && !item.terminal ? (
              <Chip tone="warn">{i18nT('pages.crewBoard.bar_vague')}</Chip>
            ) : null}
          </div>
        </div>

        <span className="shrink-0 text-[11px] tabular-nums" style={{ color: C.dim }}>
          {fmtRelative(item.last_report_at ?? item.created_at)}
        </span>

        <span className="w-[5.5rem] shrink-0 text-right text-[11px]" style={{ color: C.dim }}>
          {kindKey ? i18nT(kindKey) : item.state}
        </span>
      </div>

      {/* The worker's own account, in place of the session ledger's `next` —
          this store has no equivalent field, so `summary` is the closest true
          thing and is labelled as the worker's words, not as a plan. */}
      {item.summary ? (
        <div className="ml-[15px] mt-1 text-[12px]" style={{ color: C.dim }}>
          {item.summary}
        </div>
      ) : null}

      {item.decision ? (
        <div className="ml-[15px] mt-1 text-[12px]" style={{ color: C.text }}>
          <span className="mr-1 text-[11px]" style={{ color: C.dim }}>
            {i18nT('pages.crewBoard.decision')}
          </span>
          {item.decision}
        </div>
      ) : null}

      {item.orphaned ? (
        <div className="ml-[15px] mt-1 text-[12px]" style={{ color: C.warn }}>
          <AlertTriangle size={12} className="mr-1 inline align-[-2px]" />
          {i18nT('pages.crewBoard.orphaned_note')}
        </div>
      ) : null}

      {/* Only an orphaned item gets them, which is also the only state the action
          route accepts — so the page cannot offer a click the server will refuse. */}
      {item.orphaned ? <RowActions item={item} conductor={conductor} caps={caps} /> : null}

      <div className="ml-[15px] mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
        {item.pr !== null ? (
          <span className="text-[11px] tabular-nums" style={{ color: C.dim }}>
            {/* Reuses the pull-request panel's own key rather than adding a
                twelfth-locale translation for a string the product already has.
                One spelling of "PR #12" across the dashboard is also the point. */}
            {i18nT('components.pullRequestPanel.pr_number', { number: item.pr })}
          </span>
        ) : null}
        {artifacts.map(([key, value]) => (
          <span key={key} className="text-[11px]" style={{ color: C.dim }}>
            <span className="mr-1 opacity-70">{key}</span>
            <span style={{ color: C.text }}>{value}</span>
          </span>
        ))}
        {item.fails > 0 ? (
          <span className="text-[11px] tabular-nums" style={{ color: C.warn }}>
            {i18nT('pages.crewBoard.fails')} {item.fails}
          </span>
        ) : null}

        {item.events.length > 0 ? (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-1 text-[11px]"
            style={{ color: C.dim }}
            aria-expanded={open}
          >
            {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
            {i18nT('pages.crewBoard.events')}
            <span className="tabular-nums">{item.events.length}</span>
          </button>
        ) : null}
      </div>

      {open ? (
        <div className="ml-[15px] mt-1.5 flex flex-col gap-0.5">
          {item.events.map((event) => (
            <div key={event.id} className="flex items-baseline gap-2 text-[11px]">
              <span className="w-[4.5rem] shrink-0 tabular-nums" style={{ color: C.dim }}>
                {fmtRelative(event.ts)}
              </span>
              <span className="w-[4rem] shrink-0" style={{ color: C.dim }}>
                {event.kind}
              </span>
              <span className="min-w-0 flex-1" style={{ color: C.text }}>
                {event.text}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

/** The board for ONE conductor. Exported so the Crew page can host it without
 *  going through the route, which is how it will be embedded there. */
export function CrewBoard({ conductor }: { conductor: string }) {
  const [showFinished, setShowFinished] = useState(false)

  const board = useQuery<WorkBoardResponse>({
    queryKey: crewBoardQueryKey(conductor),
    queryFn: () => api.crewBoard(conductor),
    refetchInterval: CREW_BOARD_POLL_MS,
    enabled: Boolean(conductor),
  })

  const bands = useMemo(
    () => partitionBoardRows(board.data?.items ?? []),
    [board.data?.items],
  )

  // Straight from the server's own answer, never inferred here: whether a stop or
  // a take-over can be performed is a property of the gateway, and a page that
  // guessed would offer a button the route refuses. Defaults to "cannot" so a
  // board still loading never renders an enabled action.
  const caps: BoardCaps = useMemo(
    () => ({
      stopAvailable: board.data?.stop_available ?? false,
      takeOverAvailable: board.data?.take_over_available ?? false,
      takeOverUnavailableCode: board.data?.take_over_unavailable_code ?? '',
    }),
    [
      board.data?.stop_available,
      board.data?.take_over_available,
      board.data?.take_over_unavailable_code,
    ],
  )

  const anyOrphaned = useMemo(
    () => (board.data?.items ?? []).some((item) => item.orphaned),
    [board.data?.items],
  )

  if (!conductor) {
    return <EmptyState icon={<Inbox size={20} />} title={i18nT('pages.crewBoard.missing_conductor')} />
  }

  // A session that owns no work ledger is an expected GAP, not a failure: only a
  // session dispatched through the conductor tooling opens one, so an ad-hoc
  // conductor legitimately has none. Rendering it as an error would teach people
  // the board is broken.
  if (board.isError) {
    const err = board.error
    const noLedger = err instanceof ApiError && err.status === 404
    return (
      <EmptyState
        icon={noLedger ? <Inbox size={20} /> : <AlertTriangle size={20} />}
        title={i18nT(noLedger ? 'pages.crewBoard.no_ledger_title' : 'pages.crewBoard.error_title')}
        subtitle={noLedger ? i18nT('pages.crewBoard.no_ledger_subtitle') : undefined}
      />
    )
  }

  if (!board.data) return null

  const { conductor: record, items } = board.data
  if (items.length === 0) {
    return <EmptyState icon={<Inbox size={20} />} title={i18nT('pages.crewBoard.empty_title')} />
  }

  return (
    <div className="flex flex-col">
      {record.goal ? (
        <div className="flex items-baseline gap-2 pb-1">
          <span className="text-[11px] uppercase tracking-wide" style={{ color: C.dim }}>
            {i18nT('pages.crewBoard.goal')}
          </span>
          <span className="min-w-0 flex-1 text-[12px]" style={{ color: C.text }}>
            {record.goal}
          </span>
          <span className="shrink-0 text-[11px] tabular-nums" style={{ color: C.dim }}>
            {i18nT('pages.crewBoard.round')} {record.round}
          </span>
        </div>
      ) : null}

      {/* Stated ONCE, because take-over's unavailability is a property of the
          gateway rather than of any row. Rendered only when a row could actually
          offer the button, so a board with nothing orphaned says nothing. */}
      {!caps.takeOverAvailable && anyOrphaned ? (
        <div className="mt-1 text-[11px]" style={{ color: C.dim }}>
          {i18nT('pages.crewBoard.take_over_unavailable')}
        </div>
      ) : null}

      {bands.ruling.length > 0 ? (
        <>
          <BandHeading label={i18nT('pages.crewBoard.band_ruling')} count={bands.ruling.length} />
          {bands.ruling.map((item) => (
            <ItemRow key={item.item_id} item={item} conductor={conductor} caps={caps} />
          ))}
        </>
      ) : null}
      {bands.working.length > 0 ? (
        <>
          <BandHeading label={i18nT('pages.crewBoard.band_working')} count={bands.working.length} />
          {bands.working.map((item) => (
            <ItemRow key={item.item_id} item={item} conductor={conductor} caps={caps} />
          ))}
        </>
      ) : null}

      {bands.finished.length > 0 ? (
        <>
          <button
            type="button"
            onClick={() => setShowFinished((v) => !v)}
            className="mt-3 flex items-center gap-1.5 text-[11px] uppercase tracking-wide"
            style={{ color: C.dim }}
            aria-expanded={showFinished}
          >
            {showFinished ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {i18nT('pages.crewBoard.band_finished')}
            <span className="tabular-nums">{bands.finished.length}</span>
          </button>
          {showFinished
            ? bands.finished.map((item) => (
                <ItemRow key={item.item_id} item={item} conductor={conductor} caps={caps} />
              ))
            : null}
        </>
      ) : null}
    </div>
  )
}

/** Route wrapper: the conductor comes from `?conductor=`, so a board is a URL
 *  someone can bookmark or paste, and two conductors can be open side by side. */
export default function CrewBoardPage() {
  const [params] = useSearchParams()
  const conductor = (params.get('conductor') ?? '').trim()

  return (
    <div className="mx-auto w-full max-w-5xl p-4">
      <div className="flex items-baseline gap-2 pb-2">
        <h1 className="text-[13px] font-semibold" style={{ color: C.text }}>
          {i18nT('pages.crewBoard.title')}
        </h1>
        {conductor ? (
          <Link
            to={`/chat?sid=${encodeURIComponent(conductor)}`}
            className="text-[11px] underline-offset-2 hover:underline"
            style={{ color: C.dim }}
          >
            {i18nT('pages.crewBoard.open_conductor')}
          </Link>
        ) : null}
      </div>
      <Card className="p-3">
        <CrewBoard conductor={conductor} />
      </Card>
    </div>
  )
}
