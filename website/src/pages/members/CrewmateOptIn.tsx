/**
 * Crewmate opt-in — the one-time step for an EXISTING user who already has
 * custom agents (`~/.kiro/agents/*.json`) but no crewmate yet.
 *
 * Shown over the Crewmates page on the first visit after the feature ships,
 * in the shipped split-screen first-run chrome (OnboardingChapterShell,
 * standalone mode — this page sits outside App's persistent shell host). The
 * step lists the user's custom agents with how many chats each was used in,
 * pre-checks the used ones, and "Add N crewmates" creates one crewmate per
 * checked agent through the existing crew-create route (name = agent id,
 * Built from = that agent, private memory allocated by the server). It then
 * lands on the most recently used new crewmate's chat; the roster itself is the
 * confirmation, so there is no success banner (launch review, 2026-09-22).
 *
 * Both exits — "Not now" and a finished add — record the step as over in
 * gateway config (`dashboard.crewmate_optin_done`), so no browser sees it
 * twice. Whether to show it is decided from ONE server read
 * (`GET /api/members/optin`: not done, zero crewmates, at least one candidate)
 * and latched open for the rest of the visit, so a partial add (some created,
 * then a failure) does not yank the step away mid-recovery.
 */
import { useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type CrewmateOptinCandidate, type CrewmateOptinState } from '../../api/client'
import OnboardingChapterShell, { OnboardingShellContext } from '../../components/OnboardingChapterShell'
import CrewAvatar from '../../components/CrewAvatar'
import ErrorNotice from '../../components/ErrorNotice'
import { Btn, Checkbox, SendBtn } from '../../components/ui'
import { useDialogFocusTrap } from '../../hooks/useDialogFocusTrap'
import { useOptionalTheme } from '../../hooks/useTheme'

/** Query key for the step's state. Under `kirocrew-agents` like the roster:
 *  a crew written anywhere flips `crewmates` here through the same
 *  invalidation, so the step can never open over a roster that has one. */
export const MEMBERS_OPTIN_QUERY_KEY = ['kirocrew-agents', 'members-optin'] as const

/** What a failed add carries back: which name refused, and which names had
 *  already landed before it (crewmates now, whatever happens next). */
type AddFailure = Error & { failedName?: string; created?: CrewmateOptinCandidate[] }

/** The page-level gate. Renders nothing until the server says the step is
 *  due; then keeps it open until the user leaves it. */
export default function CrewmateOptIn() {
  // Never over a first-run chapter: those overlay every page at the same
  // layer, and this step is for users who are already through them. Optional
  // read: a host with no ThemeProvider (an isolated render) has no first-run
  // state to consult, and the step simply stays off there.
  const theme = useOptionalTheme()
  const firstRunDone = !!theme && theme.onboarded && theme.importOnboarded && theme.privacyAcked
  const stateQuery = useQuery({
    queryKey: MEMBERS_OPTIN_QUERY_KEY,
    queryFn: () => api.membersOptin(),
    enabled: firstRunDone,
    staleTime: Infinity,
    retry: false,
  })
  const due =
    !!stateQuery.data
    && !stateQuery.data.done
    && stateQuery.data.crewmates === 0
    && stateQuery.data.candidates.length > 0
  // Latched: once open, stays open until an exit — the candidate list and the
  // crewmate count both move while an add is in flight.
  const [open, setOpen] = useState(false)
  const [left, setLeft] = useState(false)
  useEffect(() => {
    if (due && !left) setOpen(true)
  }, [due, left])
  if (!open || !stateQuery.data) return null
  return (
    <CrewmateOptInStep
      state={stateQuery.data}
      onLeave={() => {
        setLeft(true)
        setOpen(false)
      }}
    />
  )
}

function CrewmateOptInStep({ state, onLeave }: { state: CrewmateOptinState; onLeave: () => void }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  // Focus trap on the dialog element. Standalone here (no shell host on this
  // page), but read the host's ref when one exists so the component stays
  // correct if the page is ever wrapped in one.
  const shellHost = useContext(OnboardingShellContext)
  const localDialogRef = useRef<HTMLDivElement>(null)
  const dialogRef = shellHost?.dialogRef ?? localDialogRef
  // The candidate list is frozen at open: a row must not vanish from under the
  // user's cursor while a partial add is being recovered.
  const [candidates] = useState<CrewmateOptinCandidate[]>(state.candidates)
  // Pre-checked: the agents that were actually used. A never-used agent is
  // offered, not assumed.
  const [picked, setPicked] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(candidates.map(c => [c.name, c.chats > 0])),
  )
  // Names that became crewmates in a partial add. Shown checked and locked:
  // they are done, and a retry adds only what is still missing.
  const [landed, setLanded] = useState<ReadonlySet<string>>(() => new Set())
  const chosen = useMemo(
    () => candidates.filter(c => picked[c.name] && !landed.has(c.name)),
    [candidates, picked, landed],
  )
  const count = chosen.length
  const [failedName, setFailedName] = useState<string | null>(null)
  const [closeFailed, setCloseFailed] = useState(false)

  // Both exits record the step as over. For "Not now" the record IS the exit,
  // so its failure keeps the step open with the reason; after an add the
  // crewmates exist regardless, and the gate hides on `crewmates > 0`, so a
  // failed record there is logged by the server and not shown.
  const finish = useMutation({
    mutationFn: () => api.membersOptinDone(),
    onSuccess: () => {
      queryClient.setQueryData<CrewmateOptinState>(MEMBERS_OPTIN_QUERY_KEY, prev =>
        prev ? { ...prev, done: true } : prev,
      )
      onLeave()
    },
    onError: () => setCloseFailed(true),
  })

  // One crewmate per checked agent, in order, through the ONE create route.
  // Sequential on purpose: the server serializes crew creation under its
  // config lock, and a burst would only queue there while hiding which name
  // failed. `created` carries the names that landed so the landing page can
  // open the most recently used one even after a mid-list failure.
  const add = useMutation({
    mutationFn: async (rows: CrewmateOptinCandidate[]) => {
      const created: CrewmateOptinCandidate[] = []
      for (const row of rows) {
        try {
          await api.createKirocrewAgent({
            name: row.name,
            kiro_agent: row.name,
            description: row.description,
            source: 'kirocrew',
          })
        } catch (err) {
          const failure: AddFailure = err instanceof Error ? err : new Error(String(err))
          failure.failedName = row.name
          failure.created = created
          throw failure
        }
        created.push(row)
      }
      return created
    },
    onSuccess: async created => {
      await queryClient.invalidateQueries({ queryKey: ['kirocrew-agents'] })
      api.membersOptinDone().catch(() => undefined)
      onLeave()
      // The most recently used of the new crewmates opens; the roster is the
      // confirmation. Exact name, not slug — MembersPage resolves `?member=`
      // by name. `landed` first: a crewmate created in an earlier, partially
      // failed attempt is as new as the ones from this one.
      const all = [...candidates.filter(c => landed.has(c.name)), ...created]
      const open = all.sort((a, b) => b.last_used_ts - a.last_used_ts)[0]
      if (open) navigate(`/members?member=${encodeURIComponent(open.name)}`, { replace: true })
    },
    onError: (err: AddFailure) => {
      setFailedName(err.failedName ?? null)
      if (err.created?.length) {
        const names = err.created.map(c => c.name)
        setLanded(prev => new Set([...prev, ...names]))
        void queryClient.invalidateQueries({ queryKey: ['kirocrew-agents'] })
      }
    },
  })
  const pending = add.isPending || finish.isPending

  // Escape means "Not now" — the same exit, recorded the same way — but never
  // while a write is in flight. No focus restore: the page behind has nothing
  // that had focus before the step opened over it.
  useDialogFocusTrap(dialogRef, () => { if (!pending) finish.mutate() }, { restoreFocus: false })

  return (
    <OnboardingChapterShell
      ariaLabel={t('pages.membersPage.optin_title')}
      panelHeadline={t('pages.membersPage.optin_aside_headline')}
      panelBody={t('pages.membersPage.optin_aside_body')}
      panelFootnote={t('pages.membersPage.optin_aside_footnote')}
      eyebrow={t('pages.membersPage.optin_eyebrow')}
      dialogRef={dialogRef}
      header={
        <div className="mt-6">
          <h1 className="text-2xl font-semibold text-text-strong outline-hidden" data-testid="optin-title">
            {t('pages.membersPage.optin_title')}
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-muted">{t('pages.membersPage.optin_lead')}</p>
        </div>
      }
      footer={
        <>
          <Btn
            type="button"
            onClick={() => finish.mutate()}
            disabled={pending}
            className="px-3 min-h-9 text-sm"
            data-testid="optin-not-now"
          >
            {t('pages.membersPage.optin_not_now')}
          </Btn>
          <SendBtn
            type="button"
            disabled={count === 0 || pending}
            aria-busy={add.isPending || undefined}
            onClick={() => {
              setFailedName(null)
              add.mutate(chosen)
            }}
            data-testid="optin-add"
          >
            {t('pages.membersPage.optin_add', { count })}
          </SendBtn>
        </>
      }
    >
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[.04em] text-muted">
        {t('pages.membersPage.optin_list_heading', { count: candidates.length })}
      </p>
      <ul className="m-0 list-none divide-y divide-border rounded-lg border border-border bg-bg" data-testid="optin-list">
        {candidates.map(c => {
          const done = landed.has(c.name)
          const on = done || !!picked[c.name]
          const locked = done || pending
          const id = `optin-${c.name}`
          return (
            <li key={c.name}>
              <label
                htmlFor={id}
                className={`flex items-center gap-3 px-3.5 py-2.5 transition-colors ${
                  locked ? 'cursor-default' : 'cursor-pointer hover:bg-bg-hover'
                } ${on ? '' : 'opacity-80'}`}
              >
                <Checkbox
                  id={id}
                  checked={on}
                  disabled={locked}
                  onChange={e => setPicked(p => ({ ...p, [c.name]: e.target.checked }))}
                  className="h-4 w-4 shrink-0"
                />
                <CrewAvatar seed={c.name} size={36} className={on ? '' : 'grayscale-[.6]'} />
                <span className="min-w-0 flex-1">
                  <span className={`block text-sm font-medium ${on ? 'text-text-strong' : 'text-text'}`}>{c.name}</span>
                  {c.description && (
                    <span className="block text-[12px] leading-snug text-muted truncate">{c.description}</span>
                  )}
                </span>
                {c.chats === 0 ? (
                  <span className="shrink-0 text-[11px] italic text-muted/80">
                    {t('pages.membersPage.optin_chats_none')}
                  </span>
                ) : (
                  <span className="shrink-0 text-[11px] text-muted">
                    {t('pages.membersPage.optin_chats', { count: c.chats })}
                  </span>
                )}
              </label>
            </li>
          )
        })}
      </ul>
      <p className="mt-4 text-[12px] leading-relaxed text-muted" data-testid="optin-note">
        {t('pages.membersPage.optin_note')}
      </p>
      {/* No hand-off: the checklist selection above is an unsaved draft. */}
      <ErrorNotice
        message={
          failedName
            ? t('pages.membersPage.optin_add_failed', { name: failedName })
            : closeFailed
              ? t('pages.membersPage.optin_close_failed')
              : null
        }
        className="mt-4"
        testId="optin-error"
      />
    </OnboardingChapterShell>
  )
}
