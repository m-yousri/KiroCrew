/**
 * Notes tab — the crewmate's own standing notes ("what it learned").
 *
 * The body is the crewmate's self-maintained briefing markdown
 * (`members/<slug>/briefing.md`), read through
 * `GET /api/members/{slug}/briefing` and rendered by the real
 * `MarkdownRenderer`. Read-only here by design: the crewmate writes this file
 * as it works, and a human who wants to change it opens the file in the
 * panel's file viewer (the Edit button), which is the dashboard's one editor
 * for files on disk. Six states, never conflated: loading, a refused read
 * because two crewmates share the slug (the file would belong to neither),
 * failed read, unsupported platform (the backend fails closed rather than
 * reading the file racily), no notes yet (the normal state of a fresh
 * crewmate — an empty state, not an error), and content.
 */
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { Pencil } from 'lucide-react'
import { api } from '../../api/client'
import { ApiError } from '../../api/apiError'
import { memberBriefingQueryKey } from '../../api/membersQuery'
import { Btn, Skeleton } from '../../components/ui'
import ErrorNotice from '../../components/ErrorNotice'
import MarkdownRenderer from '../../components/MarkdownRenderer'
import { timeAgo } from '../../utils/timeAgo'

interface CrewNotesTabProps {
  slug: string
  /** The exact crew name — slugs are lossy, so the read is keyed by both. */
  member: string
  /** The identity row (avatar + name + live status) the three panel tabs share. */
  header: ReactNode
  /** Whether this tab body is on screen — gates the briefing read, so a panel
   *  showing another tab does not pay for notes nobody is looking at. */
  visible: boolean
  /** Opens the notes file as a document tab in the same panel. */
  onOpenFile: (path: string) => void
}

/** The backend refused the read because two crewmates derive this slug: the
 *  notes file is shared, so it is nobody's to show or edit. A state of its own,
 *  in plain words, not a generic failure — the fix is a rename, not a retry. */
function isSlugCollision(error: unknown): boolean {
  return error instanceof ApiError && error.status === 409
}

export default function CrewNotesTab({ slug, member, header, visible, onOpenFile }: CrewNotesTabProps) {
  const { t } = useTranslation()
  const query = useQuery({
    queryKey: memberBriefingQueryKey(slug, member),
    queryFn: () => api.memberBriefing(slug, member),
    enabled: visible && !!slug && !!member,
    // A 409 is a stable answer about the roster, not a transient fault.
    retry: (count, error) => !isSlugCollision(error) && count < 3,
  })
  const data = query.data
  // Pending and failed are told apart the way every block on this page does
  // it: a failed read must not render the affirmative empty state.
  const loading = data === undefined && !query.isError
  const collision = data === undefined && query.isError && isSlugCollision(query.error)
  const failed = data === undefined && query.isError && !collision
  const editable = !!data?.path && data.supported !== false

  return (
    <div className="flex flex-col h-full" data-testid="member-notes">
      <div className="px-3 pt-3 shrink-0">{header}</div>
      <div className="flex-1 min-h-0 overflow-y-auto px-3 pb-3">
        {loading ? (
          <div className="space-y-2" data-testid="member-notes-loading" aria-hidden>
            <Skeleton className="h-3 w-3/4" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
          </div>
        ) : collision ? (
          <p className="text-[12px] text-muted" data-testid="member-notes-collision">
            {t('pages.membersPage.notes_collision')}
          </p>
        ) : failed ? (
          /* The shared notice, not a hand-rolled alert. askAgent is safe here:
             a read failure on a read-only surface holds no draft to lose. */
          <ErrorNotice
            message={t('pages.membersPage.notes_error')}
            variant="inline"
            askAgent
            testId="member-notes-error"
          />
        ) : data?.supported === false ? (
          <p className="text-[12px] text-muted" data-testid="member-notes-unsupported">
            {t('pages.membersPage.notes_unsupported')}
          </p>
        ) : !data?.text ? (
          <div className="space-y-1.5" data-testid="member-notes-empty">
            <p className="text-[12px] text-text">{t('pages.membersPage.notes_empty', { name: member })}</p>
            <p className="text-[11.5px] text-muted">{t('pages.membersPage.notes_empty_hint')}</p>
          </div>
        ) : (
          <div className="msg-content text-[13px] leading-relaxed" data-testid="member-notes-body">
            <MarkdownRenderer content={data.text} />
          </div>
        )}
      </div>
      {/* Footer: when the notes were last written, and the one way to change
          them. Rendered once the read has answered (either way): a footer over
          a skeleton would date notes that have not arrived. */}
      {data && (
        <div
          className="flex items-center gap-2 px-3 py-1.5 border-t border-border text-[10.5px] text-muted shrink-0"
          data-testid="member-notes-footer"
        >
          <span className="truncate">
            {data.updated_ts ? t('pages.membersPage.notes_updated', { when: timeAgo(data.updated_ts) }) : null}
          </span>
          <Btn
            className="ml-auto"
            onClick={() => onOpenFile(data.path)}
            disabled={!editable}
            title={editable ? undefined : t('pages.membersPage.notes_edit_unavailable')}
            data-testid="member-notes-edit"
          >
            <Pencil className="lucide-inline" aria-hidden="true" />
            {t('pages.membersPage.notes_edit')}
          </Btn>
        </div>
      )}
    </div>
  )
}
