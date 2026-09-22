// DirectDeployFlow — everything the direct-deploy flow renders BESIDE its button
// (issue #12816), so the Artifact Deploy page's row and the webapp artifact card
// share one copy of the acknowledgment and the three refusal affordances instead
// of wiring them twice.
//
// The caller owns the button (the two sites style and label it differently) and
// drops this in; `useDirectDeploy` owns the state machine.
import { AlertCircle, AlertTriangle, Check, ExternalLink, Rocket } from 'lucide-react'
import PublicPublishAckModal from './PublicPublishAckModal'
import ErrorDetails from './ErrorDetails'
import { Btn } from './ui'
import { safeHttpUrl } from '../lib/safeUrl'
import type { useDirectDeploy } from '../hooks/useDirectDeploy'

import { i18nT } from '../i18n/t'

export default function DirectDeployFlow({
  slug,
  flow,
}: {
  slug: string
  flow: ReturnType<typeof useDirectDeploy>
}) {
  const { phase, confirm, reset, ttlHours } = flow

  return (
    <>
      {/* Reaper precondition: a finite TTL needs auto-cleanup infrastructure the
          account does not have. Two ways forward rather than a dead banner —
          permanent needs no infrastructure at all, and the install command is
          exact. The stack names live behind Details. */}
      {phase.kind === 'refused' && phase.refusal.code === 'reaper_required' && (
        <div className="mt-2 flex flex-col gap-2 rounded border border-warn/30 bg-warn-subtle p-2.5">
          <div className="flex items-start gap-2 text-[12px] text-warn">
            <AlertTriangle className="lucide-inline shrink-0" />
            <span>{phase.refusal.error}</span>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Btn primary onClick={() => {
              // Re-run the SAME previewed content at ttl_hours=0 rather than
              // making the user start the flow over. There is no preview to bind
              // here (the refusal came from the confirm call), so the digest
              // fields are left empty and the backend re-scans.
              void confirm(
                { content_digest: '', profile: '', region: '', bytes: 0, scan: '', site_id: slug },
                false,
                0,
              )
            }}>
              <Rocket size={11} /> {i18nT('components.directDeploy.deploy_as_permanent')}
            </Btn>
            <Btn onClick={reset}>{i18nT('components.publishHub.cancel')}</Btn>
          </div>
          <ErrorDetails details={phase.refusal.details} remediation={phase.refusal.remediation} />
        </div>
      )}

      {/* No built static root: this flow cannot publish the app at all. The
          caller has already swapped its button for the agent hand-off, so this
          only explains why. */}
      {phase.kind === 'refused' && phase.refusal.code === 'webapp_root_unavailable' && (
        <div className="mt-2 flex flex-col gap-1.5 rounded border border-border bg-bg p-2.5">
          <div className="flex items-start gap-2 text-[12px] text-muted">
            <AlertCircle className="lucide-inline shrink-0" />
            <span>{phase.refusal.error}</span>
          </div>
          <ErrorDetails details={phase.refusal.details} remediation={phase.refusal.remediation} />
        </div>
      )}

      {/* Scan gate. Credential findings can NEVER be overridden — offering a
          button there would teach the user to click past the one refusal that
          does not bend. */}
      {phase.kind === 'scan-blocked' && (
        <div className="mt-2 flex flex-col gap-2 rounded border border-warn/30 bg-warn-subtle p-2.5">
          <div className="flex items-center gap-2 text-[12px] text-warn">
            <AlertCircle size={13} />
            {i18nT('components.publishHub.scan_blocked_finding', { count: phase.block.count })}
          </div>
          <div className="text-[11px] text-muted whitespace-pre-line">{phase.block.findings}</div>
          {phase.block.credential ? (
            <>
              <div className="text-[11px] font-medium text-warn">
                {i18nT('components.publishHub.credential_security_findings_cannot_be_overridde')}
              </div>
              <div><Btn onClick={reset}>{i18nT('components.publishHub.cancel')}</Btn></div>
            </>
          ) : (
            <div className="flex gap-2">
              <Btn danger onClick={() => {
                const p = phase.preview
                  ?? { content_digest: '', profile: '', region: '', bytes: 0, scan: '', site_id: slug }
                void confirm(p, true)
              }}>
                {i18nT('pages.artifactDeployPage.deploy_anyway')}
              </Btn>
              <Btn onClick={reset}>{i18nT('components.publishHub.cancel')}</Btn>
            </div>
          )}
        </div>
      )}

      {phase.kind === 'failed' && (
        <div className="mt-2 flex items-start gap-2 rounded border border-danger/40 bg-danger-subtle p-2.5 text-[12px] text-danger">
          <AlertCircle className="lucide-inline shrink-0" />
          <span>{phase.message}</span>
        </div>
      )}

      {phase.kind === 'done' && (
        <div className="mt-2 flex items-center gap-2 text-[12px] text-ok">
          <Check size={13} /> {i18nT('components.publishHub.published')}
          {phase.url && safeHttpUrl(phase.url) && (
            <a
              href={safeHttpUrl(phase.url)!}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-accent hover:underline"
            >
              <ExternalLink size={12} /> {phase.url}
            </a>
          )}
        </div>
      )}

      {/* The blocking public-by-link acknowledgment. Held MOUNTED and
          busy-disabled until the deploy settles: closing first hands the exiting
          <AnimatePresence> subtree an enabled confirm button for the exit
          duration, which is a second deploy waiting to happen. */}
      <PublicPublishAckModal
        open={phase.kind === 'ack'}
        target={slug}
        ttlHours={ttlHours}
        busy={phase.kind === 'deploying'}
        onCancel={reset}
        onConfirm={() => {
          if (phase.kind !== 'ack') return
          void confirm(phase.preview, phase.overrideScan)
        }}
      />
    </>
  )
}
