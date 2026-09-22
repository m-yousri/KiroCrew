/**
 * The corner rule for a run of consecutive bubbles by one author (screen 05's
 * grouped bubbles), as inline radii for the thread panel's small bubbles:
 *
 *   single       all four corners full
 *   first of run full, except the run side's bottom corner small
 *   middle       the run side's top AND bottom corners small
 *   last         the run side's top corner small, rest full
 *
 * The far side stays full. `side` is where the run sits: the crewmate's
 * replies on the left, the user's on the right.
 *
 * Local to the thread panel until the main chat's grouped-bubble module lands;
 * the panel then reads the rule from there instead.
 */

export type BubblePos = 'single' | 'start' | 'cont' | 'end'

/** Tailwind v4 defaults: rounded-2xl = 1rem, rounded-md = 0.375rem. */
const R_FULL = '1rem'
const R_SMALL = '0.375rem'

/** A run breaks after this much silence, Slack-style. */
export const RUN_GAP_MS = 5 * 60_000

/**
 * Position of each message in a run of consecutive messages by the same
 * author. Two adjacent messages by that author are one run only when the
 * second follows within `gapMs` of the first.
 */
export function runPositions(msgs: { author: string; ts?: string }[], gapMs = RUN_GAP_MS): BubblePos[] {
  const t = (m: { ts?: string } | undefined) => (m?.ts ? new Date(m.ts).getTime() : NaN)
  const chained = (a: { author: string; ts?: string } | undefined, b: { author: string; ts?: string } | undefined) => {
    if (!a || !b || a.author !== b.author) return false
    const gap = t(b) - t(a)
    return Number.isNaN(gap) ? true : gap >= 0 && gap <= gapMs
  }
  return msgs.map((m, i) => {
    const first = !chained(msgs[i - 1], m)
    const last = !chained(m, msgs[i + 1])
    return first && last ? 'single' : first ? 'start' : last ? 'end' : 'cont'
  })
}

export function bubbleRadiusStyle(pos: BubblePos, side: 'left' | 'right'): React.CSSProperties {
  const topSmall = pos === 'cont' || pos === 'end'
  const bottomSmall = pos === 'cont' || pos === 'start'
  const top = topSmall ? R_SMALL : R_FULL
  const bottom = bottomSmall ? R_SMALL : R_FULL
  return side === 'left'
    ? { borderTopLeftRadius: top, borderBottomLeftRadius: bottom, borderTopRightRadius: R_FULL, borderBottomRightRadius: R_FULL }
    : { borderTopRightRadius: top, borderBottomRightRadius: bottom, borderTopLeftRadius: R_FULL, borderBottomLeftRadius: R_FULL }
}
