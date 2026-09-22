/**
 * Screenshot harness for the Crew page work-item board (RFC Phase 4, the surfaces).
 *
 * Runs the REAL built SPA (website/dist) behind the shared in-process static server
 * with SPA fallback, and answers GET /api/crew-board from a fixture.
 *
 * The two fixtures are not hand-written. They are the VERBATIM output of the real
 * handler — `dashboard/handlers/work_ledger_board.api_work_ledger_board` called
 * in-process over a seeded two-conductor work-ledger store, then dumped. That
 * matters for what these shots are evidence OF: a hand-written payload would agree
 * with the page by construction, so it could not show that the masked projection
 * and the page actually fit together. Neither fixture contains a `chat-*` worker
 * key or the string `worker_session_key`, which is the masking criterion holding.
 *
 * Two scenes, because orphanhood is a property of the CONDUCTOR's session and the
 * two states show different affordances:
 *
 *   alive     — the conductor's slot is open. Nothing is orphaned, so no row offers
 *               an action. The ordinary board: ruling band, working band, and
 *               finished items collapsed behind the expander.
 *   orphaned  — the conductor's slot is gone, so every open item is orphaned. This
 *               is the state Phase 4 names its affordances for. Stop is live on the
 *               item whose worker is still running, disabled with a reason on the
 *               one whose session has closed, and Take over is disabled everywhere
 *               because no server-side primitive exists to perform it.
 *
 * Each scene in light and dark, so the theme-variable claim is checked rather than
 * asserted: a hard-coded colour would survive one of the two and fail the other.
 *
 * Usage: node scripts/capture-crew-board.mjs [outDir]
 *
 * Writes into `docs/request-for-change/assets/` by DEFAULT, and those four files
 * are committed. Design and UX review both read rendered evidence from the
 * revision itself, so evidence that lives only in a gitignored scratch directory
 * is evidence no reviewer can see. Defaulting here means re-running the harness
 * refreshes the committed images rather than silently leaving them stale.
 */
import { chromium } from 'playwright'
import { mkdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { serveDist } from './lib/serve-dist.mjs'
import { bootApiStub } from './lib/boot-stubs.mjs'

const OUT = process.argv[2] || '../docs/request-for-change/assets'
const PREFIX = 'crew-board-'
const FIXTURES = fileURLToPath(new URL('./fixtures/', import.meta.url))

mkdirSync(OUT, { recursive: true })

const scene = (name) =>
  JSON.parse(readFileSync(`${FIXTURES}crew-board-${name}.json`, 'utf8'))

const SCENES = { alive: scene('alive'), orphaned: scene('orphaned') }
const CONDUCTOR = SCENES.alive.conductor.slot_key

async function shoot(page, base, { scene, mode, file }) {
  await page.addInitScript((theme) => {
    localStorage.clear()
    localStorage.setItem('mc-theme', theme)
    localStorage.setItem('mc-onboarded', '1')
  }, mode)

  // The shell opens a live socket on boot. Nothing here needs it, and left
  // unrouted it logs a handshake error on every run of this harness.
  await page.routeWebSocket(/\/api\/ws/, () => {})

  // Only `/api/crew-board` is this harness's own; everything else the shell asks
  // for on boot comes from the shared stub, which is why that block no longer sits
  // inline here.
  await page.route('**/api/**', bootApiStub({
    mode,
    routes: { '/api/crew-board': SCENES[scene] },
  }))

  await page.goto(`${base}/crew-board?conductor=${encodeURIComponent(CONDUCTOR)}`, {
    waitUntil: 'domcontentloaded',
  })

  // Wait on CONTENT from the fixture, not a timeout: if the board failed to render
  // the shot must fail here rather than quietly capturing an empty page.
  await page.getByText('Needs a ruling').waitFor({ timeout: 20000 })
  await page.getByText('Wake hook on the ledger append path').waitFor({ timeout: 20000 })

  // Expand the finished band so the terminal collapse is visible in both states.
  const expander = page.getByRole('button', { name: /Finished/i })
  if (await expander.count()) await expander.first().click()

  await page.waitForTimeout(400)
  await page.screenshot({ path: `${OUT}/${PREFIX}${file}`, fullPage: true })
  console.log('wrote', `${OUT}/${PREFIX}${file}`)

  await page.unroute('**/api/**')
}

async function main() {
  const { srv, base } = await serveDist()
  const browser = await chromium.launch()

  for (const [scene, label] of [['alive', 'board'], ['orphaned', 'orphaned-actions']]) {
    for (const mode of ['light', 'dark']) {
      const context = await browser.newContext({
        viewport: { width: 1280, height: 900 },
        deviceScaleFactor: 2,
        colorScheme: mode,
      })
      const page = await context.newPage()
      page.on('pageerror', err => console.log('PAGEERROR:', String(err).slice(0, 300)))
      page.on('console', msg => {
        if (msg.type() === 'error') console.log('CONSOLE:', msg.text().slice(0, 300))
      })
      await shoot(page, base, { scene, mode, file: `${label}-${mode}.png` })
      await context.close()
    }
  }

  await browser.close()
  srv.close()
}

main().catch(err => { console.error(err); process.exit(1) })
