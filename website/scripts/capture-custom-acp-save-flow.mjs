/**
 * Screenshots of the Custom ACP form's SAVE FLOW, via the capture/custom-acp-form
 * harness (the real AgentBackendTab, only `fetch` stubbed, stub stateful across the
 * page so a PATCH becomes what later GETs answer with).
 *
 * The stills the empty/configured scenes give are both "Save disabled" -- the
 * blank form and the already-saved form -- so a reader cannot see what makes the
 * form's contents stick. This walks the write from the empty scene:
 *
 *   form-empty.png       the unconfigured scene: row "Not set up", strip pointing at
 *                        the form, no Check again, no Remove control; every
 *                        placeholder reads as an example ("e.g. …")
 *   0-incomplete.png     the incomplete warning with a field still blank (the state
 *                        between blank and filled), Save disabled
 *   1-filled.png         every field filled, Save ENABLED, no incomplete warning
 *   2-saving.png         Save pressed, "Saving…" while the PATCH is held open
 *   3-saved.png          "Saved. The command was found." and the
 *                        row re-checked to Ready with Use Custom ACP offered
 *   form-configured.png  the configured scene: the strip carries a descriptor
 *                        sentence for this row, and Remove Custom ACP is offered
 *   4-confirm.png        Remove Custom ACP pressed: the question and its Keep it /
 *                        Yes, remove it answers, nothing written yet
 *   5-removed.png        removal confirmed: fields emptied, "Removed." line
 *
 * Each frame asserts the state it documents before the shutter, so a stale build
 * or a regressed form fails the script rather than producing a misleading PNG.
 *
 * Usage: node scripts/capture-custom-acp-save-flow.mjs <viteBase> <outDir>
 */
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright'

const base = process.argv[2] || 'http://localhost:5199'
const out = process.argv[3] || '../temp-screenshots/custom-acp'
mkdirSync(out, { recursive: true })

const b = await chromium.launch()
const p = await (
  await b.newContext({
    viewport: { width: 1100, height: 1000 },
    deviceScaleFactor: 2,
  })
).newPage()
const errors = []
p.on('pageerror', (e) => errors.push(`PAGEERROR: ${e.message}`))

// A held PATCH is what gives "Saving…" a frame to be caught in.
await p.goto(`${base}/capture/custom-acp-form.html?scene=empty&theme=dark&patch_delay_ms=2500`, {
  waitUntil: 'networkidle',
})
await p.getByRole('tab', { name: /Custom ACP/ }).click()
const form = p.getByRole('form', { name: 'Custom ACP agent' })
await form.waitFor({ timeout: 20_000 })
const save = form.getByRole('button', { name: 'Save' })
const shot = async (name) => {
  const file = `${out}/custom-acp-save-${name}.png`
  await p.locator('[data-capture-root]').screenshot({ path: file })
  console.log(`captured ${file}`)
}
const still = async (name) => {
  const file = `${out}/custom-acp-form-${name}.png`
  await p.locator('[data-capture-root]').screenshot({ path: file })
  console.log(`captured ${file}`)
}

// The empty scene as a still: one word for one state. The row reads "Not set up"
// (not "Missing" -- nothing is absent from the machine), the strip points at the
// form, and no Check again is offered for a record that was never written.
const customTab = p.getByRole('tab', { name: /Custom ACP/ })
if (!/Not set up/.test((await customTab.textContent()) ?? '')) {
  throw new Error('unconfigured custom row does not read "Not set up"')
}
if (/Missing/.test((await customTab.textContent()) ?? '')) {
  throw new Error('unconfigured custom row still reads "Missing"')
}
const strip = p.locator('#agent-backend-status')
await strip.getByText('Not set up yet. Fill in the form below.').waitFor({ timeout: 20_000 })
if (await p.getByRole('button', { name: 'Check again' }).count()) {
  throw new Error('Check again offered for an unwritten record')
}
if (await p.getByRole('button', { name: 'Remove Custom ACP' }).count()) {
  throw new Error('Remove Custom ACP offered with nothing stored')
}
if ((await form.getByLabel(/^Command/).getAttribute('placeholder')) !== 'e.g. my-acp-agent') {
  throw new Error('command placeholder does not read as an example')
}
await still('empty')

// Between blank and filled: the reader is told which fields are still missing --
// only the two still empty, not the command they can already see.
await form.getByLabel(/^Command/).fill('my-acp')
await form.getByLabel(/^Arguments/).fill('serve\n--stdio')
await p
  .getByText('Still needed: Permission option and Permission value.')
  .waitFor({ timeout: 20_000 })
if (await save.isEnabled()) throw new Error('Save enabled with the permission option still blank')
await shot('0-incomplete')

await form.getByLabel(/^Permission option/).fill('mode')
await form.getByLabel(/^Permission value/).fill('read-only')
await p.waitForFunction(() => {
  const btn = [...document.querySelectorAll('form button')].find(
    (el) => el.textContent?.trim() === 'Save',
  )
  return btn instanceof HTMLButtonElement && !btn.disabled
})
if (await p.getByText(/^Still needed:/).count()) {
  throw new Error('incomplete warning still shown with every field filled')
}
await shot('1-filled')

await save.click()
await p.getByText('Saving…').waitFor({ timeout: 20_000 })
await shot('2-saving')

// The write triggered a config reload and a re-check; the status line closes with
// what the re-check found, the row must now be Ready and the switch offered, or the
// frame would show a save that changed nothing.
await p.getByText('Saved. The command was found.').waitFor({ timeout: 20_000 })
await p.getByRole('button', { name: /Use Custom ACP/ }).waitFor({ timeout: 20_000 })
await shot('3-saved')

// The configured scene as a still: the strip carries a sentence about THIS row,
// not the bare "Experimental" word that reads as one more input above the inputs.
await strip.getByText(/^The ACP agent you set up below\./).waitFor({ timeout: 20_000 })
await still('configured')

// The Use button's outcome, shown rather than guessed at: one click writes the
// switch, the row gains the leading dot and the word "In use", and the button
// reads "In use" while the row is the one new sessions take.
const use = p.getByRole('button', { name: 'Use Custom ACP' })
await use.waitFor({ timeout: 20_000 })
await use.click()
await p.getByRole('button', { name: 'In use' }).waitFor({ timeout: 20_000 })
if (!/In use/.test((await customTab.textContent()) ?? '')) {
  throw new Error('the custom row does not read "In use" after the switch')
}
await still('in-use')
// While this row is the one in use, the removal question says where new sessions
// go, not only which four fields are deleted -- the fact a first-time reader
// wanted before daring to answer it.
const inUseRemove = form.getByRole('button', { name: 'Remove Custom ACP' })
await inUseRemove.click()
const inUseQuestion = form.getByRole('group', { name: /^Remove Custom ACP\? / })
await inUseQuestion.waitFor({ timeout: 20_000 })
if (!/New sessions go back to Kiro CLI\./.test((await inUseQuestion.textContent()) ?? '')) {
  throw new Error('the in-use removal question does not name the fallback')
}
await still('in-use-remove')
await inUseQuestion.getByRole('button', { name: 'Keep it' }).click()
await form.getByRole('button', { name: 'Remove Custom ACP' }).waitFor({ timeout: 20_000 })
// Back to the default so the removal below is captured on a row that is not the
// one in use, the same state the earlier frames show. The Use button lives in the
// highlighted row's detail, so the default row is opened first.
await p.getByRole('tab', { name: /Kiro CLI/ }).click()
await p.getByRole('button', { name: 'Use Kiro CLI' }).click()
await p.getByRole('tab', { name: /Custom ACP/ }).click()
await use.waitFor({ timeout: 20_000 })

// Removal names its object and takes two clicks: the first turns the button into
// a question with Keep it / Yes, remove it; nothing is written until the second.
const remove = form.getByRole('button', { name: 'Remove Custom ACP' })
await remove.waitFor({ timeout: 20_000 })
await remove.click()
await form.getByText(/^Remove Custom ACP\? /).waitFor({ timeout: 20_000 })
if ((await form.getByLabel(/^Command/).inputValue()) === '') {
  throw new Error('fields emptied before the removal was confirmed')
}
// The question and its two answers are a row of their own above Save: no row
// holds three buttons, and focus moved to Keep it when Remove unmounted.
const question = form.getByRole('group', { name: /^Remove Custom ACP\? / })
const answers = await question.getByRole('button').allTextContents()
if (answers.join('|') !== 'Keep it|Yes, remove it') {
  throw new Error(`confirmation row holds ${JSON.stringify(answers)}`)
}
if (await question.getByRole('button', { name: 'Save' }).count()) {
  throw new Error('Save shares a row with the removal confirmation')
}
const focused = await p.evaluate(() => document.activeElement?.textContent ?? '')
if (focused !== 'Keep it') {
  throw new Error(`focus landed on ${JSON.stringify(focused)}, not Keep it`)
}
await shot('4-confirm')
await form.getByRole('button', { name: 'Yes, remove it' }).click()
await p
  .getByText('Removed. Custom ACP is not selectable until set up again.')
  .waitFor({ timeout: 20_000 })
if ((await form.getByLabel(/^Command/).inputValue()) !== '') {
  throw new Error('command field not emptied by the confirmed removal')
}
await shot('5-removed')

await b.close()
if (errors.length) {
  console.error('page errors:\n' + errors.join('\n'))
  process.exit(1)
}
