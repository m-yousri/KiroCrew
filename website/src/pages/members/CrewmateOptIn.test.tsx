import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, fireEvent, waitFor } from '@testing-library/react'
import { renderWithProviders } from '../../test/helpers'

/* The one-time opt-in for an existing user with custom agents and no
 * crewmate. These cases pin the contract the launch review settled: it opens
 * only when the server says it is due, pre-checks the agents that were used,
 * creates one crewmate per checked agent through the ONE create route, lands
 * on the most recently used new crewmate, and records itself as over on both
 * exits. A partial add keeps the step open with the refusing name and locks
 * what already landed. */

vi.mock('../../api/client', () => ({
  api: {
    membersOptin: vi.fn(),
    membersOptinDone: vi.fn(() => Promise.resolve({ ok: true, done: true })),
    createKirocrewAgent: vi.fn(() => Promise.resolve({ ok: true })),
  },
}))

/* The gate reads first-run state through the optional theme hook; the
 * provider itself stays real so the render helper keeps working. */
vi.mock('../../hooks/useTheme', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../hooks/useTheme')>()
  return {
    ...actual,
    useOptionalTheme: () => ({ onboarded: true, importOnboarded: true, privacyAcked: true }),
  }
})

const navigateSpy = vi.fn()
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return { ...actual, useNavigate: () => navigateSpy }
})

import { api } from '../../api/client'
import CrewmateOptIn from './CrewmateOptIn'

const membersOptin = vi.mocked(api.membersOptin)
const membersOptinDone = vi.mocked(api.membersOptinDone)
const createKirocrewAgent = vi.mocked(api.createKirocrewAgent)

const CANDIDATES = [
  { name: 'radar', description: 'Triages issues.', source: 'builtin', chats: 42, last_used_ts: 1000 },
  { name: 'scribe', description: 'Drafts release notes.', source: 'builtin', chats: 17, last_used_ts: 2000 },
  { name: 'scout', description: '', source: 'builtin', chats: 0, last_used_ts: 0 },
]

function due(overrides: Partial<{ done: boolean; crewmates: number; candidates: typeof CANDIDATES }> = {}) {
  membersOptin.mockResolvedValue({ done: false, crewmates: 0, candidates: CANDIDATES, ...overrides })
}

beforeEach(() => {
  vi.clearAllMocks()
  membersOptinDone.mockResolvedValue({ ok: true, done: true })
  createKirocrewAgent.mockResolvedValue({ ok: true })
})

describe('CrewmateOptIn', () => {
  it('opens when due, pre-checks the used agents and counts them in the primary', async () => {
    due()
    renderWithProviders(<CrewmateOptIn />)
    const dialog = await screen.findByRole('dialog', { name: 'Meet your crewmates' })
    expect(dialog).toBeInTheDocument()
    expect(screen.getByTestId('optin-list').querySelectorAll('li')).toHaveLength(3)
    expect((screen.getByLabelText(/radar/) as HTMLInputElement).checked).toBe(true)
    expect((screen.getByLabelText(/scribe/) as HTMLInputElement).checked).toBe(true)
    expect((screen.getByLabelText(/scout/) as HTMLInputElement).checked).toBe(false)
    expect(screen.getByText('used in 0 chats')).toBeInTheDocument()
    expect(screen.getByText('42 chats')).toBeInTheDocument()
    expect(screen.getByTestId('optin-add')).toHaveTextContent('Add 2 crewmates')
    expect(screen.getByTestId('optin-note')).toHaveTextContent('Nothing changes for your existing chats.')
  })

  it.each([
    ['already done', { done: true }],
    ['a crewmate exists', { crewmates: 1 }],
    ['no custom agents', { candidates: [] }],
  ])('stays off when %s', async (_label, overrides) => {
    due(overrides as Parameters<typeof due>[0])
    renderWithProviders(<CrewmateOptIn />)
    await waitFor(() => expect(membersOptin).toHaveBeenCalled())
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('the primary follows the selection and disables at zero', async () => {
    due()
    renderWithProviders(<CrewmateOptIn />)
    await screen.findByTestId('optin-add')
    fireEvent.click(screen.getByLabelText(/scribe/))
    expect(screen.getByTestId('optin-add')).toHaveTextContent('Add 1 crewmate')
    fireEvent.click(screen.getByLabelText(/radar/))
    expect(screen.getByTestId('optin-add')).toHaveTextContent('Add 0 crewmates')
    expect(screen.getByTestId('optin-add')).toBeDisabled()
  })

  it('adds one crewmate per checked agent through the create route and opens the most recent one', async () => {
    due()
    renderWithProviders(<CrewmateOptIn />)
    fireEvent.click(await screen.findByTestId('optin-add'))
    await waitFor(() => expect(navigateSpy).toHaveBeenCalled())
    expect(createKirocrewAgent.mock.calls.map(([body]) => body)).toEqual([
      { name: 'radar', kiro_agent: 'radar', description: 'Triages issues.', source: 'kirocrew' },
      { name: 'scribe', kiro_agent: 'scribe', description: 'Drafts release notes.', source: 'kirocrew' },
    ])
    expect(membersOptinDone).toHaveBeenCalledTimes(1)
    // scribe was used more recently than radar.
    expect(navigateSpy).toHaveBeenCalledWith('/members?member=scribe', { replace: true })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('Not now records the step as over and creates nothing', async () => {
    due()
    renderWithProviders(<CrewmateOptIn />)
    fireEvent.click(await screen.findByTestId('optin-not-now'))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(membersOptinDone).toHaveBeenCalledTimes(1)
    expect(createKirocrewAgent).not.toHaveBeenCalled()
    expect(navigateSpy).not.toHaveBeenCalled()
  })

  it('a refused Not now keeps the step open and says so', async () => {
    due()
    membersOptinDone.mockRejectedValueOnce(new Error('503'))
    renderWithProviders(<CrewmateOptIn />)
    fireEvent.click(await screen.findByTestId('optin-not-now'))
    await screen.findByTestId('optin-error')
    expect(screen.getByTestId('optin-error')).toHaveTextContent('Your choice could not be saved. Try again.')
    expect(screen.getByRole('dialog')).toBeInTheDocument()
  })

  it('a partial add names the refusing agent, locks what landed and retries only the rest', async () => {
    due()
    createKirocrewAgent
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('409'))
    renderWithProviders(<CrewmateOptIn />)
    fireEvent.click(await screen.findByTestId('optin-add'))
    await screen.findByTestId('optin-error')
    expect(screen.getByTestId('optin-error')).toHaveTextContent('Could not add “scribe”. Try again.')
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(navigateSpy).not.toHaveBeenCalled()
    // radar landed: checked, locked, and no longer counted in the primary.
    const radar = screen.getByLabelText(/radar/) as HTMLInputElement
    expect(radar.checked).toBe(true)
    expect(radar).toBeDisabled()
    expect(screen.getByTestId('optin-add')).toHaveTextContent('Add 1 crewmate')
    // The retry sends only scribe, then lands on the most recent of BOTH.
    createKirocrewAgent.mockResolvedValueOnce({ ok: true })
    fireEvent.click(screen.getByTestId('optin-add'))
    await waitFor(() => expect(navigateSpy).toHaveBeenCalled())
    expect(createKirocrewAgent.mock.calls.at(-1)?.[0]).toMatchObject({ name: 'scribe' })
    expect(createKirocrewAgent).toHaveBeenCalledTimes(3)
    expect(navigateSpy).toHaveBeenCalledWith('/members?member=scribe', { replace: true })
  })
})
