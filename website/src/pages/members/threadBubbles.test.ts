import { describe, expect, it } from 'vitest'
import { bubbleRadiusStyle, runPositions } from './threadBubbles'

const at = (minute: number) => new Date(Date.UTC(2026, 8, 22, 7, minute)).toISOString()

describe('runPositions', () => {
  it('groups consecutive replies by one author within the gap', () => {
    const pos = runPositions([
      { author: 'user', ts: at(41) },
      { author: 'assistant', ts: at(41) },
      { author: 'assistant', ts: at(41) },
      { author: 'assistant', ts: at(42) },
      { author: 'user', ts: at(44) },
    ])
    expect(pos).toEqual(['single', 'start', 'cont', 'end', 'single'])
  })

  it('breaks a run after five minutes of silence', () => {
    expect(runPositions([{ author: 'assistant', ts: at(0) }, { author: 'assistant', ts: at(6) }])).toEqual(['single', 'single'])
  })

  it('treats a missing timestamp as chained', () => {
    expect(runPositions([{ author: 'assistant' }, { author: 'assistant' }])).toEqual(['start', 'end'])
  })
})

describe('bubbleRadiusStyle', () => {
  it('shrinks only the run side, and mirrors for the right', () => {
    const left = bubbleRadiusStyle('cont', 'left')
    expect(left.borderTopLeftRadius).toBe('0.375rem')
    expect(left.borderBottomLeftRadius).toBe('0.375rem')
    expect(left.borderTopRightRadius).toBe('1rem')
    const right = bubbleRadiusStyle('start', 'right')
    expect(right.borderBottomRightRadius).toBe('0.375rem')
    expect(right.borderTopRightRadius).toBe('1rem')
    expect(right.borderTopLeftRadius).toBe('1rem')
    expect(bubbleRadiusStyle('single', 'left').borderBottomLeftRadius).toBe('1rem')
  })
})
