"""``nudge.wake`` -- does the owning session need to act on this tick?

An auto-nudge loop fires a full model turn on its owning session every interval.
``PrWatchProbe`` already turns an unchanged pull request into a free re-arm
(``irq.poll``), but it can only read TYPED facts: a check conclusion, a merge
state. A conductor patrolling worker transcripts, a babysitter reading bot
comment bodies, a loop watching a log all still pay a turn per tick, because the
evidence that would justify staying quiet is prose.

This point asks a cheap typed judge to read that prose and answer one question:
does the owner need to act now? Only a yes spends the turn.

Composes with the probe, never replaces it
------------------------------------------
The verdict produced here is an :class:`irq.Verdict` -- the same value the
kernel's own tick returns -- so the driver consumes one type from two producers
rather than growing a second vocabulary. The probe's observation is an INPUT to
this decision (``kind=probe`` evidence), which is why the two do not duplicate
each other: the probe types what it can, and the judge reads what it cannot.

Everything is a refusal toward SPENDING the turn
------------------------------------------------
:func:`map_answers` returns ``FALLBACK`` for a missing answer, an unusable one,
or any provider failure, and the driver fires exactly as the ungated timer would.
A wrongly-quiet tick is silence -- the loop stops waking and the work it watched
stalls with nothing on screen to say why -- while a wrongly-spent tick costs one
turn, which is what every tick costs today. So an unsure judge hands the call to
the main session (:data:`OUTCOME_MIN_P`) and never guesses quiet.

Three CHOICE questions, not noul and score
------------------------------------------
The design this implements asked for a ``noul`` (nullable boolean) and a
``score``. This build's seam speaks ``choice`` and nothing else, in three places:
``types.Question`` is an alias of ``Choice``, ``impl_jev._to_wire`` raises on any
other question class, and ``gate._answers_are_valid`` requires the answer value
to be a string drawn from the question's own options. Adding two wire types would
widen a trust boundary six merged points already share, against a response shape
no test here can speak for.

The decomposition is lossless for the mapping below. ``needs_owner`` over
``{wake, quiet}`` is exactly the ``noul``, with the probability of ``wake``
playing the nullable boolean's role: with two options the chosen one is the
argmax, so a ``wake`` answer already carries ``p >= 0.5``, and the bar is applied
literally anyway. ``urgency``'s three levels were already named values rather
than a continuum.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Mapping, Sequence

from kiro_crew import decisions as core
from kiro_crew import irq
from kiro_crew.decisions import gate as _gate
from kiro_crew.decisions.types import Answer, Answers, Choice, Question

logger = logging.getLogger(__name__)

POINT = "nudge.wake"

# --------------------------------------------------------------------------- #
# Bounds. Every one of these is an EGRESS bound: the state is assembled from
# other sessions' transcripts and third-party comment bodies, so each field is
# clipped before it can reach the wire rather than after.
# --------------------------------------------------------------------------- #

#: The whole request's ceiling, measured on the RENDERED state so the number
#: bounds what the wire carries rather than what the dict looks like.
MAX_STATE_CHARS = 8_000

#: One evidence item's text. A transcript tail or a review comment is prose of
#: unbounded length; the signal a judge needs from it -- a status prefix, a new
#: finding -- is at the start, so this clips rather than samples.
MAX_ITEM_CHARS = 1_000

#: The owner's own loop instruction. The judge needs to know what the loop is
#: FOR; it does not need the whole brief.
MAX_INSTRUCTION_CHARS = 1_500

#: One ``wake_when`` / ``quiet_when`` line. These become question CRITERIA, so
#: they leave the machine on every call and are bounded like any other egress.
MAX_CRITERION_CHARS = 500

#: An evidence item's ``source`` label (``session:chat-1751``,
#: ``pr:owner/name#123``). Bounded because it names a target the owner supplied.
MAX_SOURCE_CHARS = 200

#: How many evidence items are even considered before the char budget applies.
#: Bounds the WALK, so a loop watching twenty targets cannot turn one decision
#: into an unbounded assembly; the char budget is what bounds the send.
MAX_EVIDENCE_ITEMS = 40

# --------------------------------------------------------------------------- #
# Question identities and their option domains.
# --------------------------------------------------------------------------- #

Q_NEEDS_OWNER = "needs_owner"
Q_OUTCOME = "outcome"
Q_URGENCY = "urgency"

#: ``needs_owner``'s domain -- the two-option decomposition of a nullable boolean.
#: Named for what each option MEANS to the driver rather than ``yes``/``no``: the
#: judge is answering "wake or stay quiet", and an option whose name matches the
#: outcome it produces is one less mapping a reader has to hold.
NEEDS_OWNER_WAKE = "wake"
NEEDS_OWNER_QUIET = "quiet"
NEEDS_OWNER_OPTIONS = (NEEDS_OWNER_WAKE, NEEDS_OWNER_QUIET)

OUTCOME_NOTHING_NEW = "nothing_new"
OUTCOME_PROGRESS_ONLY = "progress_only"
OUTCOME_NEEDS_ACTION = "needs_action"
OUTCOME_NEEDS_HUMAN = "needs_human"
OUTCOME_FINISHED = "finished"
OUTCOME_BROKEN = "broken"
OUTCOME_OPTIONS = (
    OUTCOME_NOTHING_NEW,
    OUTCOME_PROGRESS_ONLY,
    OUTCOME_NEEDS_ACTION,
    OUTCOME_NEEDS_HUMAN,
    OUTCOME_FINISHED,
    OUTCOME_BROKEN,
)

#: The two outcomes that end a watch rather than service it.
TERMINAL_OUTCOMES = frozenset({OUTCOME_FINISHED, OUTCOME_BROKEN})

#: The two outcomes that need the owning session whatever ``needs_owner`` said.
ACTION_OUTCOMES = frozenset({OUTCOME_NEEDS_ACTION, OUTCOME_NEEDS_HUMAN})

#: The ONLY two outcomes that may cost the loop its turn. An allowlist rather
#: than "everything not matched above", because that catch-all had a hole in the
#: one direction this design cannot afford: a ``finished`` or ``broken`` answer
#: that misses :data:`TERMINAL_MIN_P` is not confident enough to END the watch,
#: and under a catch-all it fell through every rule above into QUIET -- so a judge
#: saying "I think this is finished, but only half sure" produced silence about
#: the one event the owner most needs to hear. Naming the quiet outcomes makes
#: every future outcome wake by default, which is the direction each threshold
#: here already resolves toward.
QUIET_OUTCOMES = frozenset({OUTCOME_NOTHING_NEW, OUTCOME_PROGRESS_ONLY})

URGENCY_NONE = "none"
URGENCY_NEXT_TICK = "next_tick_is_fine"
URGENCY_NOW = "now"
URGENCY_OPTIONS = (URGENCY_NONE, URGENCY_NEXT_TICK, URGENCY_NOW)

# --------------------------------------------------------------------------- #
# Thresholds. ONE place, so they are tunable from the JSONL log rather than from
# prompt text, and so a test can assert the table instead of the prose.
# --------------------------------------------------------------------------- #

#: How sure the judge must be before a watch is ENDED. Above
#: :data:`OUTCOME_MIN_P` by construction: ending a watch is the one verdict the
#: owner cannot recover by waiting one more interval, so it carries the higher
#: bar and a low-confidence ``finished`` becomes an ordinary wake instead.
TERMINAL_MIN_P = 0.6

#: The bar on the probability of ``wake``. Applied literally even though a
#: two-option argmax already clears it, so the documented threshold is the enforced
#: one for any provider that returns a non-argmax choice.
NEEDS_OWNER_MIN_P = 0.5

#: Below this on ``outcome``, the judge is not sure enough to be trusted with
#: silence and the call goes to the main session. The floor under every quiet
#: verdict, and the reason an uncalibrated judge costs turns rather than signals.
OUTCOME_MIN_P = 0.4

# --------------------------------------------------------------------------- #
# Evidence vocabulary.
# --------------------------------------------------------------------------- #

KIND_TRANSCRIPT_TAIL = "transcript_tail"
KIND_PR_CHECKS = "pr_checks"
KIND_PR_COMMENT = "pr_comment"
KIND_LEDGER_EVENT = "ledger_event"
KIND_PROBE = "probe"
#: Closed, like the point-name tuple: an item naming anything else is dropped
#: rather than sent, so a collector cannot invent a category nobody reviewed.
EVIDENCE_KINDS = frozenset(
    {
        KIND_TRANSCRIPT_TAIL,
        KIND_PR_CHECKS,
        KIND_PR_COMMENT,
        KIND_LEDGER_EVENT,
        KIND_PROBE,
    }
)


def build_questions(wake_when: str = "", quiet_when: str = "") -> list[Question]:
    """The three questions, with the owner's own words as ``needs_owner``'s criteria.

    The owner's ``wake_when`` / ``quiet_when`` are the ONLY caller-supplied text in
    any question. Each is attached to the option it describes, and both are clipped
    like every other egress field. Evidence never appears here: it goes in
    ``state``, because a question is an instruction and evidence is data, and the
    two must not be concatenated.

    The criteria ride in the PROMPT rather than in the provider's own per-option
    ``criteria`` map. That map exists on the wire but ``impl_jev._to_wire`` pins
    every entry to ``None``, so populating it means editing the request builder
    every shipped point shares. Carrying the same two sentences in the prompt costs
    the judge nothing and leaves that shared layer untouched.
    """
    wake = _clip(wake_when, MAX_CRITERION_CHARS)
    quiet = _clip(quiet_when, MAX_CRITERION_CHARS)
    prompt = "Does the new evidence require the owning session to act now?"
    if wake:
        prompt = f"{prompt} Answer {NEEDS_OWNER_WAKE} when: {wake}."
    if quiet:
        prompt = f"{prompt} Answer {NEEDS_OWNER_QUIET} when: {quiet}."
    return [
        Choice(Q_NEEDS_OWNER, prompt, options=list(NEEDS_OWNER_OPTIONS)),
        Choice(
            Q_OUTCOME,
            "What state is the watched work in?",
            options=list(OUTCOME_OPTIONS),
        ),
        Choice(
            Q_URGENCY,
            "How urgent is any action the owning session would take?",
            options=list(URGENCY_OPTIONS),
        ),
    ]


def evidence_item(source: str, kind: str, age_s: float, text: str) -> dict[str, Any] | None:
    """One screened evidence item, or ``None`` when it may not be sent.

    ``None`` for an unknown *kind*, for empty text, and for text the seam's scrub
    refuses. A refused item is DROPPED rather than redacted: this state is
    assembled from transcripts and third-party comment bodies, and a redaction
    that left a partially-cleaned credential in place would be a worse outcome
    than losing one observation on a path whose failure direction is to spend the
    turn anyway.
    """
    if kind not in EVIDENCE_KINDS:
        logger.debug("nudge.wake: dropping evidence of unknown kind")
        return None
    body = _clip(text, MAX_ITEM_CHARS)
    if not body:
        return None
    item = {
        "source": _clip(source, MAX_SOURCE_CHARS),
        "kind": kind,
        "age_s": _age(age_s),
        "text": body,
    }
    # The canonical pass, reused rather than reimplemented: one item as its own
    # tiny state, with no questions, so a hit is attributable to THIS item. The
    # gate re-scans the whole assembled request before anything is sent, so this
    # is the per-item drop the design asks for, not the only scan.
    if _gate.scrub_reason({"text": body}, []) is not None:
        logger.debug("nudge.wake: dropping evidence item the scrub refused")
        return None
    return item


def screen_evidence(items: Sequence[Mapping[str, Any]] | None) -> tuple[list[dict[str, Any]], int]:
    """Evidence that may be sent, newest first, and how many items were dropped.

    Newest first because that is the order the char budget spends in: the row a
    worker just appended is the one worth a request, and the oldest reachable
    item is the one a tight budget should lose. Sorting here is also what makes
    :func:`build_state`'s truncation land on the oldest item rather than on the
    most useful one.
    """
    screened: list[dict[str, Any]] = []
    dropped = 0
    for raw in list(items or [])[:MAX_EVIDENCE_ITEMS]:
        if not isinstance(raw, Mapping):
            dropped += 1
            continue
        item = evidence_item(
            str(raw.get("source", "") or ""),
            str(raw.get("kind", "") or ""),
            raw.get("age_s", 0.0),
            str(raw.get("text", "") or ""),
        )
        if item is None:
            dropped += 1
            continue
        screened.append(item)
    screened.sort(key=lambda row: row["age_s"])
    return screened, dropped


def build_state(
    instruction: str,
    *,
    wake_when: str = "",
    quiet_when: str = "",
    evidence: Sequence[Mapping[str, Any]] | None = None,
    last_verdict: Mapping[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The state sent to the judge, inside :data:`MAX_STATE_CHARS`.

    Oldest evidence is dropped FIRST, one item at a time, until the rendered
    state fits. The measurement uses the same ``json.dumps`` call the gate's own
    scan does, so what is measured is what the wire carries -- a length taken off
    the dict's ``repr`` would pass a budget the request then exceeds.

    ``last_verdict`` is carried so a judge can see it already passed on this
    evidence once, which is what lets it tell "still nothing" from "the same thing
    again".

    *trace* receives the counts the log row needs (``evidence_items``,
    ``evidence_chars``, ``dropped``, ``state_chars``), so the caller never
    reconstructs them from the state it just built.
    """
    screened, dropped = screen_evidence(evidence)
    loop: dict[str, Any] = {"instruction": _clip(instruction, MAX_INSTRUCTION_CHARS)}
    wake = _clip(wake_when, MAX_CRITERION_CHARS)
    quiet = _clip(quiet_when, MAX_CRITERION_CHARS)
    if wake:
        loop["wake_when"] = wake
    if quiet:
        loop["quiet_when"] = quiet

    def assemble(rows: list[dict[str, Any]]) -> dict[str, Any]:
        built: dict[str, Any] = {"loop": loop, "since_last_tick": rows}
        if last_verdict:
            built["last_verdict"] = dict(last_verdict)
        return built

    rows = list(screened)
    state = assemble(rows)
    # Drop from the TAIL, which ``screen_evidence`` ordered as the oldest. The
    # loop instruction and ``last_verdict`` are never dropped: they are already
    # bounded, and a judge without the owner's instruction cannot answer the one
    # question that asks about the owner's intent.
    while rows and _rendered_len(state) > MAX_STATE_CHARS:
        rows.pop()
        dropped += 1
        state = assemble(rows)
    if trace is not None:
        trace["evidence_items"] = len(rows)
        trace["evidence_chars"] = sum(len(row["text"]) for row in rows)
        trace["dropped"] = dropped
        trace["state_chars"] = _rendered_len(state)
    return state


def map_answers(answers: Answers | None) -> irq.Verdict:
    """The judge's answer as a driver instruction. Pure, and the whole mapping.

    Precedence, and why it is this order:

    1. no answer at all -> ``FALLBACK``. The tick fires as the ungated timer
       would, which is every failure path's destination.
    2. ``finished`` / ``broken`` at or above :data:`TERMINAL_MIN_P` ->
       ``TERMINAL``. Checked before the confidence floor because its own bar is
       higher: a low-confidence ``finished`` cannot pass here and correctly falls
       through to a wake.
    3. ``outcome`` below :data:`OUTCOME_MIN_P` -> ``WAKE``. An unsure judge hands
       the call to the main session rather than guessing quiet.
    4. ``needs_owner`` answering ``wake`` at or above :data:`NEEDS_OWNER_MIN_P`, or
       an outcome in :data:`ACTION_OUTCOMES` -> ``WAKE``.
    5. an outcome in :data:`QUIET_OUTCOMES` -> ``QUIET``.
    6. anything left -> ``WAKE``. The tail is a WAKE, not a quiet: the only thing
       that reaches it is a ``finished`` or ``broken`` answer too weak to end the
       watch, and staying silent about a work item the judge believes is over is
       the one failure this design will not accept.

    ``urgency`` deliberately changes nothing here. It is recorded on the log row
    for calibration, because the thresholds above are meant to be tuned from
    logged answers rather than from prompt text, and a rule that consumed it
    before there is a curve to read would be a guess wearing a constant's name.
    """
    outcome = _answer(answers, Q_OUTCOME)
    needs_owner = _answer(answers, Q_NEEDS_OWNER)
    if outcome is None or needs_owner is None:
        return irq.Verdict(irq.Outcome.FALLBACK, body="wake judge returned no usable answer")
    value = outcome.value
    if not isinstance(value, str) or value not in OUTCOME_OPTIONS:
        return irq.Verdict(irq.Outcome.FALLBACK, body="wake judge returned an unknown outcome")
    if value in TERMINAL_OUTCOMES and outcome.p >= TERMINAL_MIN_P:
        return irq.Verdict(
            irq.Outcome.TERMINAL,
            body=f"wake judge: the watched work is {value} (p={outcome.p:.2f})",
            keys=(value,),
        )
    if outcome.p < OUTCOME_MIN_P:
        return irq.Verdict(
            irq.Outcome.WAKE,
            body=f"wake judge was unsure (outcome {value} p={outcome.p:.2f}); waking the session",
        )
    if needs_owner.value == NEEDS_OWNER_WAKE and needs_owner.p >= NEEDS_OWNER_MIN_P:
        return irq.Verdict(
            irq.Outcome.WAKE,
            body=f"wake judge: the owner needs to act (p={needs_owner.p:.2f}, outcome {value})",
        )
    if value in ACTION_OUTCOMES:
        return irq.Verdict(
            irq.Outcome.WAKE,
            body=f"wake judge: the watched work is {value} (p={outcome.p:.2f})",
        )
    if value in QUIET_OUTCOMES:
        return irq.Verdict(
            irq.Outcome.QUIET,
            body=f"wake judge: quiet (outcome {value} p={outcome.p:.2f})",
        )
    # A terminal-family answer that missed TERMINAL_MIN_P lands here. It is not
    # sure enough to stop the watch and far too important to sit on.
    return irq.Verdict(
        irq.Outcome.WAKE,
        body=f"wake judge: {value} below the bar to end the watch (p={outcome.p:.2f})",
    )


async def judge_tick(
    instruction: str,
    *,
    wake_when: str = "",
    quiet_when: str = "",
    evidence: Sequence[Mapping[str, Any]] | None = None,
    last_verdict: Mapping[str, Any] | None = None,
    session_key: str | None = None,
    extra: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
) -> irq.Verdict:
    """Ask the judge about one tick. Never raises; ``FALLBACK`` on every failure.

    Runs ON the event loop, unlike ``skills.select``'s executor-thread adapter:
    the caller is ``AutoNudgeService``, which is already a coroutine, so there is
    no cross-thread future to wait on and no loop of its own to starve. The
    provider budget is the gate's own (``decisions.provider.timeout_ms``); this
    adds no second wait, because a tick that takes a moment longer costs nothing
    -- there is no message on a critical path behind it.

    *trace* receives ``answers`` and ``evidence_items``, which is what
    :func:`notice_line` needs to render the one-line transcript notice. An
    out-parameter rather than a second return value: every caller wants the
    verdict, and only the one that renders a notice wants the rest. It is filled
    on EVERY path that reached the provider, so a caller rendering a notice for a
    FALLBACK gets the evidence count even though there are no answers.
    """
    bounds: dict[str, Any] = {}
    try:
        state = build_state(
            instruction,
            wake_when=wake_when,
            quiet_when=quiet_when,
            evidence=evidence,
            last_verdict=last_verdict,
            trace=bounds,
        )
    except Exception:
        logger.debug("nudge.wake: could not assemble state", exc_info=True)
        return irq.Verdict(irq.Outcome.FALLBACK, body="wake judge could not assemble its evidence")
    if not state.get("since_last_tick"):
        # Nothing survived the scrub and the budget, so there is nothing to judge.
        # This is NOT quiet: "we could not look" and "we looked and it was calm"
        # are different facts, and only the second may cost a turn.
        return irq.Verdict(irq.Outcome.FALLBACK, body="wake judge had no evidence it could send")
    row: dict[str, Any] = dict(extra or {})
    row.update(
        {
            "evidence_items": bounds.get("evidence_items"),
            "evidence_chars": bounds.get("evidence_chars"),
            "dropped": bounds.get("dropped"),
            "state_chars": bounds.get("state_chars"),
        }
    )
    try:
        answers = await core.decide(
            POINT,
            state,
            build_questions(wake_when, quiet_when),
            session_key=session_key,
            extra=row,
        )
    except Exception:
        # ``decide`` returns None rather than raising, so this is belt and braces
        # for a config object whose attribute reads misbehave. Cancellation is not
        # caught: it is the caller going away, not a decision failure.
        logger.debug("nudge.wake: the decision call failed", exc_info=True)
        return irq.Verdict(irq.Outcome.FALLBACK, body="wake judge could not be reached")
    if trace is not None:
        trace["answers"] = answers
        trace["evidence_items"] = bounds.get("evidence_items", 0)
    return map_answers(answers)


def notice_line(verdict: irq.Verdict, answers: Answers | None, evidence_items: int) -> str:
    """One transcript line saying what the judge decided, and on how much.

    Rendered for a human reading the tab, so a tick that spent no turn still
    leaves a trace of WHY. Probabilities, not prose: the numbers are what a
    reader needs to tell a confident quiet from a lucky one.
    """
    parts = [f"Wake judge \u00b7 {verdict.outcome.value}"]
    readings = []
    for question_id in (Q_NEEDS_OWNER, Q_OUTCOME, Q_URGENCY):
        answer = _answer(answers, question_id)
        if answer is None:
            continue
        readings.append(f"{question_id} {answer.value} {answer.p:.2f}")
    if readings:
        parts.append(", ".join(readings))
    parts.append(f"{evidence_items} evidence item(s)")
    return " \u00b7 ".join(parts)


def _answer(answers: Answers | None, question_id: str) -> Answer | None:
    """One validated answer, or ``None``. The gate checked the domain already."""
    if not isinstance(answers, dict):
        return None
    answer = answers.get(question_id)
    return answer if isinstance(answer, Answer) else None


def _clip(value: object, limit: int) -> str:
    """*value* as a string of at most *limit* characters. A non-string folds to empty."""
    return value[:limit] if isinstance(value, str) else ""


def _age(raw: object) -> float:
    """A non-negative finite age in seconds, or 0.0. A bool is not an age."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    value = float(raw)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, value)


def _rendered_len(state: Mapping[str, Any]) -> int:
    """Characters *state* occupies on the wire, rendered the way the scan renders it."""
    try:
        return len(json.dumps(state, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(repr(state))


def now() -> float:
    """Wall clock, as one seam tests can move without patching ``time`` globally."""
    return time.time()
