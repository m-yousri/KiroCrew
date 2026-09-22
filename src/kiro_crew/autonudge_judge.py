"""Evidence for the ``nudge.wake`` judge, and the one call an auto-nudge tick makes.

The point module (``decisions.points.nudge_wake``) owns the questions, the bounded
state and the mapping. This module owns WHERE the evidence comes from: a watched
session's new transcript rows, a watched pull request's observation, and (later)
work-ledger events. It sits beside ``autonudge`` rather than under ``decisions``
because collecting is a gateway concern -- it reads live slot state -- while the
point stays a pure adapter that a test can drive with literals.

Why the reads arrive as CALLABLES
---------------------------------
``AutoNudgeService`` holds no ``DashboardState``: it is constructed with a data
directory and two callbacks, and the gateway injects closures that reach the rest
(``on_fire``, ``on_monitor_tick``, ``owner_session_id``). The session read is the
same shape, and it has to be, because authorizing it needs state the service
cannot see. So :func:`collect_evidence` takes readers and this module imports no
dashboard module at all -- which is also what lets every function here be tested
without a gateway.

Creator-only, enforced by reuse
-------------------------------
A ``session`` target is read through ``session_control.read_messages``, which
authorizes with ``authorize_target`` before returning a row: deny-by-default, and
SEL-audited on refusal. The judge therefore cannot read a session its owning loop
could not read by hand, and a target that refuses is DROPPED and counted rather
than fetched. Nothing here re-implements that check, because a second copy of an
authorization rule is a second place for it to be wrong.

Assistant rows only
-------------------
``decisions.points.HISTORY_ROLES`` excludes tool output from every other point on
the grounds that it is the largest and least selective text in a transcript and
routinely quotes files nobody mentioned. The same reasoning holds here, and the
evidence a watcher actually needs -- a worker's ``RULING:`` or ``BLOCKED:`` line,
a reviewer's verdict -- is an assistant row. Tool rows are skipped.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from kiro_crew.decisions.points import nudge_wake as point

logger = logging.getLogger(__name__)

#: A dashboard chat-slot key, the spelling a `judge.targets` entry uses for a
#: session. Anchored and bounded: the value is owner-supplied and becomes a
#: `read_messages` target, so it is matched rather than trusted.
SESSION_TARGET_RE = re.compile(r"\Achat-[A-Za-z0-9][A-Za-z0-9-]{0,127}\Z")

#: The same shape, found INSIDE prose, so a loop whose instruction names the
#: sessions it watches needs no second spelling of them in ``judge.targets``. The
#: anchored pattern above still screens every hit, so this only decides where to
#: look, never what counts.
SESSION_TARGET_IN_TEXT_RE = re.compile(r"\bchat-[A-Za-z0-9][A-Za-z0-9-]{0,127}\b")

#: How many transcript rows one target contributes to a single tick. The char
#: budget in the point is what bounds egress; this bounds the READ, so a session
#: that produced a hundred rows between ticks cannot turn one decision into a
#: whole-transcript scan.
MAX_ROWS_PER_TARGET = 12

#: How many targets one loop may name. Bounds the number of authorizations and
#: reads a single tick performs.
MAX_TARGETS = 8

#: Transcript roles whose text is evidence. Assistant only -- see the module
#: docstring for why tool rows are excluded.
EVIDENCE_ROLES = frozenset({"assistant"})


def parse_targets(spec: Mapping[str, Any] | None, message: str = "") -> list[str]:
    """The targets to collect from: the spec's own list, else what *message* names.

    An explicit ``targets`` list adds or narrows; without one the loop's
    instruction is read the way the PR probe already reads it, so arming a judge
    on a loop that already names a pull request needs no second spelling of the
    subject.

    Only recognised shapes survive: a ``chat-*`` key matching
    :data:`SESSION_TARGET_RE`, or a string a pull-request target can be inferred
    from. Anything else is dropped rather than passed to a reader, because these
    strings come from the owner's tool call.
    """
    raw: list[str] = []
    if isinstance(spec, Mapping):
        listed = spec.get("targets")
        if isinstance(listed, (list, tuple)):
            raw = [str(item) for item in listed if isinstance(item, str)]
    out: list[str] = []
    for item in raw:
        value = item.strip()
        if not value or value in out:
            continue
        if SESSION_TARGET_RE.fullmatch(value) or _pr_subject(value):
            out.append(value)
        if len(out) >= MAX_TARGETS:
            break
    if out:
        return out
    # Nothing explicit, so read the instruction the way the design says: the targets
    # default to what the MESSAGE names. Both shapes, not just pull requests -- a
    # conductor's instruction names the sessions it patrols, and requiring those to be
    # repeated in ``targets`` would mean a brief that looks armed and watches nothing.
    for match in SESSION_TARGET_IN_TEXT_RE.findall(message or ""):
        if match not in out and SESSION_TARGET_RE.fullmatch(match):
            out.append(match)
        if len(out) >= MAX_TARGETS:
            break
    if _pr_subject(message):
        # The whole message, because pull-request inference reads the original
        # spelling (it carries the host a URL-armed watch is entitled to) rather
        # than a token lifted out of it.
        stripped = message.strip()
        if stripped not in out:
            out.append(stripped)
    return out


def is_session_target(value: str) -> bool:
    """Whether *value* names a dashboard chat slot rather than a pull request."""
    return SESSION_TARGET_RE.fullmatch(value or "") is not None


def session_evidence(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    *,
    now_ts: float | None = None,
) -> list[dict[str, Any]]:
    """New assistant rows from one watched session, newest first, as evidence items.

    Each row's ``ts`` becomes the item's age, so the point can order and drop by
    recency. A row without a usable timestamp reads as age 0 -- treating it as the
    newest thing available, which keeps it in the request under a tight budget
    rather than silently dropping evidence because a row lacked a field.
    """
    clock = point.now() if now_ts is None else now_ts
    items: list[dict[str, Any]] = []
    for row in list(rows)[-MAX_ROWS_PER_TARGET:]:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("role", "") or "") not in EVIDENCE_ROLES:
            continue
        text = row.get("content", "")
        if not isinstance(text, str) or not text.strip():
            continue
        items.append(
            {
                "source": f"session:{target}",
                "kind": point.KIND_TRANSCRIPT_TAIL,
                "age_s": _age_from_ts(row.get("ts"), clock),
                "text": text,
            }
        )
    return items


def pr_evidence(
    observation: Mapping[str, Any] | None,
    target: str,
    *,
    now_ts: float | None = None,
) -> list[dict[str, Any]]:
    """One watched pull request's probe reading, plus any new comment bodies.

    The probe's own observation is carried as ``kind=probe`` rather than
    re-derived: it is the typed half of this decision and it has already run this
    tick, so asking GitHub a second question would cost a subprocess to learn what
    the caller already holds. Comment bodies are the half the probe cannot type,
    which is the whole reason this point exists.
    """
    if not isinstance(observation, Mapping):
        return []
    clock = point.now() if now_ts is None else now_ts
    items: list[dict[str, Any]] = []
    summary = observation.get("summary")
    if isinstance(summary, str) and summary.strip():
        items.append(
            {
                "source": f"pr:{target}",
                "kind": point.KIND_PR_CHECKS,
                "age_s": _age_from_ts(observation.get("observed_at"), clock),
                "text": summary,
            }
        )
    comments = observation.get("comments")
    if isinstance(comments, (list, tuple)):
        for comment in list(comments)[:MAX_ROWS_PER_TARGET]:
            if not isinstance(comment, Mapping):
                continue
            body = comment.get("body")
            if not isinstance(body, str) or not body.strip():
                continue
            author = str(comment.get("author", "") or "unknown")
            items.append(
                {
                    "source": f"pr:{target}",
                    "kind": point.KIND_PR_COMMENT,
                    "age_s": _age_from_ts(comment.get("ts"), clock),
                    "text": f"{author}: {body}",
                }
            )
    return items


async def collect_evidence(
    targets: Sequence[str],
    *,
    read_session: (
        Callable[[str, int], Awaitable[tuple[Sequence[Mapping[str, Any]], int]]] | None
    ) = None,
    read_pr: Callable[[str], Awaitable[Mapping[str, Any] | None]] | None = None,
    cursors: dict[str, int] | None = None,
    now_ts: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Evidence for one tick, and how many targets were dropped. Never raises.

    *read_session* is given a target and its cursor and returns ``(rows,
    next_cursor)``; *read_pr* is given a target and returns the probe observation.
    Either may be ``None``, which simply means that collector is unavailable on
    this build or this loop -- not an error, because a judge watching only
    sessions needs no pull-request reader.

    A reader that RAISES counts as a dropped target rather than a failed tick. The
    refusal a creator-only check produces arrives exactly that way, which is what
    makes "a target the owner may not read is dropped and noted" true by
    construction instead of by a second check here.

    *cursors* is updated in place for the targets that were read, so the next tick
    sees only what arrived in between. It is only advanced on a SUCCESSFUL read: a
    target that refused or raised keeps its old cursor, so a transient failure
    cannot silently skip the rows it would have returned.
    """
    clock = point.now() if now_ts is None else now_ts
    evidence: list[dict[str, Any]] = []
    dropped = 0
    for target in list(targets)[:MAX_TARGETS]:
        try:
            if is_session_target(target):
                if read_session is None:
                    dropped += 1
                    continue
                rows, next_cursor = await read_session(target, int((cursors or {}).get(target, 0)))
                evidence.extend(session_evidence(rows, target, now_ts=clock))
                if cursors is not None and isinstance(next_cursor, int) and next_cursor >= 0:
                    cursors[target] = next_cursor
            else:
                if read_pr is None:
                    dropped += 1
                    continue
                evidence.extend(pr_evidence(await read_pr(target), target, now_ts=clock))
        except Exception:
            # Includes the creator-only refusal. The class is not logged at
            # warning: a loop naming a session it may not read is an owner
            # mistake that would otherwise repeat every interval.
            dropped += 1
            logger.debug("nudge.wake: dropping a target this loop could not read", exc_info=True)
    return evidence, dropped


def spec_of(loop: Any) -> dict[str, Any]:
    """One loop's stored judge spec as a mapping, or ``{}``. Never raises.

    ``{}`` is "no judge on this loop", which is what a record written before the
    field existed decodes to and what an unreadable value resolves to -- the tick
    then behaves exactly as it does today.
    """
    try:
        raw = getattr(loop, "judge", None)
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def criteria_of(spec: Mapping[str, Any] | None) -> tuple[str, str]:
    """The owner's ``(wake_when, quiet_when)``, each clipped, each possibly empty."""
    if not isinstance(spec, Mapping):
        return "", ""
    wake = spec.get("wake_when", "")
    quiet = spec.get("quiet_when", "")
    return (
        wake[: point.MAX_CRITERION_CHARS] if isinstance(wake, str) else "",
        quiet[: point.MAX_CRITERION_CHARS] if isinstance(quiet, str) else "",
    )


def verdict_record(verdict: Any, evidence_items: int) -> dict[str, Any]:
    """The previous-verdict summary carried into the NEXT tick's state.

    Deliberately small and text-free: the outcome, how much it was based on, and
    when. A judge seeing this knows it already passed on comparable evidence
    without being handed that evidence a second time.
    """
    try:
        outcome = verdict.outcome.value
    except Exception:
        outcome = "unknown"
    return {"outcome": outcome, "evidence_items": int(evidence_items), "at": time.time()}


def _pr_subject(value: str) -> bool:
    """Whether a pull-request target can be inferred from *value*. Never raises."""
    if not value or not value.strip():
        return False
    try:
        from kiro_crew.probes import targets as _targets

        return _targets.infer(value) is not None
    except Exception:
        logger.debug("nudge.wake: target inference unavailable", exc_info=True)
        return False


def _age_from_ts(raw: object, clock: float) -> float:
    """Seconds between *raw* and *clock*, or 0.0 when *raw* is not a usable time."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    age = clock - float(raw)
    return age if age > 0 else 0.0
