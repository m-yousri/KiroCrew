"""The wake judge: its mapping table, its bounds, its collectors and its tick.

No Jev key is spent anywhere here. The provider is stubbed at ``decisions.decide``,
which is the seam the point calls, so these tests exercise the real state builder,
the real mapping and the real tick without a network request.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any
from unittest.mock import patch

import pytest

from kiro_crew import autonudge_judge as judge
from kiro_crew.autonudge import (
    _JUDGE_QUIET_STREAK_FLOOR_DEFAULT,
    _MAX_QUIET_STREAK,
    AutoNudgeService,
    NudgeLoop,
)
from kiro_crew.decisions.points import nudge_wake as point
from kiro_crew.decisions.types import Answer
from kiro_crew.irq import Outcome
from kiro_crew.validation import ValidationError, validate_judge_spec


def answers(
    owner: str = point.NEEDS_OWNER_QUIET,
    owner_p: float = 0.9,
    outcome: str = point.OUTCOME_PROGRESS_ONLY,
    outcome_p: float = 0.9,
    urgency: str = point.URGENCY_NONE,
) -> dict[str, Answer]:
    """One complete, in-domain answer set."""
    return {
        point.Q_NEEDS_OWNER: Answer(point.Q_NEEDS_OWNER, owner, owner_p),
        point.Q_OUTCOME: Answer(point.Q_OUTCOME, outcome, outcome_p),
        point.Q_URGENCY: Answer(point.Q_URGENCY, urgency, 0.9),
    }


class TestMappingTable:
    """Every branch of :func:`nudge_wake.map_answers`, including the failure ones."""

    def test_no_answer_is_fallback(self) -> None:
        assert point.map_answers(None).outcome is Outcome.FALLBACK

    def test_missing_question_is_fallback(self) -> None:
        partial = answers()
        del partial[point.Q_OUTCOME]
        assert point.map_answers(partial).outcome is Outcome.FALLBACK

    def test_out_of_domain_outcome_is_fallback(self) -> None:
        assert point.map_answers(answers(outcome="banana")).outcome is Outcome.FALLBACK

    @pytest.mark.parametrize("value", sorted(point.TERMINAL_OUTCOMES))
    def test_terminal_at_or_above_bar(self, value: str) -> None:
        verdict = point.map_answers(answers(outcome=value, outcome_p=point.TERMINAL_MIN_P))
        assert verdict.outcome is Outcome.TERMINAL
        assert verdict.keys == (value,)

    @pytest.mark.parametrize("value", sorted(point.TERMINAL_OUTCOMES))
    def test_terminal_below_bar_wakes_rather_than_quieting(self, value: str) -> None:
        """A judge that thinks the work is over but is unsure must not go silent.

        This is the case a catch-all ``otherwise -> QUIET`` gets wrong: the answer
        clears the confidence floor, names neither an action outcome nor a quiet one,
        and would fall through to silence about the one event the owner most needs.
        """
        verdict = point.map_answers(answers(outcome=value, outcome_p=point.TERMINAL_MIN_P - 0.01))
        assert verdict.outcome is Outcome.WAKE

    def test_low_confidence_wakes(self) -> None:
        verdict = point.map_answers(answers(outcome_p=point.OUTCOME_MIN_P - 0.01))
        assert verdict.outcome is Outcome.WAKE

    def test_needs_owner_wake_at_bar(self) -> None:
        verdict = point.map_answers(
            answers(owner=point.NEEDS_OWNER_WAKE, owner_p=point.NEEDS_OWNER_MIN_P)
        )
        assert verdict.outcome is Outcome.WAKE

    def test_needs_owner_wake_below_bar_does_not_fire_on_that_rule(self) -> None:
        """The threshold is applied literally, as the design specifies.

        Only reachable from a provider that returns a non-argmax choice: with two
        options the chosen one carries at least half the mass.
        """
        verdict = point.map_answers(
            answers(owner=point.NEEDS_OWNER_WAKE, owner_p=point.NEEDS_OWNER_MIN_P - 0.01)
        )
        assert verdict.outcome is Outcome.QUIET

    @pytest.mark.parametrize("value", sorted(point.ACTION_OUTCOMES))
    def test_action_outcomes_wake_even_when_owner_says_quiet(self, value: str) -> None:
        verdict = point.map_answers(answers(owner=point.NEEDS_OWNER_QUIET, outcome=value))
        assert verdict.outcome is Outcome.WAKE

    @pytest.mark.parametrize("value", sorted(point.QUIET_OUTCOMES))
    def test_quiet_outcomes_are_the_only_quiet(self, value: str) -> None:
        assert point.map_answers(answers(outcome=value)).outcome is Outcome.QUIET

    def test_quiet_is_an_allowlist(self) -> None:
        """No outcome outside :data:`QUIET_OUTCOMES` can produce silence."""
        for value in point.OUTCOME_OPTIONS:
            if value in point.QUIET_OUTCOMES:
                continue
            for probability in (0.41, 0.5, 0.59, 0.75, 1.0):
                verdict = point.map_answers(answers(outcome=value, outcome_p=probability))
                assert verdict.outcome is not Outcome.QUIET, (value, probability)

    def test_terminal_bar_is_above_the_confidence_floor(self) -> None:
        """The precedence in the mapping depends on this, so it is asserted."""
        assert point.TERMINAL_MIN_P > point.OUTCOME_MIN_P


class TestQuestions:
    """The three questions, their domains, and where the owner's words go."""

    def test_domains(self) -> None:
        built = {q.id: q.options for q in point.build_questions()}
        assert built[point.Q_NEEDS_OWNER] == [point.NEEDS_OWNER_WAKE, point.NEEDS_OWNER_QUIET]
        assert built[point.Q_OUTCOME] == list(point.OUTCOME_OPTIONS)
        assert built[point.Q_URGENCY] == list(point.URGENCY_OPTIONS)

    def test_criteria_ride_in_the_prompt_labelled_by_option(self) -> None:
        questions = point.build_questions("a line starts with RULING", "workers say WORKING")
        prompt = questions[0].prompt
        assert "RULING" in prompt and "WORKING" in prompt
        assert point.NEEDS_OWNER_WAKE in prompt and point.NEEDS_OWNER_QUIET in prompt

    def test_criteria_are_clipped(self) -> None:
        questions = point.build_questions("w" * 5_000, "q" * 5_000)
        assert len(questions[0].prompt) < 2 * point.MAX_CRITERION_CHARS + 500

    def test_evidence_never_enters_a_question(self) -> None:
        """Evidence is data and lives in ``state``; a question is an instruction."""
        questions = point.build_questions("wake on RULING", "quiet on WORKING")
        rendered = " ".join(q.prompt + " ".join(q.options) for q in questions)
        assert "RULING" in rendered  # the owner's criterion, which does belong
        assert "leaked-evidence-marker" not in rendered


class TestStateBounds:
    """The request's ceiling, the per-item clip, and what the scrub drops."""

    def test_state_fits_the_ceiling_and_drops_oldest_first(self) -> None:
        evidence = [
            {
                "source": f"session:chat-{i}",
                "kind": point.KIND_TRANSCRIPT_TAIL,
                "age_s": float(i),
                "text": "x" * 900,
            }
            for i in range(40)
        ]
        trace: dict[str, Any] = {}
        state = point.build_state("watch the workers", evidence=evidence, trace=trace)
        assert trace["state_chars"] <= point.MAX_STATE_CHARS
        ages = [row["age_s"] for row in state["since_last_tick"]]
        assert ages == sorted(ages), "newest first"
        assert ages and max(ages) < 39.0, "the oldest items were the ones dropped"
        assert trace["dropped"] > 0

    def test_per_item_clip(self) -> None:
        state = point.build_state(
            "w",
            evidence=[
                {
                    "source": "s",
                    "kind": point.KIND_TRANSCRIPT_TAIL,
                    "age_s": 1.0,
                    "text": "y" * 9_000,
                }
            ],
        )
        assert len(state["since_last_tick"][0]["text"]) == point.MAX_ITEM_CHARS

    def test_instruction_is_clipped(self) -> None:
        state = point.build_state("i" * 9_000)
        assert len(state["loop"]["instruction"]) == point.MAX_INSTRUCTION_CHARS

    def test_last_verdict_is_carried(self) -> None:
        state = point.build_state(
            "w",
            evidence=[
                {"source": "s", "kind": point.KIND_PROBE, "age_s": 1.0, "text": "checks pending"}
            ],
            last_verdict={"outcome": "quiet", "evidence_items": 2},
        )
        assert state["last_verdict"]["outcome"] == "quiet"

    def test_scrub_drops_an_item_carrying_a_credential(self) -> None:
        assert point.evidence_item("pr:x#1", point.KIND_PR_COMMENT, 1.0, "AKIA" + "A" * 16) is None

    def test_unknown_kind_is_dropped(self) -> None:
        assert point.evidence_item("s", "not-a-kind", 1.0, "hello") is None

    def test_scrub_drop_everything_leaves_no_evidence(self) -> None:
        screened, dropped = point.screen_evidence(
            [
                {
                    "source": "s",
                    "kind": point.KIND_PR_COMMENT,
                    "age_s": 1.0,
                    "text": "AKIA" + "B" * 16,
                }
            ]
        )
        assert screened == [] and dropped == 1


class TestJudgeTickFallsOpen:
    """Every failure path reaches FALLBACK, which fires the loop."""

    def test_no_evidence_is_fallback_not_quiet(self) -> None:
        verdict = asyncio.run(point.judge_tick("watch", evidence=[]))
        assert verdict.outcome is Outcome.FALLBACK

    def test_refused_decision_is_fallback(self) -> None:
        async def refuse(*args: Any, **kwargs: Any) -> None:
            return None

        evidence = [
            {"source": "s", "kind": point.KIND_PROBE, "age_s": 1.0, "text": "checks pending"}
        ]
        with patch("kiro_crew.decisions.decide", refuse):
            verdict = asyncio.run(point.judge_tick("watch", evidence=evidence))
        assert verdict.outcome is Outcome.FALLBACK

    def test_raising_provider_is_fallback(self) -> None:
        async def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("provider exploded")

        evidence = [
            {"source": "s", "kind": point.KIND_PROBE, "age_s": 1.0, "text": "checks pending"}
        ]
        with patch("kiro_crew.decisions.decide", boom):
            verdict = asyncio.run(point.judge_tick("watch", evidence=evidence))
        assert verdict.outcome is Outcome.FALLBACK


class TestCollectors:
    """What the collectors read, what they refuse, and what they remember."""

    def test_assistant_rows_only(self) -> None:
        rows = [
            {"role": "assistant", "content": "RULING: need a call", "ts": 100.0},
            {"role": "tool", "content": "a file nobody asked to send", "ts": 101.0},
            {"role": "user", "content": "typed by the owner", "ts": 102.0},
        ]
        items = judge.session_evidence(rows, "chat-2-2", now_ts=110.0)
        assert [i["text"] for i in items] == ["RULING: need a call"]

    def test_pr_observation_and_comments(self) -> None:
        items = judge.pr_evidence(
            {
                "summary": "1 failed",
                "observed_at": 100.0,
                "comments": [{"author": "codex", "body": "new finding", "ts": 99.0}],
            },
            "owner/name#1",
            now_ts=110.0,
        )
        kinds = [i["kind"] for i in items]
        assert point.KIND_PR_CHECKS in kinds and point.KIND_PR_COMMENT in kinds

    def test_refused_target_is_dropped_and_its_cursor_held(self) -> None:
        async def refuse(target: str, since: int) -> tuple[list[dict], int]:
            raise PermissionError("not the creator")

        cursors = {"chat-2-2": 7}
        items, dropped = asyncio.run(
            judge.collect_evidence(["chat-2-2"], read_session=refuse, cursors=cursors)
        )
        assert items == [] and dropped == 1
        assert cursors == {"chat-2-2": 7}, "a refusal must not advance the cursor"

    def test_cursor_advances_on_a_successful_read(self) -> None:
        async def read(target: str, since: int) -> tuple[list[dict], int]:
            return [{"role": "assistant", "content": "progress", "ts": 1.0}], 12

        cursors: dict[str, int] = {}
        asyncio.run(judge.collect_evidence(["chat-2-2"], read_session=read, cursors=cursors))
        assert cursors == {"chat-2-2": 12}

    def test_absent_reader_drops_rather_than_raising(self) -> None:
        items, dropped = asyncio.run(judge.collect_evidence(["chat-2-2"]))
        assert items == [] and dropped == 1

    def test_targets_from_the_spec_are_deduped_and_screened(self) -> None:
        targets = judge.parse_targets(
            {"targets": ["chat-2-2", "chat-2-2", "not a target at all"]}, ""
        )
        assert targets == ["chat-2-2"]

    def test_session_target_shape(self) -> None:
        assert judge.is_session_target("chat-1751-1790052364")
        assert not judge.is_session_target("../etc/passwd")
        assert not judge.is_session_target("")


class TestSchema:
    """What ``monitor_start`` / ``monitor_update`` accept as a brief."""

    def test_accepts_and_normalises(self) -> None:
        out = validate_judge_spec(
            {"targets": ["chat-2-2", "chat-2-2"], "wake_when": "RULING", "quiet_when": "WORKING"}
        )
        assert out["targets"] == ["chat-2-2"]
        assert out["wake_when"] == "RULING"

    def test_absent_and_empty_are_both_legal(self) -> None:
        assert validate_judge_spec(None) == {}
        assert validate_judge_spec({}) == {}

    @pytest.mark.parametrize(
        "bad",
        [
            "a string",
            5,
            {"wake_whn": "a typo nobody would notice"},
            {"targets": "chat-2-2"},
            {"targets": [1]},
            {"targets": ["chat-x"] * 9},
            {"targets": ["c" * 500]},
            {"wake_when": "x" * 501},
            {"quiet_when": 5},
        ],
    )
    def test_refuses(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            validate_judge_spec(bad)


def _service() -> AutoNudgeService:
    """A service with no disk and no evidence reader wired, for tick tests."""
    svc = AutoNudgeService.__new__(AutoNudgeService)
    svc._collect_judge_evidence = None
    svc._emit_judge_notice = None
    return svc


def _loop(brief: dict | None = None) -> NudgeLoop:
    loop = NudgeLoop(id="l1", slot_key="chat-1-1", message="watch chat-2-2")
    loop.judge = dict(brief or {})
    return loop


class TestTickLeavesEverythingAloneWhenItShould:
    """``None`` means the judge had no say, so the tick behaves exactly as today."""

    def test_no_brief(self) -> None:
        assert asyncio.run(_service()._judge_tick_is_quiet(_loop())) is None

    def test_brief_but_no_evidence_reader(self) -> None:
        assert asyncio.run(_service()._judge_tick_is_quiet(_loop({"wake_when": "x"}))) is None

    def test_scope_missing_stores_the_brief_and_ignores_it(self) -> None:
        """The scope is the on switch, so without it the loop is a plain timer."""
        svc = _service()

        async def never(loop: NudgeLoop) -> tuple[list[dict], int]:
            raise AssertionError("collected evidence with no scope granted")

        svc._collect_judge_evidence = never
        loop = _loop({"wake_when": "x"})
        with (
            patch.object(AutoNudgeService, "_judge_lane", lambda self: "jev"),
            patch("kiro_crew.decisions.is_enabled", lambda *a, **k: False),
        ):
            assert asyncio.run(svc._judge_tick_is_quiet(loop)) is None
        assert loop.judge == {"wake_when": "x"}, "the brief is kept for when the scope arrives"

    def test_no_lane_available(self) -> None:
        svc = _service()

        async def never(loop: NudgeLoop) -> tuple[list[dict], int]:
            raise AssertionError("collected evidence with no provider lane")

        svc._collect_judge_evidence = never
        with patch.object(AutoNudgeService, "_judge_lane", lambda self: ""):
            assert asyncio.run(svc._judge_tick_is_quiet(_loop({"wake_when": "x"}))) is None


class TestTickVerdicts:
    """QUIET spends no turn, everything else spends one, and the floor bounds QUIET."""

    @staticmethod
    def _armed(verdict_answers: dict[str, Answer] | None) -> tuple[AutoNudgeService, NudgeLoop]:
        svc = _service()

        async def collect(loop: NudgeLoop) -> tuple[list[dict], int]:
            return [
                {
                    "source": "session:chat-2-2",
                    "kind": point.KIND_TRANSCRIPT_TAIL,
                    "age_s": 1.0,
                    "text": "WORKING: still building",
                }
            ], 0

        svc._collect_judge_evidence = collect
        svc._persist_soon = lambda: None  # type: ignore[method-assign]
        return svc, _loop({"wake_when": "RULING", "quiet_when": "WORKING"})

    def _run(self, svc: AutoNudgeService, loop: NudgeLoop, ans: Any) -> Any:
        async def decide(*args: Any, **kwargs: Any) -> Any:
            return ans

        with (
            patch.object(AutoNudgeService, "_judge_lane", lambda self: "jev"),
            patch("kiro_crew.decisions.is_enabled", lambda *a, **k: True),
            patch("kiro_crew.decisions.decide", decide),
        ):
            return asyncio.run(svc._judge_tick_is_quiet(loop))

    def test_quiet_spends_no_turn(self) -> None:
        svc, loop = self._armed(None)
        assert self._run(svc, loop, answers()) is True
        assert loop.judge_quiet_streak == 1
        assert loop.judge_last_verdict["outcome"] == "quiet"

    def test_wake_spends_a_turn_and_clears_the_streak(self) -> None:
        svc, loop = self._armed(None)
        loop.judge_quiet_streak = 4
        assert self._run(svc, loop, answers(outcome=point.OUTCOME_NEEDS_ACTION)) is False
        assert loop.judge_quiet_streak == 0

    def test_terminal_spends_a_turn(self) -> None:
        svc, loop = self._armed(None)
        verdict = self._run(svc, loop, answers(outcome=point.OUTCOME_FINISHED, outcome_p=0.95))
        assert verdict is False
        assert loop.judge_last_verdict["outcome"] == "terminal"

    def test_terminal_fires_once_then_stays_quiet(self) -> None:
        """The owner is told the work is over once, not once per interval."""
        svc, loop = self._armed(None)
        finished = answers(outcome=point.OUTCOME_FINISHED, outcome_p=0.95)

        assert self._run(svc, loop, finished) is False, "first terminal delivers"
        assert loop.judge_terminal_fired is True

        for _ in range(3):
            assert self._run(svc, loop, finished) is True, "repeats spend no turn"

    def test_a_repeated_terminal_still_counts_toward_the_floor(self) -> None:
        """A wrong terminal cannot silence a loop forever."""
        svc, loop = self._armed(None)
        finished = answers(outcome=point.OUTCOME_FINISHED, outcome_p=0.95)
        self._run(svc, loop, finished)
        floor = svc._judge_quiet_streak_floor()
        fired = 0
        for _ in range(floor + 2):
            if self._run(svc, loop, finished) is False:
                fired += 1
        assert fired >= 1, "the streak floor delivers a turn despite the repeated terminal"

    def test_a_live_verdict_rearms_the_terminal(self) -> None:
        """A subject that comes back to life can produce a fresh terminal."""
        svc, loop = self._armed(None)
        finished = answers(outcome=point.OUTCOME_FINISHED, outcome_p=0.95)
        self._run(svc, loop, finished)
        assert loop.judge_terminal_fired is True

        self._run(svc, loop, answers(outcome=point.OUTCOME_NEEDS_ACTION))
        assert loop.judge_terminal_fired is False

        assert self._run(svc, loop, finished) is False, "a fresh terminal delivers again"

    def test_refused_provider_spends_a_turn(self) -> None:
        svc, loop = self._armed(None)
        assert self._run(svc, loop, None) is False

    def test_streak_floor_fires_and_resets(self) -> None:
        svc, loop = self._armed(None)
        floor = svc._judge_quiet_streak_floor()
        loop.judge_quiet_streak = floor - 1
        assert self._run(svc, loop, answers()) is False, "the floor delivers a turn"
        assert loop.judge_quiet_streak == 0

    def test_collector_failure_spends_a_turn(self) -> None:
        svc, loop = self._armed(None)

        async def boom(loop_: NudgeLoop) -> tuple[list[dict], int]:
            raise RuntimeError("slot read exploded")

        svc._collect_judge_evidence = boom
        assert self._run(svc, loop, answers()) is False


class TestTranscriptNotice:
    """A verdict that spends no turn still leaves one line on the session."""

    @staticmethod
    def _armed_with_sink() -> tuple[AutoNudgeService, NudgeLoop, list[str]]:
        svc = _service()
        sink: list[str] = []

        async def collect(loop: NudgeLoop) -> tuple[list[dict], int]:
            return [
                {
                    "source": "session:chat-2-2",
                    "kind": point.KIND_TRANSCRIPT_TAIL,
                    "age_s": 1.0,
                    "text": "WORKING: still building",
                }
            ], 0

        async def emit(loop: NudgeLoop, line: str) -> None:
            sink.append(line)

        svc._collect_judge_evidence = collect
        svc._emit_judge_notice = emit
        svc._persist_soon = lambda: None  # type: ignore[method-assign]
        return svc, _loop({"wake_when": "RULING", "quiet_when": "WORKING"}), sink

    def _run(self, svc: AutoNudgeService, loop: NudgeLoop, ans: Any) -> Any:
        async def decide(*args: Any, **kwargs: Any) -> Any:
            return ans

        with (
            patch.object(AutoNudgeService, "_judge_lane", lambda self: "jev"),
            patch("kiro_crew.decisions.is_enabled", lambda *a, **k: True),
            patch("kiro_crew.decisions.decide", decide),
        ):
            return asyncio.run(svc._judge_tick_is_quiet(loop))

    def test_quiet_verdict_emits_one_line(self) -> None:
        svc, loop, sink = self._armed_with_sink()
        assert self._run(svc, loop, answers()) is True, "quiet spends no turn"
        assert len(sink) == 1
        line = sink[0]
        assert "quiet" in line
        assert point.Q_NEEDS_OWNER in line and point.Q_OUTCOME in line
        assert "0.9" in line, "the probabilities are what tell a confident quiet apart"

    def test_wake_verdict_also_emits(self) -> None:
        svc, loop, sink = self._armed_with_sink()
        self._run(svc, loop, answers(outcome=point.OUTCOME_NEEDS_ACTION))
        assert len(sink) == 1 and "wake" in sink[0]

    def test_fallback_emits_with_no_readings(self) -> None:
        svc, loop, sink = self._armed_with_sink()
        self._run(svc, loop, None)
        assert len(sink) == 1 and "fallback" in sink[0]

    def test_notice_carries_no_evidence_text(self) -> None:
        """The state stays in the request; the transcript gets the verdict."""
        svc, loop, sink = self._armed_with_sink()
        self._run(svc, loop, answers())
        assert "still building" not in sink[0]

    def test_a_broken_renderer_does_not_cost_the_verdict(self) -> None:
        svc, loop, _sink = self._armed_with_sink()

        async def boom(loop_: NudgeLoop, line: str) -> None:
            raise RuntimeError("renderer exploded")

        svc._emit_judge_notice = boom
        assert self._run(svc, loop, answers()) is True

    def test_a_notice_row_is_never_evidence(self) -> None:
        """A judge must not read its own previous notice back as new evidence.

        The collector admits assistant rows only, so the exclusion is structural
        rather than a name check against this feature's own output.
        """
        rows = [
            {"role": "notice", "content": "Wake judge - quiet - 2 evidence item(s)", "ts": 100.0},
            {"role": "assistant", "content": "RULING: need a call", "ts": 101.0},
        ]
        items = judge.session_evidence(rows, "chat-2-2", now_ts=110.0)
        assert [i["text"] for i in items] == ["RULING: need a call"]


class TestStreakFloorConfig:
    """The floor's default and its hard ceiling."""

    def test_default_equals_the_shipped_probe_floor(self) -> None:
        assert _JUDGE_QUIET_STREAK_FLOOR_DEFAULT == _MAX_QUIET_STREAK

    def test_unreadable_config_is_the_default(self) -> None:
        assert _service()._judge_quiet_streak_floor() == _JUDGE_QUIET_STREAK_FLOOR_DEFAULT


class TestArmPath:
    """A brief survives arming, revision, clearing and a store reload."""

    def test_round_trip(self, tmp_path: pathlib.Path) -> None:
        async def main() -> None:
            base = tmp_path / "nudges"
            base.mkdir()
            svc = AutoNudgeService(base_dir=base)
            brief = {"targets": ["chat-2-2"], "wake_when": "RULING"}
            loop = await svc.add("chat-1-1", "watch chat-2-2", idle_secs=60, judge=brief)
            assert loop.judge == brief

            loop.judge_quiet_streak = 3
            loop.judge_cursors = {"chat-2-2": 5}
            revised = await svc.update(loop.id, judge={"wake_when": "BLOCKED"})
            assert revised is not None
            assert revised.judge == {"wake_when": "BLOCKED"}
            assert revised.judge_quiet_streak == 0, "a new brief starts a new streak"
            assert revised.judge_cursors == {}

            cleared = await svc.update(loop.id, judge={})
            assert cleared is not None and cleared.judge == {}

            await svc.update(loop.id, judge=brief)
            untouched = await svc.update(loop.id, idle_secs=120)
            assert untouched is not None and untouched.judge == brief

            plain = await svc.add("chat-9-9", "no judge here", idle_secs=60)
            assert plain.judge == {}

        asyncio.run(main())
