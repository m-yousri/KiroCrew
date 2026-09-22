"""The work-ledger wake gate's decision layer.

Phase 3 of ``rfc-conductor-work-ledger``. These are the exit criteria that live
in the decision layer rather than in the probe: which events wake, which only
advance the revision, that the revision is stable across a re-read of an
unchanged ledger, and the rate limit and kill switch.

Everything here runs without a gateway, which is the point of keeping the policy
out of the probe: a probe defect cannot look like a policy defect.
"""

from __future__ import annotations

import pytest

from kiro_crew import ledger_wake as lw
from kiro_crew import session_ledger as sl
from kiro_crew import work_ledger

CONDUCTOR = "chat-conductor"
ITEM = "it_abcd1234"
OTHER_ITEM = "it_beef5678"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Own data home per test, so no rate file outlives its own test."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    sl._fold_cache.clear()


class TestActionableEvents:
    """Exit criteria 2, 3 and 4: what wakes, what only advances the revision."""

    @pytest.mark.parametrize("status", sorted(lw.WAKE_STATUSES))
    def test_a_worker_report_with_a_waking_status_wakes(self, status):
        assert lw.is_actionable_event("report", status)

    def test_a_progress_report_does_not_wake(self):
        """It advances the revision instead, so chatter stays free."""
        assert not lw.is_actionable_event("report", "progress")

    @pytest.mark.parametrize("kind", ["create", "bind", "decision", "verdict", "close"])
    def test_the_conductors_own_writes_never_wake_it(self, kind):
        """A gate woken by its owner's writes never sleeps.

        ``verdict`` matters most: it carries a status-shaped field, so a check on
        the status alone would wake the conductor on every ruling it made.
        """
        assert not lw.is_actionable_event(kind, "done")

    def test_every_conductor_action_kind_is_covered(self):
        """Guards the test above from drifting from the store's own vocabulary."""
        worker_owned = lw.WORKER_EVENT_KINDS
        assert work_ledger.EVENT_KINDS - worker_owned == {
            "create",
            "bind",
            "decision",
            "verdict",
            "close",
        }

    def test_the_waking_set_is_a_subset_of_the_stores_worker_statuses(self):
        """A status the store refuses could never reach the gate."""
        assert lw.WAKE_STATUSES <= work_ledger.WORKER_STATUSES

    def test_request_is_absent_until_the_store_has_it(self):
        """Phase 5's status is not on this base, so the set must not name it.

        This fails the day ``request`` is added, which is the reminder to add it
        here in the same change rather than leaving the two to disagree.
        """
        assert ("request" in lw.WAKE_STATUSES) == ("request" in work_ledger.WORKER_STATUSES)

    def test_an_unknown_kind_or_status_does_not_wake(self):
        assert not lw.is_actionable_event("", "")
        assert not lw.is_actionable_event("report", None)
        assert not lw.is_actionable_event("report", "nonsense")

    def test_surrounding_whitespace_does_not_change_the_verdict(self):
        assert lw.is_actionable_event("  report  ", "  done  ")


class TestRevision:
    """Exit criterion 1: an unchanged ledger must read as the same revision."""

    def test_a_re_read_of_an_unchanged_ledger_is_the_same_revision(self):
        ids = {ITEM: "ev-1", OTHER_ITEM: "ev-9"}
        assert lw.revision(ids) == lw.revision(dict(reversed(list(ids.items()))))

    def test_a_new_event_changes_the_revision(self):
        assert lw.revision({ITEM: "ev-1"}) != lw.revision({ITEM: "ev-2"})

    def test_a_new_item_changes_the_revision(self):
        assert lw.revision({ITEM: "ev-1"}) != lw.revision({ITEM: "ev-1", OTHER_ITEM: "ev-1"})

    def test_no_items_is_an_empty_token_not_a_digest(self):
        """Empty disables epoch resets; a digest would assert a revision of nothing."""
        assert lw.revision({}) == ""

    def test_the_token_carries_no_clock(self):
        """A wall clock would make every tick a new revision and defeat the dedupe."""
        import time

        first = lw.revision({ITEM: "ev-1"})
        time.sleep(0.01)
        assert lw.revision({ITEM: "ev-1"}) == first


class TestRateLimit:
    """A runaway worker cannot spend its conductor's whole turn budget."""

    def test_a_fresh_item_is_within_the_limit(self):
        assert lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0)

    def test_asking_does_not_spend_the_budget(self):
        for _ in range(lw.MAX_WAKES_PER_ITEM_PER_HOUR + 5):
            assert lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0)

    def test_the_budget_runs_out(self):
        for _ in range(lw.MAX_WAKES_PER_ITEM_PER_HOUR):
            lw.note_wake(CONDUCTOR, ITEM, now=1000.0)
        assert not lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0)

    def test_the_window_slides_rather_than_muting_the_item(self):
        for _ in range(lw.MAX_WAKES_PER_ITEM_PER_HOUR):
            lw.note_wake(CONDUCTOR, ITEM, now=1000.0)
        assert lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0 + lw._RATE_WINDOW_SECS)

    def test_items_have_separate_budgets(self):
        for _ in range(lw.MAX_WAKES_PER_ITEM_PER_HOUR):
            lw.note_wake(CONDUCTOR, ITEM, now=1000.0)
        assert not lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0)
        assert lw.within_rate_limit(CONDUCTOR, OTHER_ITEM, now=1000.0)

    def test_a_damaged_rate_file_does_not_refuse_a_wake(self):
        """Fail toward spending: a wrongly-quiet tick is silence, which is worse."""
        lw.note_wake(CONDUCTOR, ITEM, now=1000.0)
        (sl.control_dir(CONDUCTOR) / lw._RATE_FILE).write_text("[]", encoding="utf-8")
        assert lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0)

    def test_a_read_creates_no_directory(self):
        assert lw.within_rate_limit(CONDUCTOR, ITEM, now=1000.0)
        assert not sl.control_dir(CONDUCTOR).exists()

    def test_the_tracked_set_is_bounded(self, monkeypatch):
        monkeypatch.setattr(lw, "_MAX_TRACKED_ITEMS", 2)
        for index, item in enumerate(("a", "b", "c")):
            lw.note_wake(CONDUCTOR, item, now=1000.0 + index)
        assert len(lw._read_rate(CONDUCTOR)) == 2

    def test_the_bound_is_derived_from_the_stores_own_item_cap(self):
        """A hardcoded number would drift from the cap it is meant to cover."""
        assert lw._MAX_TRACKED_ITEMS >= work_ledger.MAX_ITEMS_PER_CONDUCTOR


class TestBriefs:
    def test_a_wake_brief_names_the_item_and_its_status(self):
        text = lw.wake_brief(item_id=ITEM, status="blocked")
        assert f"item={ITEM}" in text
        assert "status=blocked" in text

    def test_a_brief_carries_no_ledger_prose(self):
        """The wake text is structural, and this is a boundary test, not a style one.

        ``irq`` persists a brief into its own watch-state file, which is not the
        work ledger and is not behind the store's identity-gated read path. A
        worker's reported summary or a conductor's item title copied into a brief
        would leave that boundary, so the helpers take no prose at all -- the
        signature is the guarantee, not a redaction step that could be skipped.
        """
        import inspect

        for helper in (lw.wake_brief, lw.stall_brief):
            names = set(inspect.signature(helper).parameters)
            assert "summary" not in names
            assert "title" not in names

    def test_a_stall_brief_is_distinguishable_from_a_report_wake(self):
        """The two say opposite things, so a conductor must be able to tell them apart.

        Compared against the report brief rather than by substring: the stall
        brief's own ``last_status=`` field CONTAINS ``status=``, so a naive
        substring check passes for the wrong reason.
        """
        stall = lw.stall_brief(item_id=ITEM, status="progress")
        report = lw.wake_brief(item_id=ITEM, status="blocked")
        assert "reason=stall" in stall
        assert "reason=stall" not in report

    def test_a_stall_brief_with_no_report_yet_still_names_the_status(self):
        text = lw.stall_brief(item_id=ITEM, status=None)
        assert "reason=stall" in text
        assert "last_status=(none yet)" in text


class TestLivenessIsNotReimplemented:
    """The conjunction belongs to the store, and this pins that it stays there."""

    def test_the_gate_defers_to_the_stores_staleness_rule(self):
        """``is_stale`` already requires all three conditions of the conjunction.

        Restating any of them in the gate would be a second copy of a shipped
        rule, so this asserts the store still owns it rather than testing a
        duplicate here.
        """
        assert callable(work_ledger.is_stale)
        assert callable(work_ledger._worker_owns_next_move)
        assert work_ledger.STALE_ELIGIBLE_STATUSES == {"progress", "blocked", "question"}

    def test_a_done_item_is_not_stale_eligible_on_its_own(self):
        """After ``done`` the move is the conductor's, so silence there is expected."""
        assert "done" not in work_ledger.STALE_ELIGIBLE_STATUSES


# ── the probe itself, against a real ledger on disk ──────────────────────
#
# Imports are function-local on purpose: this file's import block is sorted, and
# adding a second top-level block for the probe would either break that order or
# land below the module docstring's contract. Nothing here is hot enough to care.


def _open_ledger(goal: str = "drive the fleet") -> None:
    work_ledger.ensure_conductor(CONDUCTOR, goal=goal)


def _create(title: str = "port the gate") -> str:
    result = work_ledger.apply_conductor_action(
        CONDUCTOR, "create", title=title, acceptance={"kind": "human_approval"}
    )
    return str(result["item"].item_id)


def _report(item_id: str, status: str, summary: str = "moving") -> None:
    work_ledger.apply_worker_report(CONDUCTOR, item_id, status=status, summary=summary)


class _Job:
    id = "loop-1"


class _Ctx:
    """The two attributes the kernel reads off a ctx, and nothing else."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.job = _Job()


def _ctx(key: str = CONDUCTOR) -> _Ctx:
    import json

    return _Ctx(json.dumps({"conductor": key}))


def _probe(*, running: bool = False):
    from kiro_crew.probes.work_ledger import WorkLedgerProbe

    return WorkLedgerProbe(worker_running=lambda _key: running)


def _observe(probe, ctx):
    probe.identity(ctx)
    return probe.observe(ctx)


def _briefs(tick) -> str:
    return "\n".join(obs.brief for obs in tick.observations)


def test_build_serves_the_work_ledger_kind_and_still_refuses_a_stranger():
    from kiro_crew import probes

    assert probes.WORK_LEDGER == "work-ledger"
    assert probes.build(probes.WORK_LEDGER) is not None
    assert probes.build(probes.GH_PR) is not None
    assert probes.build("deployment") is None


def test_neither_monitor_schema_accepts_watch_yet():
    """The field is held back until the whole arm chain can land with it.

    Pinned as an ABSENCE on purpose. A schema that accepts ``watch`` while no path
    forwards it is a worse contract than one that rejects it: the caller is told the
    request was valid and gets an ordinary timer. The field returns in the follow-up
    that also carries the directive payload, the applier and the authz forward, so
    main never holds a validate-then-discard parameter between the two.
    """
    from kiro_crew import validation

    for schema in (validation.MONITOR_START_SCHEMA, validation.MONITOR_UPDATE_SCHEMA):
        assert [f for f in schema.fields if f.name == "watch"] == []
    assert not hasattr(validation, "_WATCH_WORK_LEDGER")


def test_inference_names_the_sessions_own_ledger_only_when_it_is_asked():
    from kiro_crew.probes import targets

    asked = targets.infer("dispatch the queue", watch="work-ledger", slot_key=CONDUCTOR)
    assert asked is not None
    assert (asked.kind, asked.subject) == ("work-ledger", CONDUCTOR)
    assert CONDUCTOR in asked.message
    # The same text with no request names nothing observable, which is the whole
    # reason the field exists: a session's identity is not in its own prose.
    assert targets.infer("dispatch the queue") is None
    # And a request with no session to be about cannot invent one.
    assert targets.infer("dispatch the queue", watch="work-ledger", slot_key="") is None


def test_an_explicit_ledger_watch_ignores_a_pull_request_in_the_same_text():
    from kiro_crew.probes import targets

    text = "watch my ledger; worker A is driving https://github.com/o/r/pull/7"
    ledger = targets.infer(text, watch="work-ledger", slot_key=CONDUCTOR)
    assert ledger is not None and ledger.kind == "work-ledger"
    # Without the request the very same text gates on the pull request, so the
    # precedence is what decides this, not the wording.
    assert targets.infer(text) is not None
    assert targets.infer(text).kind == "gh-pr"


def test_a_gh_pr_hint_falls_through_to_the_text_rather_than_refusing():
    from kiro_crew.probes import targets

    text = "babysit https://github.com/o/r/pull/7"
    hinted = targets.infer(text, watch="gh-pr", slot_key=CONDUCTOR)
    assert hinted is not None and hinted.subject == "o/r#7"


def test_identity_parses_the_conductor_and_refuses_a_config_it_cannot_use():
    probe = _probe()
    assert probe.identity(_ctx()) == ("work-ledger", CONDUCTOR)
    for bad in ("", "not json", "[]", '{"conductor": ""}'):
        with pytest.raises(ValueError):
            probe.identity(_Ctx(bad))


def test_an_unreadable_ledger_is_a_blind_tick_not_a_quiet_one():
    # No ledger at all. QUIET would assert "nothing changed" about something this
    # tick never read, which is how a broken read becomes a silent watch.
    tick = _observe(_probe(), _ctx())
    assert tick.fetch_ok is False
    assert tick.observations == []


def test_the_epoch_moves_when_an_item_reports_and_holds_when_nothing_does():
    _open_ledger()
    item = _create()
    first = _observe(_probe(), _ctx()).epoch
    assert first == _observe(_probe(), _ctx()).epoch
    _report(item, "progress")
    assert _observe(_probe(), _ctx()).epoch != first


def test_each_waking_status_produces_one_observation():
    from kiro_crew import irq

    for status in ("done", "blocked", "question"):
        work_ledger.ensure_conductor(CONDUCTOR, goal="g")
        item = _create(title=f"item for {status}")
        _report(item, status, summary=f"{status} here")
        tick = _observe(_probe(), _ctx())
        mine = [o for o in tick.observations if item in o.brief]
        assert len(mine) == 1, status
        assert mine[0].severity is irq.Severity.WAKE
        assert f"status={status}" in mine[0].brief
        # Keyed on a content-addressed event id, so the key cannot recur and must
        # NOT be cleared when some other item reports.
        assert mine[0].resets_on is irq.ResetsOn.NEVER


def test_a_progress_report_advances_the_epoch_without_waking():
    _open_ledger()
    item = _create()
    _report(item, "progress", summary="halfway")
    tick = _observe(_probe(), _ctx())
    assert tick.epoch
    assert [o for o in tick.observations if item in o.brief] == []


def test_a_conductor_written_event_never_wakes_its_own_author():
    # ``verdict`` carries a status-shaped field, so a gate that checked only the
    # status would wake the conductor on every ruling it made.
    _open_ledger()
    item = _create()
    work_ledger.apply_conductor_action(CONDUCTOR, "verdict", item_id=item, verdict="fail", fails=1)
    tick = _observe(_probe(), _ctx())
    assert [o for o in tick.observations if "reason=stall" not in o.brief] == []


def test_a_silent_worker_wakes_only_while_its_session_is_not_running(monkeypatch):
    _open_ledger()
    item = _create()
    seen: list[bool] = []

    def _stale(_item, *, worker_running, **_kw):
        seen.append(worker_running)
        return not worker_running

    monkeypatch.setattr(work_ledger, "is_stale", _stale)
    idle = _observe(_probe(running=False), _ctx())
    assert "reason=stall" in _briefs(idle)
    assert item in _briefs(idle)
    busy = _observe(_probe(running=True), _ctx())
    assert "reason=stall" not in _briefs(busy)
    # The probe asks with the RESOLVER's answer rather than deciding liveness
    # itself; the conjunction stays the store's.
    assert seen == [False, True]


def test_every_item_closed_is_terminal_but_an_empty_ledger_is_not():
    from kiro_crew import irq

    _open_ledger()
    empty = _observe(_probe(), _ctx())
    assert [o for o in empty.observations if o.severity is irq.Severity.TERMINAL] == []
    item = _create()
    work_ledger.apply_conductor_action(
        CONDUCTOR, "close", item_id=item, state="accepted", decision="verified"
    )
    closed = _observe(_probe(), _ctx())
    assert [o.severity for o in closed.observations] == [irq.Severity.TERMINAL]


class TestWorkerRunning:
    """The one hop the probe cannot make for itself, exercised without a gateway.

    The gateway binds this to its dashboard state and passes the result in, so this
    is where the two-spelling lookup and every unreadable case are pinned.
    """

    class _Slot:
        def __init__(self, running: bool) -> None:
            self.running = running

    class _Table:
        def __init__(self, slots: dict) -> None:
            self.slots = slots

        def get_slot(self, key):
            return self.slots.get(key)

    def test_a_bare_key_reports_its_running_flag(self):
        table = self._Table({"chat-9": self._Slot(True)})
        assert lw.worker_running(table, "chat-9") is True

    def test_an_idle_slot_is_not_running(self):
        table = self._Table({"chat-9": self._Slot(False)})
        assert lw.worker_running(table, "chat-9") is False

    def test_the_prefixed_spelling_is_found_too(self):
        # A dashboard slot is registered under the prefixed key as well as the bare
        # one; missing this spelling would read every dashboard worker as idle.
        table = self._Table({"dashboard_chat-9": self._Slot(True)})
        assert lw.worker_running(table, "chat-9") is True

    def test_an_unknown_key_is_not_running(self):
        assert lw.worker_running(self._Table({}), "chat-9") is False

    def test_no_table_and_no_key_are_not_running(self):
        assert lw.worker_running(None, "chat-9") is False
        assert lw.worker_running(self._Table({}), "") is False

    def test_a_table_without_the_reader_is_not_running(self):
        assert lw.worker_running(object(), "chat-9") is False

    def test_a_raising_lookup_is_not_running_rather_than_an_error(self):
        class _Angry:
            def get_slot(self, key):
                raise RuntimeError("slot table is mid-write")

        assert lw.worker_running(_Angry(), "chat-9") is False


def test_a_finished_goal_is_success_not_a_blocked_pull_request():
    # The driver records success or blocked from the probe's own terminal keys. Read
    # through the pull-request vocabulary alone, a conductor whose every item is
    # closed has no "merged" key and is persisted as BLOCKED -- a succeeded goal
    # filed as a failure.
    from kiro_crew import probes
    from kiro_crew.probes import work_ledger as probe_module

    assert probes.terminal_succeeded((probe_module.DONE_KEY,)) is True
    assert probes.terminal_succeeded(("merged",)) is True
    assert probes.terminal_succeeded(("closed",)) is False
    # Empty or unusable keys mean "ended, not necessarily well", which is what
    # irq.Verdict already documents for an unattributable end.
    assert probes.terminal_succeeded(()) is False
    assert probes.terminal_succeeded(None) is False


def test_the_remembered_token_bound_covers_what_one_window_can_charge():
    """A bound below the budget re-charges evicted tokens and empties it early."""
    from kiro_crew.probes import work_ledger as probe_module

    assert lw._MAX_REMEMBERED_TOKENS >= lw.MAX_WAKES_PER_ITEM_PER_HOUR
    assert lw._MAX_REMEMBERED_TOKENS >= probe_module._EVENT_TAIL


def test_an_unreadable_item_cannot_terminate_the_watch():
    from kiro_crew import irq

    _open_ledger()
    closed = _create(title="finished")
    torn = _create(title="still open, and unreadable")
    work_ledger.apply_conductor_action(
        CONDUCTOR, "close", item_id=closed, state="accepted", decision="verified"
    )
    # The OPEN item's file is damaged. list_work_items skips it so one torn file
    # cannot hide the rest, which leaves every item this tick can read closed --
    # and a watch that trusted that would deactivate with live work outstanding.
    work_ledger.item_path(CONDUCTOR, torn).write_text("{ not json", encoding="utf-8")
    tick = _observe(_probe(), _ctx())
    assert [o for o in tick.observations if o.severity is irq.Severity.TERMINAL] == []


def test_no_config_switch_gates_the_wake():
    import inspect

    # The gate has no kill switch, and this asserts the SUBTRACTION rather than the
    # absence of a line: arming is already opt-in (a conductor must ask for the
    # subject by name) and a loop is already stoppable, so a third off-position was
    # a setting with one consumer -- and the one it had never read a persisted
    # value, because no loader hydration line was ever written for it. Two review
    # lanes asked for the removal on those grounds.
    from kiro_crew import ledger_wake as module
    from kiro_crew import probes as probes_module
    from kiro_crew.config.loader import KiroCrewConfig
    from kiro_crew.probes.work_ledger import WorkLedgerProbe

    assert not hasattr(module, "enabled")
    assert not hasattr(KiroCrewConfig, "ledger_wake")
    assert "enabled" not in inspect.signature(WorkLedgerProbe.__init__).parameters
    assert "enabled" not in inspect.signature(probes_module.build).parameters


def test_the_budget_counts_distinct_wakes_rather_than_re_reads():
    for index in range(lw.MAX_WAKES_PER_ITEM_PER_HOUR):
        assert lw.within_rate_limit(CONDUCTOR, ITEM, token=f"e{index}") is True
        lw.note_wake(CONDUCTOR, ITEM, token=f"e{index}")
    # Budget spent: a NEW wake folds forward.
    assert lw.within_rate_limit(CONDUCTOR, ITEM, token="fresh") is False
    # The wake already charged is still allowed, because a probe re-reports an
    # unchanged condition every tick and those re-reads must not spend anything.
    last = f"e{lw.MAX_WAKES_PER_ITEM_PER_HOUR - 1}"
    assert lw.within_rate_limit(CONDUCTOR, ITEM, token=last) is True
    # A different item has its own budget.
    assert lw.within_rate_limit(CONDUCTOR, OTHER_ITEM, token="e0") is True
