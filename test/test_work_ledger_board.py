"""Tests for the Crew page's masked work-ledger read (RFC Phase 4, the surfaces).

Two conductors and two workers, the shape Phase 2's own tests established, so
isolation is exercised rather than assumed. Ledgers are written through the
store's real API (``ensure_conductor`` / ``apply_conductor_action`` /
``apply_worker_report``), because the two criteria under test are both about the
shape the store actually produces.

The outstanding rule is tested twice over: once end to end against a real
unanswered question, and once as a unit against explicit timestamps. The ordering
cases (decided-then-asked-again versus asked-then-decided) need timestamps that
differ by a known sign, which back-to-back store writes cannot guarantee.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import make_mocked_request

from kiro_crew import work_ledger as wl
from kiro_crew.dashboard.handlers import work_ledger_board as board

CONDUCTOR = "chat-1-conductor"
OTHER_CONDUCTOR = "chat-9-other-conductor"
WORKER = "chat-2-worker"
OTHER_WORKER = "chat-8-other-worker"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test writes into its own data home, never the live one."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    yield


def _item(conductor: str = CONDUCTOR, *, title: str = "port the gate") -> str:
    """Mint one item on *conductor*'s ledger and return its id."""
    wl.ensure_conductor(conductor, goal="drive the fleet")
    result = wl.apply_conductor_action(
        conductor, "create", title=title, acceptance={"kind": "human_approval"}
    )
    return result["item"].item_id


def _bind(item_id: str, worker: str = WORKER, conductor: str = CONDUCTOR) -> None:
    wl.apply_conductor_action(conductor, "bind", item_id=item_id, worker_session_key=worker)


def _report(item_id: str, status: str, summary: str = "here is where it is", **kw) -> None:
    wl.apply_worker_report(CONDUCTOR, item_id, status=status, summary=summary, **kw)


def _slot(*, running: bool = False):
    return SimpleNamespace(running=running)


def _state(slots: dict | None = None):
    table = dict(slots or {})
    return SimpleNamespace(_slots=table, get_slot=lambda key: table.get(key))


async def _call(state, conductor: str):
    request = make_mocked_request(
        "GET", f"/api/crew-board?conductor={conductor}", app={"state": state}
    )
    response = await board.api_work_ledger_board(request)
    return json.loads(response.body.decode()), response


# ── criterion 4: the masked field is absent ───────────────────────────────


@pytest.mark.asyncio
async def test_no_row_carries_worker_session_key():
    """The one field Phase 4 forbids on a Crew page payload.

    Asserted against an item that HAS one bound, so the test would fail if the
    mask were merely never exercised.
    """
    item_id = _item()
    _bind(item_id)
    _report(item_id, "progress")
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    assert body["items"], "the fixture must produce at least one row"
    for row in body["items"]:
        assert "worker_session_key" not in row


@pytest.mark.asyncio
async def test_the_key_is_absent_from_the_whole_serialized_payload():
    """Not just the row dicts — the spelling appears nowhere in the response.

    Catches a key that gets reintroduced somewhere other than the row, which a
    per-row check would pass.
    """
    item_id = _item()
    _bind(item_id)
    body, response = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    assert "worker_session_key" not in response.body.decode()
    assert WORKER not in json.dumps(body)


@pytest.mark.asyncio
async def test_the_bind_event_text_is_emptied_because_it_is_the_key():
    """The leak a per-row mask does not close.

    A ``bind`` line's text IS the worker session key, so stripping only
    ``WorkItem.worker_session_key`` still publishes it inside the event log. The
    line is kept — its ``kind`` and ``ts`` are when dispatch happened — and only
    the text goes.
    """
    item_id = _item()
    _bind(item_id)
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    binds = [e for e in body["items"][0]["events"] if e["kind"] == "bind"]
    assert binds, "the fixture must produce a bind event"
    for event in binds:
        assert event["text"] == ""
        assert event["ts"], "the timeline still needs when dispatch happened"


# ── criterion 1: the conductor's shape, minus the masked field ─────────────


@pytest.mark.asyncio
async def test_a_row_is_the_conductor_item_shape_minus_the_masked_field():
    """Pins the projection against the store's own serializer, key by key.

    Built from ``WorkItem.to_dict()`` rather than a hand-written list, so adding a
    field to the item without deciding whether the page shows it fails here.
    """
    item_id = _item()
    _bind(item_id)
    _report(item_id, "progress")
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    row = body["items"][0]

    item_keys = set(wl.WorkItem().to_dict()) - {"worker_session_key"}
    derived = {"orphaned", "stale", "acceptance_concrete", "events"}
    joined = {"alive", "outstanding", "terminal"}
    assert set(row) == item_keys | derived | joined


@pytest.mark.asyncio
async def test_events_ride_on_the_row_in_the_store_s_event_shape():
    """Criterion 1's "items and events from the same endpoint" half."""
    item_id = _item()
    _bind(item_id)
    _report(item_id, "progress")
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    events = body["items"][0]["events"]
    assert events, "a bound, reported item has events"
    assert set(events[0]) == set(wl.WorkEvent().to_dict())
    assert {e["kind"] for e in events} <= wl.EVENT_KINDS


# ── the "Needs a ruling" band ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_question_with_no_decision_is_outstanding():
    item_id = _item()
    _bind(item_id)
    _report(item_id, "question", summary="which option do you want?")
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    assert body["items"][0]["outstanding"] is True


@pytest.mark.asyncio
async def test_a_progress_report_is_not_outstanding():
    item_id = _item()
    _bind(item_id)
    _report(item_id, "progress")
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    assert body["items"][0]["outstanding"] is False


def test_a_decision_after_the_question_answers_it():
    item = wl.WorkItem(status="question", last_report_at="2026-09-22T04:00:00+00:00")
    events = [
        {"ts": "2026-09-22T04:00:00+00:00", "kind": "report", "status": "question"},
        {"ts": "2026-09-22T05:00:00+00:00", "kind": "decision", "status": None},
    ]
    assert board._is_outstanding(item, events) is False


def test_a_decision_before_a_fresh_question_does_not_answer_it():
    """Decided, handed back, asked again — the old ruling must not silence it."""
    item = wl.WorkItem(status="question", last_report_at="2026-09-22T06:00:00+00:00")
    events = [
        {"ts": "2026-09-22T04:00:00+00:00", "kind": "report", "status": "question"},
        {"ts": "2026-09-22T05:00:00+00:00", "kind": "decision", "status": None},
        {"ts": "2026-09-22T06:00:00+00:00", "kind": "report", "status": "question"},
    ]
    assert board._is_outstanding(item, events) is True


def test_a_question_whose_ask_cannot_be_dated_is_left_outstanding():
    """Fails toward the human: an undateable question still gets looked at."""
    item = wl.WorkItem(status="question", last_report_at="not a timestamp")
    events = [{"ts": "nonsense", "kind": "report", "status": "question"}]
    assert board._is_outstanding(item, events) is True


def test_only_a_question_status_can_be_outstanding():
    for status in ("progress", "done", "blocked"):
        item = wl.WorkItem(status=status)
        assert board._is_outstanding(item, []) is False


# ── the alive join ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_alive_is_joined_server_side_from_the_worker_slot():
    """Three states, derived from a key the payload never carries."""
    running = _item(title="busy")
    idle = _item(title="waiting")
    gone = _item(title="finished")
    for item_id, worker in ((running, "w-run"), (idle, "w-idle"), (gone, "w-gone")):
        _bind(item_id, worker)
    state = _state({"w-run": _slot(running=True), "w-idle": _slot()})
    body, _ = await _call(state, CONDUCTOR)
    alive = {row["title"]: row["alive"] for row in body["items"]}
    assert alive == {"busy": "running", "waiting": "idle", "finished": "closed"}


@pytest.mark.asyncio
async def test_an_unbound_item_is_closed_not_running():
    """Nothing is running for an item that was never dispatched."""
    _item(title="never dispatched")
    body, _ = await _call(_state(), CONDUCTOR)
    assert body["items"][0]["alive"] == "closed"


# ── terminal and orphaned ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_terminal_comes_from_the_store_s_own_property():
    open_id = _item(title="still open")
    closed_id = _item(title="finished")
    wl.apply_conductor_action(
        CONDUCTOR, "close", item_id=closed_id, state="accepted", decision="looks right"
    )
    body, _ = await _call(_state(), CONDUCTOR)
    terminal = {row["title"]: row["terminal"] for row in body["items"]}
    assert terminal == {"still open": False, "finished": True}
    assert open_id != closed_id


@pytest.mark.asyncio
async def test_the_orphaned_flag_is_carried_through_to_the_page():
    """It drives the take-over and stop affordances, so it must survive the mask."""
    item_id = _item()
    _bind(item_id)
    body, _ = await _call(_state({WORKER: _slot()}), CONDUCTOR)
    assert "orphaned" in body["items"][0]
    # The conductor's own slot is absent from this fixture's table.
    assert body["conductor_alive"] == "closed"
    assert item_id


# ── two conductors, two workers: isolation ────────────────────────────────


@pytest.mark.asyncio
async def test_a_board_shows_only_its_own_conductor_s_items():
    mine = _item(CONDUCTOR, title="mine")
    theirs = _item(OTHER_CONDUCTOR, title="theirs")
    _bind(mine, WORKER, CONDUCTOR)
    wl.apply_conductor_action(
        OTHER_CONDUCTOR, "bind", item_id=theirs, worker_session_key=OTHER_WORKER
    )
    body, _ = await _call(_state({WORKER: _slot(), OTHER_WORKER: _slot()}), CONDUCTOR)
    assert [row["title"] for row in body["items"]] == ["mine"]


# ── refusals and the ad-hoc gap ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_missing_conductor_parameter_is_refused():
    body, response = await _call(_state(), "")
    assert response.status == 400
    assert body["code"] == "missing_conductor"


@pytest.mark.asyncio
async def test_a_session_with_no_work_ledger_answers_no_ledger():
    """Every ad-hoc conductor today: only the conductor tooling opens a ledger."""
    _body, response = await _call(_state(), "chat-77-never-conducted")
    assert response.status != 200


@pytest.mark.asyncio
async def test_channel_rendering_is_reported_unavailable_until_phase_5():
    _item()
    body, _ = await _call(_state(), CONDUCTOR)
    assert body["channels_available"] is False


# ── the action route ──────────────────────────────────────────────────────


async def _act(state, conductor: str, item_id: str, action: str):
    request = make_mocked_request("POST", "/api/crew-board/action", app={"state": state})
    request.json = _async_json({"conductor": conductor, "item_id": item_id, "action": action})
    response = await board.api_work_ledger_board_action(request)
    return json.loads(response.body.decode()), response


def _async_json(payload):
    async def _read():
        return payload

    return _read


@pytest.fixture
def _stop_calls(monkeypatch):
    """Record every delegation to the shared stop primitive."""
    calls: list[dict] = []

    async def _fake_stop(state, slot, *, source="dashboard", escalate=True, **kw):
        calls.append({"slot": slot, "source": source, "escalate": escalate})
        return {"ok": True}

    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_handlers.stop_slot_turn", _fake_stop, raising=True
    )
    return calls


def _orphaned_item() -> str:
    """An item whose conductor slot is gone and which is not terminal."""
    item_id = _item()
    _bind(item_id)
    _report(item_id, "progress")
    return item_id


@pytest.mark.asyncio
async def test_stop_delegates_to_the_shared_primitive(_stop_calls):
    """The Stop button's own path, not a second one invented here."""
    item_id = _orphaned_item()
    body, response = await _act(_state({WORKER: _slot(running=True)}), CONDUCTOR, item_id, "stop")
    assert response.status == 200
    assert body["ok"] is True
    assert len(_stop_calls) == 1
    assert _stop_calls[0]["source"] == "crew_board"


@pytest.mark.asyncio
async def test_stop_does_not_escalate_because_a_browser_retries(_stop_calls):
    """A timed-out request re-sent must not discard the worker's queue."""
    item_id = _orphaned_item()
    await _act(_state({WORKER: _slot(running=True)}), CONDUCTOR, item_id, "stop")
    assert _stop_calls[0]["escalate"] is False


@pytest.mark.asyncio
async def test_no_chat_key_appears_anywhere_in_an_action_response(_stop_calls):
    """The whole-response assertion: the route acts on a key it never reveals."""
    item_id = _orphaned_item()
    body, response = await _act(_state({WORKER: _slot(running=True)}), CONDUCTOR, item_id, "stop")
    raw = response.body.decode()
    assert "chat-" not in raw
    assert WORKER not in raw
    assert "worker_session_key" not in raw
    assert CONDUCTOR not in json.dumps(body)


@pytest.mark.asyncio
async def test_another_conductors_item_is_not_actionable(_stop_calls):
    """Non-owner refusal: the item is looked up only on the named ledger."""
    theirs = _item(OTHER_CONDUCTOR, title="not yours")
    _bind(theirs, OTHER_WORKER, OTHER_CONDUCTOR)
    _item()  # our own ledger exists, so this is isolation and not an empty store
    body, response = await _act(_state({OTHER_WORKER: _slot()}), CONDUCTOR, theirs, "stop")
    assert response.status == 404
    assert body["code"] == "item_not_found"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_a_non_orphaned_item_is_refused_409(_stop_calls):
    """The conductor is still reading its own reports, so there is nothing to take over."""
    item_id = _orphaned_item()
    alive = _state({CONDUCTOR: _slot(), WORKER: _slot(running=True)})
    body, response = await _act(alive, CONDUCTOR, item_id, "stop")
    assert response.status == 409
    assert body["code"] == "not_orphaned"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_a_terminal_item_is_refused_because_it_is_not_orphaned(_stop_calls):
    """``is_orphaned`` is a conjunction: a closed item never qualifies."""
    item_id = _orphaned_item()
    wl.apply_conductor_action(CONDUCTOR, "close", item_id=item_id, state="accepted")
    body, response = await _act(_state({WORKER: _slot()}), CONDUCTOR, item_id, "stop")
    assert response.status == 409
    assert body["code"] == "not_orphaned"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_an_unknown_item_is_refused_404(_stop_calls):
    _item()
    body, response = await _act(_state(), CONDUCTOR, "it_deadbeef", "stop")
    assert response.status == 404
    assert body["code"] == "item_not_found"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_an_unknown_action_is_refused_400(_stop_calls):
    item_id = _orphaned_item()
    body, response = await _act(_state({WORKER: _slot()}), CONDUCTOR, item_id, "delete")
    assert response.status == 400
    assert body["code"] == "unknown_action"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_take_over_is_refused_because_no_primitive_exists(_stop_calls):
    """Phase 4 names the affordance; nothing on this gateway performs it.

    Refused with the same code the read advertises, so the page's disabled reason
    and the route's refusal cannot drift apart.
    """
    item_id = _orphaned_item()
    body, response = await _act(_state({WORKER: _slot()}), CONDUCTOR, item_id, "take_over")
    assert response.status == 501
    assert body["code"] == board._TAKE_OVER_UNAVAILABLE_CODE
    assert not _stop_calls


@pytest.mark.asyncio
async def test_an_item_with_no_bound_worker_has_nothing_to_stop(_stop_calls):
    item_id = _item()
    body, response = await _act(_state(), CONDUCTOR, item_id, "stop")
    assert response.status == 409
    assert body["code"] == "no_worker_bound"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_a_closed_worker_session_has_nothing_to_stop(_stop_calls):
    """Bound, orphaned, but its session is gone — refused rather than a false ok."""
    item_id = _orphaned_item()
    body, response = await _act(_state(), CONDUCTOR, item_id, "stop")
    assert response.status == 409
    assert body["code"] == "worker_closed"
    assert not _stop_calls


@pytest.mark.asyncio
async def test_the_read_advertises_which_actions_this_gateway_can_perform():
    """The page renders take-over disabled from the server's own answer."""
    _item()
    body, _ = await _call(_state(), CONDUCTOR)
    assert body["stop_available"] is True
    assert body["take_over_available"] is False
    assert body["take_over_unavailable_code"] == "no_server_primitive"
