"""Reply threads on crewmate chat messages (``dashboard/chat_threads.py``).

Uses ``async with _client()`` inside each test rather than an async-gen fixture:
the CI-pinned ``pytest-asyncio`` is incompatible with the pinned ``pytest`` for
async fixtures (see test_denied_commands_api.py docstring).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.acp_backends import ACP_BACKEND_KIRO
from kiro_crew.dashboard import chat_threads
from kiro_crew.dashboard.chat_threads import (
    THREAD_BOUNDARY_PROMPT,
    THREAD_BOUNDARY_PROMPT_NO_TOOLS,
    THREAD_INSTRUCTIONS,
    _run_thread_turn,
    api_chat_thread_detail,
    api_chat_thread_reply,
    api_chat_threads_summary,
    build_thread_message,
    summarize,
    thread_session_key,
)
from kiro_crew.dashboard.chat_utils import slot_history_key
from kiro_crew.dashboard.side_readonly_spec import PublishedSpec, ReadOnlySpecError
from kiro_crew.history import ThreadStoreUnreadable
from kiro_crew.members import DM_SLOT_MODE

_MEMBER_SLOT = "member-radar"
_ANSWER = "Five are covered by open PRs, three are queued."


def _make_app(state) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/api/chat/threads", api_chat_threads_summary)
    app.router.add_get("/api/chat/threads/{mid}", api_chat_thread_detail)
    app.router.add_post("/api/chat/threads/{mid}/reply", api_chat_thread_reply)
    return app


def _client(state, *, app_name: str = "") -> TestClient:
    app = _make_app(state)
    if app_name:

        @web.middleware
        async def _as_app(request, handler):
            request["app"] = app_name
            return await handler(request)

        app.middlewares.append(_as_app)
    return TestClient(TestServer(app))


def _member_slot(state, key: str = _MEMBER_SLOT):
    """A crewmate's chat with one exchange in it; returns ``(slot, parent_mid)``.

    The transcript is written to disk because the thread store only writes
    beside a transcript that exists (a chat younger than its first flush answers
    ``transcript_missing``).
    """
    slot = state.get_or_create_slot(key, agent="Radar", mode=DM_SLOT_MODE)
    slot.append("user", "Anything overnight?", broadcast=False)
    row = slot.append(
        "assistant", "Overnight triage: 9 new issues, one needs you.", broadcast=False
    )
    state.conversation_log.append(slot_history_key(slot), "user", "Anything overnight?")
    return slot, row["meta"]["mid"]


def _store_path(state, slot):
    return state.conversation_log.threads_sidecar_path(slot_history_key(slot))


def _threads(state, slot):
    return state.conversation_log.read_threads(slot_history_key(slot))


def _seed_reply(state, slot, mid, role, content, *, max_total: int = 5000):
    return state.conversation_log.append_thread_reply(
        slot_history_key(slot),
        mid,
        chat_threads._new_reply(role, content),
        max_replies=500,
        max_total=max_total,
    )


def _capture_broadcasts(state) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []

    def _record(msg_type, data):
        events.append((msg_type, data))

    state.broadcast_ws = _record
    state.broadcast_ws_owners = _record
    return events


def _finals(events):
    return [d for t, d in events if t == "chat.thread_reply" and d.get("final")]


@pytest.fixture(autouse=True)
def _no_carry_over():
    """The in-flight set is module state; a test that leaves a turn marked in
    flight would refuse the next test's reply."""
    chat_threads._in_flight.clear()
    yield
    chat_threads._in_flight.clear()


def _stub_turn(monkeypatch):
    """Replace the background turn with a no-op that only clears the in-flight
    mark, as the real turn's ``finally`` does."""
    calls: list[dict[str, Any]] = []

    async def _fake(state, slot, mid, run_id, text, parent, context_before, flight_key, identity):
        calls.append({"mid": mid, "text": text, "parent": parent, "context_before": context_before})
        chat_threads._in_flight.discard(flight_key)

    monkeypatch.setattr(chat_threads, "_run_thread_turn", _fake)
    return calls


def _parent() -> dict[str, Any]:
    return {"role": "assistant", "content": "Overnight triage: 9 new issues."}


def _arm_turn(state, monkeypatch, *, answer: str, backend: str = ACP_BACKEND_KIRO):
    """A fake cold session that answers *answer*; returns the recorded calls."""
    calls: list[dict[str, Any]] = []
    provider = MagicMock()

    async def _fake_get_or_create(key, **kwargs):
        calls.append({"key": key, **kwargs})
        return provider, True, False

    state.sessions.get_or_create = _fake_get_or_create
    state.sessions.release = MagicMock()

    async def _fake_stream(provider, message, *, on_chunk=None, **kwargs):
        calls.append({"message": message, **kwargs})
        if answer and on_chunk is not None:
            on_chunk(answer)
        return answer

    monkeypatch.setattr(chat_threads, "stream_and_collect", _fake_stream)
    monkeypatch.setattr(
        chat_threads,
        "publish_readonly_spec",
        lambda base, project=None: PublishedSpec(name=f"{base}--readonly", digest="d" * 64),
    )
    monkeypatch.setattr(
        chat_threads.KiroCrewConfig,
        "load",
        classmethod(lambda cls: MagicMock(agent=MagicMock(acp_backend=backend))),
    )
    monkeypatch.setattr(chat_threads, "warm_project_agent_names", AsyncMock())
    monkeypatch.setattr(
        chat_threads,
        "resolve_agent_bindings",
        lambda cfg, agent, project: MagicMock(kiro_agent="kirocrew"),
    )
    return calls


# ── Store ──


def test_summarize_reports_count_last_ts_and_participants_in_first_appearance_order():
    threads = {
        "m1": [
            {"role": "user", "content": "a", "ts": "2026-01-01T00:00:00+00:00"},
            {"role": "assistant", "content": "b", "ts": "2026-01-01T00:01:00+00:00"},
            {"role": "user", "content": "c", "ts": "2026-01-01T00:02:00+00:00"},
        ],
        "m2": [],
    }
    assert summarize(threads) == {
        "m1": {
            "count": 3,
            "last_reply_ts": "2026-01-01T00:02:00+00:00",
            "participants": ["user", "assistant"],
        }
    }


def test_sidecar_lives_beside_the_transcript_and_is_removed_with_it(tmp_path):
    state = _make_state(tmp_path)
    key = "dashboard:member-radar"
    log = state.conversation_log
    path = log.threads_sidecar_path(key)
    assert path.parent == tmp_path / ".threads"
    log.append(key, "user", "hi")
    assert (
        log.append_thread_reply(
            key, "m-1", chat_threads._new_reply("user", "x"), max_replies=5, max_total=50
        )
        == "ok"
    )
    assert path.exists()
    assert log.delete_session(key)
    assert not path.exists()


def test_store_refuses_to_write_beside_a_missing_transcript(tmp_path):
    state = _make_state(tmp_path)
    log = state.conversation_log
    key = "dashboard:member-nobody"
    reply = chat_threads._new_reply("user", "x")
    assert log.append_thread_reply(key, "m-1", reply, max_replies=5, max_total=50) == "missing"
    assert not log.threads_sidecar_path(key).exists()


def test_a_reply_admitted_against_one_transcript_never_lands_in_its_replacement(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    log.update_metadata(key, {"created_at": "2026-09-22T07:00:00+00:00"})
    identity = log.thread_transcript_identity(key)
    assert identity == "2026-09-22T07:00:00+00:00"
    # The chat is deleted and recreated under the same key while a turn is in flight.
    assert log.delete_session(key)
    log.append(key, "user", "a new chat")
    log.update_metadata(key, {"created_at": "2026-09-22T08:00:00+00:00"})
    reply = chat_threads._new_reply("assistant", "late")
    assert (
        log.append_thread_reply(
            key, mid, reply, max_replies=5, max_total=50, expected_created_at=identity
        )
        == "replaced"
    )
    assert log.read_threads(key) == {}
    # No identity on either side falls through to the existence check alone.
    assert log.append_thread_reply(key, mid, reply, max_replies=5, max_total=50) == "ok"


def test_delete_takes_the_thread_sidecar_with_the_transcript_or_neither(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    assert _seed_reply(state, slot, mid, "user", "kept") == "ok"
    path = _store_path(state, slot)
    transcript = log._path(key)
    # The transcript refuses to go: the sidecar is put back where it was.
    real_unlink = pathlib.Path.unlink

    def _refuse(self, missing_ok=False):
        if self == transcript:
            raise OSError("busy")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", _refuse)
    assert log.delete_session(key) is False
    assert transcript.exists() and path.exists()
    assert log.read_threads(key)[mid][0]["content"] == "kept"
    assert not list(path.parent.glob("*.deleting-*"))
    monkeypatch.undo()
    assert log.delete_session(key)
    assert not path.exists() and not transcript.exists()
    assert not list(path.parent.glob("*.deleting-*"))


def test_unreadable_sidecar_is_refused_never_overwritten(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    log = state.conversation_log
    key = slot_history_key(slot)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ThreadStoreUnreadable):
        log.read_threads(key)
    with pytest.raises(ThreadStoreUnreadable):
        log.append_thread_reply(
            key, mid, chat_threads._new_reply("user", "x"), max_replies=5, max_total=50
        )
    assert path.read_text(encoding="utf-8") == "{not json"
    # Rows of the wrong shape drop one by one; a whole map of the wrong shape refuses.
    path.write_text(json.dumps({"threads": {"m": "not-a-list", "n": [{"role": "user"}, 3]}}))
    assert log.read_threads(key) == {"n": [{"role": "user"}]}
    path.write_text(json.dumps({"threads": []}))
    with pytest.raises(ThreadStoreUnreadable):
        log.read_threads(key)


# ── Summary + detail ──


@pytest.mark.asyncio
async def test_summary_requires_slot_and_refuses_a_plain_chat(tmp_path):
    state = _make_state(tmp_path)
    state.get_or_create_slot("chat-1")
    async with _client(state) as client:
        resp = await client.get("/api/chat/threads")
        assert resp.status == 400
        resp = await client.get("/api/chat/threads", params={"slot": "nope"})
        assert resp.status == 404
        assert (await resp.json())["code"] == "slot_not_found"
        resp = await client.get("/api/chat/threads", params={"slot": "chat-1"})
        assert resp.status == 409
        assert (await resp.json())["code"] == "not_crewmate_chat"


@pytest.mark.asyncio
async def test_summary_is_empty_before_any_reply(tmp_path):
    state = _make_state(tmp_path)
    _member_slot(state)
    async with _client(state) as client:
        resp = await client.get("/api/chat/threads", params={"slot": _MEMBER_SLOT})
        assert resp.status == 200
        assert await resp.json() == {"threads": {}}


@pytest.mark.asyncio
async def test_app_caller_gets_an_indistinguishable_404(tmp_path):
    state = _make_state(tmp_path)
    _, mid = _member_slot(state)
    async with _client(state, app_name="some-app") as client:
        resp = await client.get("/api/chat/threads", params={"slot": _MEMBER_SLOT})
        assert resp.status == 404
        assert (await resp.json())["code"] == "slot_not_found"
        resp = await client.get(f"/api/chat/threads/{mid}", params={"slot": _MEMBER_SLOT})
        assert resp.status == 404
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "x"}
        )
        assert resp.status == 404


@pytest.mark.asyncio
async def test_detail_quotes_the_parent_and_404s_an_unknown_mid(tmp_path):
    state = _make_state(tmp_path)
    _, mid = _member_slot(state)
    async with _client(state) as client:
        resp = await client.get(f"/api/chat/threads/{mid}", params={"slot": _MEMBER_SLOT})
        assert resp.status == 200
        body = await resp.json()
        assert body["parent"]["mid"] == mid
        assert body["parent"]["role"] == "assistant"
        assert body["parent"]["content"].startswith("Overnight triage")
        assert body["replies"] == []
        assert body["in_flight"] is False
        resp = await client.get("/api/chat/threads/m-missing", params={"slot": _MEMBER_SLOT})
        assert resp.status == 404
        assert (await resp.json())["code"] == "parent_not_found"


@pytest.mark.asyncio
async def test_unreadable_sidecar_answers_503_on_every_route(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    _stub_turn(monkeypatch)
    path = _store_path(state, slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    async with _client(state) as client:
        responses = [
            await client.get("/api/chat/threads", params={"slot": _MEMBER_SLOT}),
            await client.get(f"/api/chat/threads/{mid}", params={"slot": _MEMBER_SLOT}),
            await client.post(
                f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "hi"}
            ),
        ]
        for resp in responses:
            assert resp.status == 503
            assert (await resp.json())["code"] == "threads_unavailable"
    assert path.read_text(encoding="utf-8") == "{not json"
    assert f"{slot.key}:{mid}" not in chat_threads._in_flight


# ── Reply ──


@pytest.mark.asyncio
async def test_reply_is_stored_broadcast_and_starts_the_turn(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    slot, mid = _member_slot(state)
    calls = _stub_turn(monkeypatch)
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply",
            json={"slot_key": _MEMBER_SLOT, "text": "  What about the other 8?  "},
        )
        assert resp.status == 202
        body = await resp.json()
        assert body["reply"]["role"] == "user"
        assert body["reply"]["content"] == "What about the other 8?"
        assert body["run_id"]
        for task in list(state._background_tasks):
            await task
        # The store holds the reply; the summary reports it; the main chat did not grow.
        resp = await client.get("/api/chat/threads", params={"slot": _MEMBER_SLOT})
        assert (await resp.json())["threads"] == {
            mid: {"count": 1, "last_reply_ts": body["reply"]["ts"], "participants": ["user"]}
        }
        assert len(slot.messages) == 2
        # The user's frame went out on the thread channel, not the main transcript.
        thread_frames = [d for t, d in events if t == "chat.thread_reply"]
        assert [f["role"] for f in thread_frames] == ["user"]
        assert thread_frames[0]["mid"] == mid
        assert thread_frames[0]["slot"] == _MEMBER_SLOT
        assert not [t for t, _ in events if t in ("chat_message", "chat_done")]
        # The turn got the parent and the chat before it.
        assert calls and calls[0]["mid"] == mid
        assert calls[0]["parent"]["content"].startswith("Overnight triage")
        assert [r["content"] for r in calls[0]["context_before"]] == ["Anything overnight?"]


@pytest.mark.asyncio
async def test_reply_validation(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    _, mid = _member_slot(state)
    _stub_turn(monkeypatch)
    async with _client(state) as client:
        resp = await client.post(f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT})
        assert resp.status == 400
        assert (await resp.json())["code"] == "missing_required_fields"
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "   "}
        )
        assert resp.status == 400
        assert (await resp.json())["code"] == "empty_reply"
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply",
            json={"slot_key": _MEMBER_SLOT, "text": "x" * (chat_threads._MAX_REPLY_BYTES + 1)},
        )
        assert resp.status == 413
        resp = await client.post(
            "/api/chat/threads/m-missing/reply", json={"slot_key": _MEMBER_SLOT, "text": "hi"}
        )
        assert resp.status == 404
        assert (await resp.json())["code"] == "parent_not_found"
        resp = await client.post(
            f"/api/chat/threads/{'m' * 129}/reply", json={"slot_key": _MEMBER_SLOT, "text": "hi"}
        )
        assert resp.status == 400


@pytest.mark.asyncio
async def test_second_reply_while_the_crewmate_is_replying_is_refused(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    _, mid = _member_slot(state)

    async def _hold(state, slot, mid, run_id, text, parent, context_before, flight_key, identity):
        pass  # never clears the mark: the turn is still running

    monkeypatch.setattr(chat_threads, "_run_thread_turn", _hold)
    async with _client(state) as client:
        first = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "one"}
        )
        assert first.status == 202
        second = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "two"}
        )
        assert second.status == 409
        assert (await second.json())["code"] == "thread_turn_in_flight"
        resp = await client.get(f"/api/chat/threads/{mid}", params={"slot": _MEMBER_SLOT})
        assert (await resp.json())["in_flight"] is True


@pytest.mark.asyncio
async def test_two_replies_racing_through_the_store_write_run_one_turn(tmp_path, monkeypatch):
    """The reservation is taken before the store write suspends, so a
    double-click cannot start two turns on one thread."""
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    calls = _stub_turn(monkeypatch)
    async with _client(state) as client:
        a, b = await asyncio.gather(
            client.post(
                f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "one"}
            ),
            client.post(
                f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "two"}
            ),
        )
        for task in list(state._background_tasks):
            await task
    assert sorted([a.status, b.status]) == [202, 409]
    assert len(calls) == 1
    assert len(_threads(state, slot)[mid]) == 1


@pytest.mark.asyncio
async def test_thread_full_is_refused_and_releases_the_mark(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    _stub_turn(monkeypatch)
    monkeypatch.setattr(chat_threads, "_MAX_REPLIES_PER_THREAD", 1)
    assert _seed_reply(state, slot, mid, "user", "one") == "ok"
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "two"}
        )
        assert resp.status == 409
        assert (await resp.json())["code"] == "thread_full"
    assert f"{slot.key}:{mid}" not in chat_threads._in_flight


def test_the_sidecar_as_a_whole_is_bounded_across_threads(tmp_path):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    assert _seed_reply(state, slot, "m-a", "user", "1", max_total=2) == "ok"
    assert _seed_reply(state, slot, "m-b", "user", "2", max_total=2) == "ok"
    assert _seed_reply(state, slot, mid, "user", "3", max_total=2) == "sidecar_full"
    assert sum(len(r) for r in _threads(state, slot).values()) == 2


@pytest.mark.asyncio
async def test_a_full_sidecar_is_refused_with_its_own_code(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    _stub_turn(monkeypatch)
    monkeypatch.setattr(chat_threads, "_MAX_REPLIES_PER_SIDECAR", 1)
    assert _seed_reply(state, slot, "m-other", "user", "one") == "ok"
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "two"}
        )
        assert resp.status == 409
        assert (await resp.json())["code"] == "threads_full"
    assert f"{slot.key}:{mid}" not in chat_threads._in_flight


@pytest.mark.asyncio
async def test_a_reply_before_the_first_flush_is_refused_and_releases_the_mark(
    tmp_path, monkeypatch
):
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot(_MEMBER_SLOT, agent="Radar", mode=DM_SLOT_MODE)
    mid = slot.append("assistant", "hello", broadcast=False)["meta"]["mid"]
    _stub_turn(monkeypatch)
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply", json={"slot_key": _MEMBER_SLOT, "text": "hi"}
        )
        assert resp.status == 409
        assert (await resp.json())["code"] == "transcript_missing"
    assert f"{slot.key}:{mid}" not in chat_threads._in_flight


@pytest.mark.asyncio
async def test_a_parent_that_exists_only_on_disk_is_found_when_the_window_is_idle(tmp_path):
    """The memory window claims the whole chat (_disk_older_count == 0) yet the
    disk transcript is longer: an idle window reconciles against disk, so a
    parent only disk holds is found rather than answered 404."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot(_MEMBER_SLOT, agent="Radar", mode=DM_SLOT_MODE)
    log = state.conversation_log
    key = slot_history_key(slot)
    log.append(key, "user", "Anything overnight?", mid="m-user")
    log.append(key, "assistant", "a row only disk holds", mid="m-disk-only")
    assert slot.messages == [] and slot._disk_older_count == 0
    async with _client(state) as client:
        resp = await client.get("/api/chat/threads/m-disk-only", params={"slot": _MEMBER_SLOT})
        assert resp.status == 200
        assert (await resp.json())["parent"]["content"] == "a row only disk holds"


# ── Envelope ──


def test_first_turn_envelope_carries_context_parent_thread_and_boundary():
    msg = build_thread_message(
        _parent(),
        [{"role": "user", "content": "Anything overnight?"}],
        [
            {"role": "user", "content": "What about the other 8?"},
            {"role": "assistant", "content": "5 covered, 3 queued."},
            {"role": "user", "content": "Is #4198 among them?"},
        ],
        "Is #4198 among them?",
        is_first_turn=True,
        tools_available=True,
    )
    assert msg.startswith(THREAD_INSTRUCTIONS)
    assert "User: Anything overnight?" in msg
    assert "You: Overnight triage: 9 new issues." in msg
    assert "You: 5 covered, 3 queued." in msg
    # The reply being answered appears once, at the end, not also in the thread block.
    assert msg.count("Is #4198 among them?") == 1
    assert msg.rstrip().endswith("User: Is #4198 among them?")
    assert THREAD_BOUNDARY_PROMPT in msg
    assert THREAD_BOUNDARY_PROMPT_NO_TOOLS not in msg


def test_no_tools_boundary_and_bare_follow_up():
    msg = build_thread_message(_parent(), [], [], "why?", is_first_turn=True, tools_available=False)
    assert THREAD_BOUNDARY_PROMPT_NO_TOOLS in msg
    assert THREAD_BOUNDARY_PROMPT not in msg
    bare = build_thread_message(
        _parent(), [], [], " why? ", is_first_turn=False, tools_available=True
    )
    assert bare == "User: why?"


def test_thread_session_key_is_its_own_stateless_dashboard_surface():
    from kiro_crew import session as session_module
    from kiro_crew.messaging.link import telemetry_channel_of
    from kiro_crew.sel import _infer_source

    key = thread_session_key("member-radar", "m-1")
    assert key == "thread:member-radar:m-1"
    assert _infer_source(key) == "dashboard"
    assert any(key.startswith(p) for p in session_module._STATELESS_PREFIXES)
    assert telemetry_channel_of(key) == "thread"


# ── The turn ──


@pytest.mark.asyncio
async def test_turn_runs_read_only_in_its_own_session_and_lands_in_the_thread(
    tmp_path, monkeypatch
):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    slot, mid = _member_slot(state)
    calls = _arm_turn(state, monkeypatch, answer=_ANSWER)
    async with _client(state) as client:
        resp = await client.post(
            f"/api/chat/threads/{mid}/reply",
            json={"slot_key": _MEMBER_SLOT, "text": "the other 8?"},
        )
        assert resp.status == 202
        for task in list(state._background_tasks):
            await task
        detail = await (
            await client.get(f"/api/chat/threads/{mid}", params={"slot": _MEMBER_SLOT})
        ).json()
    assert [r["role"] for r in detail["replies"]] == ["user", "assistant"]
    assert detail["replies"][1]["content"] == _ANSWER
    assert detail["in_flight"] is False
    # The main chat is untouched.
    assert len(slot.messages) == 2
    # Own isolated session, derived read-only spec, READ_ONLY policy.
    acquired = calls[0]
    assert acquired["key"] == thread_session_key(_MEMBER_SLOT, mid)
    assert acquired["agent"] == "kirocrew--readonly"
    streamed = calls[1]
    assert streamed["approval_policy"] == chat_threads.ToolApprovalPolicy.READ_ONLY
    assert streamed["session_key"] == thread_session_key(_MEMBER_SLOT, mid)
    assert streamed["message"].startswith(THREAD_INSTRUCTIONS)
    assert streamed["message"].rstrip().endswith("User: the other 8?")
    state.sessions.release.assert_called_once_with(thread_session_key(_MEMBER_SLOT, mid))
    # Every reply cold-starts: the session is destroyed after the turn.
    state.sessions.destroy.assert_awaited_once_with(thread_session_key(_MEMBER_SLOT, mid))
    # Streamed delta, then the terminal frame carrying the stored row.
    frames = [d for t, d in events if t == "chat.thread_reply" and d["role"] == "assistant"]
    assert frames[0]["content"] == _ANSWER and "final" not in frames[0]
    assert frames[-1]["final"] is True
    assert frames[-1]["reply"]["id"] == detail["replies"][1]["id"]
    assert not [t for t, _ in events if t in ("chat_message", "chat_done")]


@pytest.mark.asyncio
async def test_turn_without_tools_uses_reject_all_and_the_no_tools_boundary(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    calls = _arm_turn(state, monkeypatch, answer="ok", backend="other-harness")
    assert _seed_reply(state, slot, mid, "user", "hi") == "ok"
    await _run_thread_turn(
        state, slot, mid, "run-1", "hi", _parent(), [], f"{slot.key}:{mid}", None
    )
    assert calls[0]["agent"] == "kirocrew"
    assert calls[1]["approval_policy"] == chat_threads.ToolApprovalPolicy.REJECT_ALL
    assert THREAD_BOUNDARY_PROMPT_NO_TOOLS in calls[1]["message"]


@pytest.mark.asyncio
async def test_empty_answer_becomes_the_visible_boundary_line(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    slot, mid = _member_slot(state)
    _arm_turn(state, monkeypatch, answer="")
    await _run_thread_turn(
        state, slot, mid, "run-1", "hi", _parent(), [], f"{slot.key}:{mid}", None
    )
    final = _finals(events)[-1]
    assert "read-only" in final["content"]
    assert "main chat" in final["content"]
    assert "is_error" not in final
    stored = _threads(state, slot)[mid]
    assert stored[-1]["role"] == "assistant" and stored[-1]["content"] == final["content"]


@pytest.mark.asyncio
async def test_a_long_answer_is_clipped_before_it_is_stored(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    slot, mid = _member_slot(state)
    _arm_turn(state, monkeypatch, answer="x" * (chat_threads._MAX_STORED_REPLY_CHARS + 500))
    await _run_thread_turn(
        state, slot, mid, "run-1", "hi", _parent(), [], f"{slot.key}:{mid}", None
    )
    stored = _threads(state, slot)[mid][-1]["content"]
    assert len(stored) == chat_threads._MAX_STORED_REPLY_CHARS + len(chat_threads._CLIPPED_MARKER)
    assert stored.endswith(chat_threads._CLIPPED_MARKER)


@pytest.mark.asyncio
async def test_a_reply_the_store_refuses_is_published_as_a_failure(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    slot, mid = _member_slot(state)
    _arm_turn(state, monkeypatch, answer="a fine answer")
    # The thread fills up between the user's reply and the crewmate's.
    monkeypatch.setattr(chat_threads, "_MAX_REPLIES_PER_THREAD", 1)
    assert _seed_reply(state, slot, mid, "user", "hi") == "ok"
    await _run_thread_turn(
        state, slot, mid, "run-1", "hi", _parent(), [], f"{slot.key}:{mid}", None
    )
    final = _finals(events)[-1]
    assert final["is_error"] is True
    assert "reply" not in final
    assert "full" in final["content"]
    assert [r["role"] for r in _threads(state, slot)[mid]] == ["user"]


@pytest.mark.asyncio
async def test_turn_failure_sends_a_plain_final_error_and_persists_nothing(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    slot, mid = _member_slot(state)
    _arm_turn(state, monkeypatch, answer="never")

    def _refuse(base, project=None):
        raise ReadOnlySpecError("base_spec_missing", "no spec")

    monkeypatch.setattr(chat_threads, "publish_readonly_spec", _refuse)
    flight = f"{slot.key}:{mid}"
    chat_threads._in_flight.add(flight)
    await _run_thread_turn(state, slot, mid, "run-1", "hi", _parent(), [], flight, None)
    final = _finals(events)
    assert len(final) == 1
    assert final[0]["is_error"] is True
    assert "server" not in final[0]["content"].lower()
    # A refused spec is not a transient: the sentence points at the main chat, not at a retry.
    assert "main chat" in final[0]["content"]
    assert "Try again" not in final[0]["content"]
    assert _threads(state, slot) == {}
    assert flight not in chat_threads._in_flight
    # No session was created, so nothing was released.
    state.sessions.release.assert_not_called()


@pytest.mark.asyncio
async def test_a_signed_out_harness_says_so(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    slot, mid = _member_slot(state)
    _arm_turn(state, monkeypatch, answer="never")

    class AcpAuthRequired(Exception):
        """Stands in for the harness's own signed-out error, matched by name."""

    async def _signed_out(*args, **kwargs):
        raise AcpAuthRequired("signed out")

    monkeypatch.setattr(chat_threads, "stream_and_collect", _signed_out)
    monkeypatch.setattr("kiro_crew.dashboard.chat_runner._mark_kiro_signed_out", lambda state: None)
    await _run_thread_turn(
        state, slot, mid, "run-1", "hi", _parent(), [], f"{slot.key}:{mid}", None
    )
    final = _finals(events)[-1]
    assert final["is_error"] is True
    assert final["content"]
    assert _threads(state, slot) == {}
