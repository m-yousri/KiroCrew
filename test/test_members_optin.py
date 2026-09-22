"""The one-time crewmate opt-in for existing custom agents (``/api/members/optin``).

Covers the contract the Crewmates page decides from:

* which installed specs are offered as crewmates-to-be (and which are not:
  the runtime's own specs, private copies, names already bound, names the
  create route would refuse) with their chat counts and ordering;
* the ``GET`` shape (``done`` / ``crewmates`` / ``candidates``), its app-token
  denial and its degrade when history is unreadable;
* the ``POST .../done`` record: owner-gated, a delta write of exactly one key.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.agent_discovery import AgentInfo
from kiro_crew.config.loader import KiroCrewAgentConfig
from kiro_crew.dashboard.handlers import members as members_mod


def _spec(name: str, **kw) -> AgentInfo:
    base: dict[str, Any] = dict(
        name=name, filename=f"{name}.json", description=f"{name} does things", model="auto"
    )
    base.update(kw)
    return AgentInfo(**base)


def _cfg(agents: dict | None = None, done: bool = False):
    return SimpleNamespace(
        agents=agents or {},
        default_agent="kirocrew",
        memory_stores={},
        dashboard=SimpleNamespace(crewmate_optin_done=done),
    )


_SPECS = [
    _spec("kirocrew", source="kirocrew", kirocrew_owned=True),
    _spec("conductor", kirocrew_owned=True),
    _spec("radar"),
    _spec("scribe"),
    _spec("scout"),
    _spec("radar-copy", private_to="radar"),
    _spec("bound-already"),
    _spec("ghost", filename=""),
    _spec("bad/name"),
]


def _candidates(cfg, usage):
    with (
        patch("kiro_crew.agent_discovery.list_agents", return_value=list(_SPECS)),
        patch("kiro_crew.agent.kiro_agents_dir_path", return_value="/nowhere"),
    ):
        return members_mod._optin_candidates(cfg, usage)


class TestCandidates:
    def test_offers_only_unbound_user_specs(self):
        cfg = _cfg({"triage": KiroCrewAgentConfig(kiro_agent="bound-already")})
        names = [row["name"] for row in _candidates(cfg, {})]
        assert set(names) == {"radar", "scribe", "scout"}

    def test_a_registered_crew_name_is_not_offered(self):
        cfg = _cfg({"scout": KiroCrewAgentConfig(kiro_agent="kirocrew")})
        names = [row["name"] for row in _candidates(cfg, {})]
        assert "scout" not in names

    def test_used_first_then_recent_then_name(self):
        usage = {"scribe": (3, 100.0), "scout": (3, 200.0), "radar": (0, 0.0)}
        rows = _candidates(_cfg(), usage)
        assert [row["name"] for row in rows] == ["scout", "scribe", "radar"]
        by_name = {row["name"]: row for row in rows}
        assert by_name["scout"]["chats"] == 3
        assert by_name["scout"]["last_used_ts"] == 200.0
        assert by_name["radar"]["chats"] == 0

    def test_row_shape(self):
        row = _candidates(_cfg(), {})[0]
        assert set(row) == {"name", "description", "source", "chats", "last_used_ts"}
        assert row["source"] == "builtin"


def _app(state) -> web.Application:
    @web.middleware
    async def _auth(request: web.Request, handler):
        request.setdefault("app", "")
        request.setdefault("user", "local-app")
        return await handler(request)

    app = web.Application(middlewares=[_auth])
    app["state"] = state
    app.router.add_get("/api/members/optin", members_mod.api_members_optin)
    app.router.add_post("/api/members/optin/done", members_mod.api_members_optin_done)
    return app


class TestGet:
    @pytest.mark.asyncio
    async def test_state_and_candidates_in_one_read(self, tmp_path):
        state = _make_state(tmp_path)
        state.conversation_log = SimpleNamespace(agent_usage=lambda: {"radar": (2, 50.0)})
        cfg = _cfg({"triage": KiroCrewAgentConfig(kiro_agent="bound-already")})
        with (
            patch.object(members_mod.KiroCrewConfig, "load", return_value=cfg),
            patch("kiro_crew.agent_discovery.list_agents", return_value=list(_SPECS)),
            patch("kiro_crew.agent.kiro_agents_dir_path", return_value="/nowhere"),
        ):
            async with TestClient(TestServer(_app(state))) as client:
                resp = await client.get("/api/members/optin")
                assert resp.status == 200
                body = await resp.json()
        assert body["done"] is False
        assert body["crewmates"] == 1
        assert [c["name"] for c in body["candidates"]] == ["radar", "scout", "scribe"]
        assert body["candidates"][0]["chats"] == 2

    @pytest.mark.asyncio
    async def test_unreadable_history_degrades_to_zero_chats(self, tmp_path):
        state = _make_state(tmp_path)

        def _boom():
            raise OSError("history gone")

        state.conversation_log = SimpleNamespace(agent_usage=_boom)
        with (
            patch.object(members_mod.KiroCrewConfig, "load", return_value=_cfg()),
            patch("kiro_crew.agent_discovery.list_agents", return_value=list(_SPECS)),
            patch("kiro_crew.agent.kiro_agents_dir_path", return_value="/nowhere"),
        ):
            async with TestClient(TestServer(_app(state))) as client:
                body = await (await client.get("/api/members/optin")).json()
        assert body["candidates"] and all(c["chats"] == 0 for c in body["candidates"])

    @pytest.mark.asyncio
    async def test_app_tokens_are_denied(self, tmp_path):
        state = _make_state(tmp_path)

        @web.middleware
        async def _as_app(request: web.Request, handler):
            request["app"] = "some-app"
            return await handler(request)

        app = _app(state)
        app.middlewares.insert(0, _as_app)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/api/members/optin")).status == 404
            assert (await client.post("/api/members/optin/done")).status == 404


class TestDone:
    @pytest.mark.asyncio
    async def test_records_the_gate_through_a_delta_write(self, tmp_path):
        state = _make_state(tmp_path)
        doc: dict = {"dashboard": {"onboarded": True}, "agent": {"model": "auto"}}

        def _fake_update(*, mutate):
            mutate(doc)

        async def _owner(request, operation):
            return None

        with (
            patch.object(members_mod, "require_owner_dashboard_request", _owner),
            patch("kiro_crew.config.loader.update_config_locked", _fake_update),
        ):
            async with TestClient(TestServer(_app(state))) as client:
                resp = await client.post("/api/members/optin/done")
                assert resp.status == 200
                assert await resp.json() == {"ok": True, "done": True}
        # Exactly one key moved; the rest of the document is untouched.
        assert doc == {
            "dashboard": {"onboarded": True, "crewmate_optin_done": True},
            "agent": {"model": "auto"},
        }

    @pytest.mark.asyncio
    async def test_non_owner_is_refused_before_any_write(self, tmp_path):
        state = _make_state(tmp_path)

        async def _deny(request, operation):
            return web.json_response({"error": "forbidden"}, status=403)

        with (
            patch.object(members_mod, "require_owner_dashboard_request", _deny),
            patch.object(members_mod, "_persist_optin_done") as persist,
        ):
            async with TestClient(TestServer(_app(state))) as client:
                assert (await client.post("/api/members/optin/done")).status == 403
        persist.assert_not_called()
