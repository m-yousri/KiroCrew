"""Crewmate teams: the ``crew-teams/teams.json`` store and the ``/api/teams`` routes.

``KIROCREW_HOME`` is pinned per test by the autouse ``_isolate_kirocrew_home``
fixture in conftest, so every store read and write here lands in a temp home.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew import crew_teams as teams
from kiro_crew.config.loader import KiroCrewAgentConfig

RADAR = "Radar"
FIXER = "Fixer"
SCRIBE = "Scribe"
KNOWN = {RADAR, FIXER, SCRIBE}


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class TestTeamsStore:
    def test_absent_file_reads_as_no_teams(self):
        assert teams.read_teams() == []
        assert not teams.teams_path().exists()

    def test_create_persists_in_the_masked_directory(self):
        team = teams.create_team("Issue triage", [RADAR, FIXER], known=KNOWN)
        assert team.name == "Issue triage"
        assert team.members == [RADAR, FIXER]
        assert teams.teams_path().parent.name == teams.TEAMS_DIR_NAME
        stored = teams.read_teams()
        assert [t.to_dict() for t in stored] == [team.to_dict()]

    def test_name_is_stripped_and_bounded(self):
        team = teams.create_team("  Docs  ", [], known=KNOWN)
        assert team.name == "Docs"
        with pytest.raises(teams.TeamError) as exc:
            teams.create_team("   ", [], known=KNOWN)
        assert exc.value.code == "invalid_team_name"
        with pytest.raises(teams.TeamError) as exc:
            teams.create_team("x" * (teams.TEAM_NAME_MAX_CHARS + 1), [], known=KNOWN)
        assert exc.value.code == "team_name_too_long"

    def test_control_characters_and_lone_surrogates_are_refused(self):
        with pytest.raises(teams.TeamError):
            teams.create_team("Docs\x1b[2J", [], known=KNOWN)
        with pytest.raises(teams.TeamError):
            teams.create_team("Docs\u2028Two", [], known=KNOWN)
        with pytest.raises(teams.TeamError):
            teams.create_team("Docs\ud800", [], known=KNOWN)
        # A refused write leaves no file behind.
        assert teams.read_teams() == []

    def test_format_characters_and_no_break_space_are_names(self):
        """A ZWJ emoji sequence and a pasted no-break space are legitimate in a
        name; only control characters and line separators are refused."""
        name = "Ops\u00a0\U0001f469\u200d\U0001f4bb"
        assert teams.create_team(name, [], known=KNOWN).name == name

    def test_store_directory_is_masked_and_fenced(self):
        """The document is the owner's grouping of the crewmates it names, so the
        crewmates must not be able to rewrite it: masked from every sandboxed
        process (and pre-created so the mask is never vacuous) and fenced from
        agent file tools. Under ``trust/`` it would be sandbox read-write."""
        from kiro_crew import sandbox
        from kiro_crew.security import paths as security_paths

        leaf = teams.TEAMS_DIR_NAME
        assert leaf in sandbox._CREW_HIDDEN_LEAVES
        assert leaf in sandbox._CREW_PRECREATE_HIDDEN_DIR_LEAVES
        assert leaf in security_paths._CREW_SECRET_LEAVES
        assert "trust" not in teams.teams_path().parts

    def test_a_valid_write_can_never_outgrow_the_read_cap(self):
        """The largest document the caps allow must still read back; a write
        that would exceed the read cap is refused BEFORE it lands, so a valid
        sequence of writes cannot leave the store unreadable."""
        widest = "x" * 63
        assert teams.max_document_bytes() <= teams.TEAMS_FILE_MAX_BYTES
        crews = {f"{widest[:-4]}{i:04d}" for i in range(teams.TEAM_MEMBERS_MAX)}
        team = teams.create_team("w" * teams.TEAM_NAME_MAX_CHARS, sorted(crews), known=crews)
        assert len(teams.read_teams()[0].members) == teams.TEAM_MEMBERS_MAX
        # Shrink the cap under the current document: the next write is refused
        # with a coded error and the document on disk is untouched.
        before = teams.teams_path().read_bytes()
        with patch.object(teams, "TEAMS_FILE_MAX_BYTES", len(before) - 1):
            with pytest.raises(teams.TeamError) as exc:
                teams.update_team(team.id, name="renamed", known=crews)
        assert exc.value.code == "teams_too_large"
        assert teams.teams_path().read_bytes() == before

    def test_prune_unknown_drops_names_no_registry_holds(self):
        """A crew removed by a path that did not call drop_member (the CLI, a
        package prune) must never be ANSWERED as a member: the routes reconcile
        the document against the registry on every read."""
        team = teams.create_team("Issue triage", [RADAR, FIXER], known=KNOWN)
        pruned = teams.prune_unknown([team], {RADAR})
        assert pruned[0].members == [RADAR]
        # The document itself is left alone: reconciliation is a read-side view.
        assert teams.read_teams()[0].members == [RADAR, FIXER]

    def test_unknown_member_is_refused(self):
        with pytest.raises(teams.TeamError) as exc:
            teams.create_team("Docs", ["Ghost"], known=KNOWN)
        assert exc.value.code == "unknown_member"
        assert teams.read_teams() == []

    def test_a_crewmate_is_on_at_most_one_team(self):
        triage = teams.create_team("Issue triage", [RADAR, FIXER], known=KNOWN)
        docs = teams.create_team("Docs", [FIXER, SCRIBE], known=KNOWN)
        stored = {t.id: t for t in teams.read_teams()}
        # Creating Docs with Fixer moved Fixer OUT of Issue triage in the same write.
        assert stored[triage.id].members == [RADAR]
        assert stored[docs.id].members == [FIXER, SCRIBE]
        # Re-membering a team moves the newcomers out of their current team too.
        teams.update_team(triage.id, members=[RADAR, SCRIBE], known=KNOWN)
        stored = {t.id: t for t in teams.read_teams()}
        assert stored[triage.id].members == [RADAR, SCRIBE]
        assert stored[docs.id].members == [FIXER]

    def test_duplicate_members_collapse_in_order(self):
        team = teams.create_team("Docs", [SCRIBE, RADAR, SCRIBE], known=KNOWN)
        assert team.members == [SCRIBE, RADAR]

    def test_update_renames_without_touching_members(self):
        team = teams.create_team("Docs", [SCRIBE], known=KNOWN)
        renamed = teams.update_team(team.id, name="Documentation", known=KNOWN)
        assert renamed.name == "Documentation"
        assert renamed.members == [SCRIBE]
        with pytest.raises(teams.TeamError) as exc:
            teams.update_team(team.id, known=KNOWN)
        assert exc.value.code == "nothing_to_update"

    def test_unknown_team_id(self):
        with pytest.raises(teams.TeamError) as exc:
            teams.update_team("deadbeef0000", name="x", known=KNOWN)
        assert exc.value.code == "team_not_found"
        with pytest.raises(teams.TeamError) as exc:
            teams.delete_team("deadbeef0000")
        assert exc.value.code == "team_not_found"

    def test_delete_removes_only_that_team(self):
        triage = teams.create_team("Issue triage", [RADAR], known=KNOWN)
        docs = teams.create_team("Docs", [SCRIBE], known=KNOWN)
        teams.delete_team(triage.id)
        assert [t.id for t in teams.read_teams()] == [docs.id]

    def test_drop_member_removes_a_deleted_crew(self):
        team = teams.create_team("Issue triage", [RADAR, FIXER], known=KNOWN)
        assert teams.drop_member(FIXER) is True
        assert teams.read_teams()[0].members == [RADAR]
        assert teams.read_teams()[0].id == team.id
        # Nothing to drop: no write.
        assert teams.drop_member("Ghost") is False

    def test_document_carries_the_schema_version(self):
        teams.create_team("Docs", [SCRIBE], known=KNOWN)
        doc = json.loads(teams.teams_path().read_text(encoding="utf-8"))
        assert doc["version"] == teams.TEAMS_SCHEMA_VERSION
        # A document from a NEWER build is refused whole, never half-read.
        doc["version"] = teams.TEAMS_SCHEMA_VERSION + 1
        teams.teams_path().write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(teams.TeamsUnreadable):
            teams.read_teams()

    def test_drop_member_swallows_a_failed_rewrite(self, monkeypatch):
        """The caller is the crew-delete path, whose config write has already
        committed; a disk-full on crew-teams/ must not turn that into a 500."""
        teams.create_team("Issue triage", [RADAR, FIXER], known=KNOWN)

        def _boom(_teams):
            raise OSError("disk full")

        monkeypatch.setattr(teams, "write_teams", _boom)
        assert teams.drop_member(FIXER) is False
        monkeypatch.undo()
        # Nothing was lost: the document still names both crewmates.
        assert teams.read_teams()[0].members == [RADAR, FIXER]

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="msvcrt's contended acquire ignores the patched ceiling; the POSIX run pins the contract",
    )
    def test_writers_hold_a_cross_process_file_lock(self):
        """``kirocrew agents delete`` writes from another process, which the
        threading lock cannot reach: every read -> mutate -> rewrite also holds
        the advisory lock on the directory's lock file, and a write attempted
        while another holder has it fails closed instead of racing."""
        from kiro_crew import platform_compat

        teams.create_team("Issue triage", [RADAR], known=KNOWN)
        lock_path = teams.teams_path().parent / teams.LOCK_FILE_NAME
        assert lock_path.exists()
        with platform_compat.open_lock_file(lock_path) as fd:
            with platform_compat.file_lock(fd, exclusive=True, wait=True):
                # Single-shot from the holder's point of view: a second acquire
                # on the same file from this process would deadlock the test,
                # so exercise the store's fail-closed branch through a zero
                # ceiling rather than a real second process.
                with patch.object(platform_compat, "_LOCK_TIMEOUT_SECS", 0.2):
                    with pytest.raises(OSError):
                        teams.create_team("Docs", [SCRIBE], known=KNOWN)
        # The document is untouched by the refused write.
        assert [t.name for t in teams.read_teams()] == ["Issue triage"]

    def test_concurrent_writers_never_lose_an_update(self):
        """Every writer runs read -> mutate -> rewrite off-loop; without the
        store lock two interleaved creates would have the second drop the
        first's team from the document."""
        import threading

        errors: list[BaseException] = []

        def _create(i: int) -> None:
            try:
                teams.create_team(f"Team {i}", [], known=KNOWN)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=_create, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert sorted(t.name for t in teams.read_teams()) == sorted(f"Team {i}" for i in range(12))

    def test_unparseable_file_raises_rather_than_reading_empty(self):
        path = teams.teams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(teams.TeamsUnreadable):
            teams.read_teams()
        # A write path must not erase the unreadable document either.
        with pytest.raises(teams.TeamsUnreadable):
            teams.create_team("Docs", [], known=KNOWN)
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_foreign_document_keeps_first_team_for_a_doubled_member(self):
        path = teams.teams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"teams": [{"id": "aa", "name": "A", "members": ["Radar"]},'
            ' {"id": "bb", "name": "B", "members": ["Radar", "Fixer"]}]}',
            encoding="utf-8",
        )
        stored = teams.read_teams()
        assert stored[0].members == [RADAR]
        assert stored[1].members == [FIXER]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


def _make_app() -> web.Application:
    from kiro_crew.dashboard.handlers.teams import (
        api_teams_create,
        api_teams_delete,
        api_teams_list,
        api_teams_update,
    )

    @web.middleware
    async def _auth(request: web.Request, handler):
        if "app" not in request:
            request["app"] = request.headers.get("X-Test-App", "")
        return await handler(request)

    app = web.Application(middlewares=[_auth])
    app.router.add_get("/api/teams", api_teams_list)
    app.router.add_post("/api/teams", api_teams_create)
    app.router.add_put("/api/teams/{id}", api_teams_update)
    app.router.add_delete("/api/teams/{id}", api_teams_delete)
    return app


def _fake_config():
    from kiro_crew.config.loader import KiroCrewConfig

    cfg = KiroCrewConfig()
    cfg.agents = {name: KiroCrewAgentConfig(kiro_agent="kirocrew") for name in sorted(KNOWN)}
    return cfg


def _patched_config():
    return patch(
        "kiro_crew.dashboard.handlers.teams.KiroCrewConfig.load", return_value=_fake_config()
    )


def _as_owner():
    """The write routes are owner-only (``require_owner_dashboard_request``);
    the deny path is exercised by the repo-wide owner-gate invariant walk, so
    these tests patch the gate open to test the handlers' own contracts."""
    return patch(
        "kiro_crew.dashboard.handlers.teams.require_owner_dashboard_request",
        new=AsyncMock(return_value=None),
    )


class TestTeamsRoutes:
    @pytest.mark.asyncio
    async def test_list_is_empty_before_any_team(self):
        async with TestClient(TestServer(_make_app())) as client:
            resp = await client.get("/api/teams")
            assert resp.status == 200
            assert await resp.json() == {"teams": []}

    @pytest.mark.asyncio
    async def test_create_update_delete_round_trip(self):
        with _as_owner(), _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.post(
                    "/api/teams", json={"name": "Issue triage", "members": [RADAR, FIXER]}
                )
                assert resp.status == 201
                team = (await resp.json())["team"]
                assert team["members"] == [RADAR, FIXER]

                resp = await client.put(
                    f"/api/teams/{team['id']}", json={"name": "Triage", "members": [RADAR]}
                )
                assert resp.status == 200
                assert (await resp.json())["team"] == {
                    "id": team["id"],
                    "name": "Triage",
                    "members": [RADAR],
                }

                resp = await client.get("/api/teams")
                assert (await resp.json())["teams"][0]["name"] == "Triage"

                resp = await client.delete(f"/api/teams/{team['id']}")
                assert resp.status == 200
                resp = await client.get("/api/teams")
                assert await resp.json() == {"teams": []}

    @pytest.mark.asyncio
    async def test_list_never_names_a_crew_the_registry_lacks(self):
        teams.create_team("Issue triage", [RADAR, "Ghost"], known=KNOWN | {"Ghost"})
        with _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.get("/api/teams")
                assert resp.status == 200
                assert (await resp.json())["teams"][0]["members"] == [RADAR]

    @pytest.mark.asyncio
    async def test_unknown_member_is_400_with_code(self):
        with _as_owner(), _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.post("/api/teams", json={"name": "Docs", "members": ["Ghost"]})
                assert resp.status == 400
                assert (await resp.json())["code"] == "unknown_member"

    @pytest.mark.asyncio
    async def test_bad_bodies_are_coded_400s(self):
        with _as_owner(), _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.post("/api/teams", data="not json")
                assert resp.status == 400
                assert (await resp.json())["code"] == "invalid_json"
                resp = await client.post("/api/teams", json=["a", "list"])
                assert resp.status == 400
                assert (await resp.json())["code"] == "invalid_json"
                resp = await client.post("/api/teams", json={"members": []})
                assert resp.status == 400
                assert (await resp.json())["code"] == "invalid_team_name"
                resp = await client.put("/api/teams/not-hex!", json={"name": "x"})
                assert resp.status == 400
                assert (await resp.json())["code"] == "invalid_team_id"

    @pytest.mark.asyncio
    async def test_unknown_team_is_404(self):
        with _as_owner(), _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.put("/api/teams/deadbeef0000", json={"name": "x"})
                assert resp.status == 404
                assert (await resp.json())["code"] == "team_not_found"
                resp = await client.delete("/api/teams/deadbeef0000")
                assert resp.status == 404

    @pytest.mark.asyncio
    async def test_app_callers_are_denied_with_404(self):
        teams.create_team("Docs", [SCRIBE], known=KNOWN)
        headers = {"X-Test-App": "some-app"}
        with _as_owner(), _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.get("/api/teams", headers=headers)
                assert resp.status == 404
                assert "Docs" not in await resp.text()
                resp = await client.post(
                    "/api/teams", json={"name": "X", "members": []}, headers=headers
                )
                assert resp.status == 404
        assert len(teams.read_teams()) == 1

    @pytest.mark.asyncio
    async def test_non_owner_write_is_refused_before_validation(self):
        # A fresh Response per call: an aiohttp response can be sent once, and
        # returning one prepared object to two requests hangs the second.
        def _refuse(*_a, **_k):
            return web.json_response({"error": "forbidden", "code": "owner_only"}, status=403)

        with (
            patch(
                "kiro_crew.dashboard.handlers.teams.require_owner_dashboard_request",
                new=AsyncMock(side_effect=_refuse),
            ),
            _patched_config(),
        ):
            async with TestClient(TestServer(_make_app())) as client:
                # An INVALID body still gets the gate's answer, never a 400 that
                # tells a non-owner what would have validated.
                resp = await client.post("/api/teams", data="not json")
                assert resp.status == 403
                resp = await client.delete("/api/teams/not-hex!")
                assert resp.status == 403
        assert teams.read_teams() == []

    @pytest.mark.asyncio
    async def test_unreadable_document_is_500_not_empty_list(self):
        path = teams.teams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with _as_owner(), _patched_config():
            async with TestClient(TestServer(_make_app())) as client:
                resp = await client.get("/api/teams")
                assert resp.status == 500
                assert (await resp.json())["code"] == "teams_unreadable"
                resp = await client.post("/api/teams", json={"name": "X", "members": []})
                assert resp.status == 500
        assert path.read_text(encoding="utf-8") == "{not json"
