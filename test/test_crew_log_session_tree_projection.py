"""The session tree projection -- one test per promise that makes it safe, not fast.

Speed is the easy half and is not what these tests defend. The projection replaces a
scan with in-memory state, so the properties a second implementation would be free to
break are the ones that decide whether the state is ever WRONG: a read does no I/O at
all, the checkpoint is a shortcut that is discarded rather than trusted when it does not
match this build, the tail replay pays for the delta rather than the store, and the
emitter advances the fold only after the append that justifies it has succeeded.

The zero-I/O claims are asserted by COUNTING calls to the store's own readers rather
than by timing anything: a timing test would pass on a machine fast enough to hide a
scan, which is exactly the regression worth catching.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro_crew import crew_log as lg
from kiro_crew.crew_log import CrewLog, emit
from kiro_crew.crew_log import session_tree_projection as stp
from kiro_crew.crew_log import store as crew_store
from kiro_crew.crew_log.session_tree import OpenedRecord
from kiro_crew.crew_log.session_tree_projection import (
    CHECKPOINT_NAME,
    CHECKPOINT_VERSION,
    SessionTreeProjection,
)
from kiro_crew.session_ledger import _store_name

GATEWAY = "gateway"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test writes into its own data home, never the live one.

    The process-wide projection is dropped too: it is bound to one store, so a test
    inheriting the previous test's fold would be reading another store's records.
    """
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    # The emitter is inert without the flag, and the hook tests below exercise the
    # emitter's real path rather than a hand-written entry.
    monkeypatch.setenv(emit.CREW_LOG_ENV, "1")
    stp.reset_for_tests()
    yield
    stp.reset_for_tests()


def _rec(sid: str, slot: str, created: int = 1, parent: str | None = None) -> OpenedRecord:
    return OpenedRecord(sid=sid, slot=slot, created_at=created, parent_slot=parent)


def _log(sid: str, slot: str) -> CrewLog:
    return CrewLog.create(lg.KIND_SESSION, sid, owner="raymond", agent="kirocrew", slot=slot)


def _opened(handle: CrewLog, slot: str, *, parent: dict[str, str] | None = None) -> None:
    data = {
        "agent": "kirocrew",
        "slot": slot,
        "model": "opus",
        "cwd": "/w",
        "owner": "raymond",
        "resumed": False,
    }
    if parent is not None:
        data["parent"] = parent
    handle.append("session/opened", data, src=GATEWAY)


def _write_unit(sid: str, slot: str, *, parent: str | None = None) -> None:
    """A real, announced unit on disk, written the way the GATEWAY writes one.

    Through ``emit.on_session_opened`` rather than a hand-built append, because that
    is the path carrying the projection hook -- a test that wrote the entry directly
    would prove the fold and nothing about what advances it.
    """
    emit.on_session_opened(
        sid,
        agent="kirocrew",
        slot=slot,
        model="claude-opus-5",
        cwd="/w",
        owner="raymond",
        parent_slot=parent or "",
    )


def _checkpoint_path() -> Path:
    from kiro_crew.crew_log.store import crew_log_root

    return crew_log_root(lg.KIND_SESSION) / "projections" / CHECKPOINT_NAME


def _remove_unit(sid: str) -> str:
    """``remove_unit`` with the guard it requires. Unconditional here: the guard is
    retention's own re-check of expiry, which these tests are not exercising."""
    return crew_store.remove_unit(lg.KIND_SESSION, sid, guard=lambda _directory: True)


def _write_unit_unheld(sid: str, slot: str, *, parent: str | None = None) -> None:
    """A real announced unit whose lease nobody holds, for the removal tests.

    ``remove_unit`` asks for SOLE ownership and is refused while any handle that wrote
    still exists -- and the emitter keeps its handle, which is correct for a live
    session and is exactly what retention waits out. So a removal test writes the unit
    directly and drops the handle, rather than asserting a refusal it did not mean to
    set up.
    """
    handle = _log(sid, slot)
    _opened(handle, slot, parent={"slot": parent} if parent else None)
    del handle


class _StoreSpy:
    """Counts the store reads a projection makes. Zero is the claim under test."""

    def __init__(self, monkeypatch) -> None:
        self.unit_dirs = 0
        self.oldest_segment = 0
        self.read_head = 0
        self.iterdir = 0
        real_unit_dirs = crew_store.unit_dirs
        real_oldest = crew_store.oldest_segment
        real_head = crew_store.read_head

        def spy_unit_dirs(*a, **k):
            self.unit_dirs += 1
            return real_unit_dirs(*a, **k)

        def spy_oldest(*a, **k):
            self.oldest_segment += 1
            return real_oldest(*a, **k)

        def spy_head(*a, **k):
            self.read_head += 1
            return real_head(*a, **k)

        monkeypatch.setattr(crew_store, "unit_dirs", spy_unit_dirs)
        monkeypatch.setattr(crew_store, "oldest_segment", spy_oldest)
        monkeypatch.setattr(crew_store, "read_head", spy_head)

    @property
    def total(self) -> int:
        return self.unit_dirs + self.oldest_segment + self.read_head


# ── apply, forget, and the same-reference rule ─────────────────────────────


def test_apply_adds_an_edge_without_touching_the_store(monkeypatch):
    """The whole point: a delta folded in memory, with no disk access at all."""
    spy = _StoreSpy(monkeypatch)
    proj = SessionTreeProjection()

    proj.apply(_rec("s-parent", "slot-a"))
    proj.apply(_rec("s-child", "slot-b", created=2, parent="slot-a"))
    nodes = proj.nodes()

    assert nodes["slot-b"].parent_slot == "slot-a"
    assert spy.total == 0, "a projection read the store; it is supposed to be pure memory"


def test_a_parentless_record_is_what_lets_a_child_edge_be_followed(monkeypatch):
    """A root's own record is load-bearing, so the hook cannot skip parentless entries.

    ``fold_tree`` follows an edge only when the cited parent slot ``has_log`` -- and a
    root conductor session's record is the ONLY thing that puts its slot there. Drop
    it and the common case (a root session that opens children) silently renders every
    child as a root, which is the feature failing rather than degrading.
    """
    proj = SessionTreeProjection()
    proj.apply(_rec("s-child", "slot-b", created=2, parent="slot-a"))

    # The parent's own record has not arrived: a citation, not yet a place in the tree.
    assert proj.nodes()["slot-b"].parent_slot == "slot-a"
    assert "slot-a" not in proj.nodes()

    proj.apply(_rec("s-parent", "slot-a", created=1))
    assert "slot-a" in proj.nodes(), "the parentless record must establish the parent slot"


def test_forget_drops_the_record(monkeypatch):
    spy = _StoreSpy(monkeypatch)
    proj = SessionTreeProjection()
    proj.apply(_rec("s-parent", "slot-a"))
    proj.apply(_rec("s-child", "slot-b", created=2, parent="slot-a"))

    proj.forget("s-child")

    assert "slot-b" not in proj.nodes()
    assert "slot-a" in proj.nodes()
    assert spy.total == 0


def test_forget_is_silent_about_a_record_it_does_not_hold():
    proj = SessionTreeProjection()
    proj.forget("never-seen")
    proj.forget("")
    assert proj.nodes() == {}


def test_nodes_returns_the_same_object_while_unchanged():
    """dsh's same-reference rule: a timer-driven reader does no work and may compare
    by identity."""
    proj = SessionTreeProjection()
    proj.apply(_rec("s-a", "slot-a"))

    first = proj.nodes()
    assert proj.nodes() is first

    # A record already held is not a change, so it must not invalidate the fold.
    proj.apply(_rec("s-a", "slot-a"))
    assert proj.nodes() is first

    # A real delta must.
    proj.apply(_rec("s-b", "slot-b", created=2, parent="slot-a"))
    assert proj.nodes() is not first


def test_a_record_without_a_sid_is_dropped():
    """The state is keyed by sid, so a blank key would collide every such record."""
    proj = SessionTreeProjection()
    proj.apply(_rec("", "slot-a"))
    assert proj.nodes() == {}


# ── the checkpoint ─────────────────────────────────────────────────────────


def test_checkpoint_round_trip():
    """A checkpoint written by one process seeds the next one with the same tree."""
    _write_unit("s-parent", "slot-a")
    _write_unit("s-child", "slot-b", parent="slot-a")

    proj = SessionTreeProjection()
    proj.ensure_seeded()
    assert proj.flush_checkpoint() is True

    payload = json.loads(_checkpoint_path().read_text(encoding="utf-8"))
    assert payload["ver"] == CHECKPOINT_VERSION
    assert {row["sid"] for row in payload["records"]} == {"s-parent", "s-child"}

    revived = SessionTreeProjection()
    revived.ensure_seeded()
    assert revived.nodes()["slot-b"].parent_slot == "slot-a"


def test_a_checkpoint_record_whose_unit_is_gone_is_dropped_on_revival():
    """The checkpoint is a shortcut, never an authority: it cannot resurrect a unit
    that is absent from the store."""
    _write_unit("s-parent", "slot-a")
    proj = SessionTreeProjection()
    proj.ensure_seeded()
    proj.apply(_rec("s-ghost", "slot-ghost", created=9, parent="slot-a"))
    assert proj.flush_checkpoint() is True

    revived = SessionTreeProjection()
    revived.ensure_seeded()
    assert "slot-ghost" not in revived.nodes()
    assert "slot-a" in revived.nodes()


def test_ver_mismatch_discards_the_checkpoint_and_rebuilds_cold(monkeypatch):
    """A checkpoint from another build is DISCARDED, never migrated."""
    _write_unit("s-parent", "slot-a")
    _write_unit("s-child", "slot-b", parent="slot-a")

    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ver": CHECKPOINT_VERSION + 1, "records": [{"sid": "ghost", "slot": "gone"}]}),
        encoding="utf-8",
    )

    proj = SessionTreeProjection()
    proj.ensure_seeded()

    nodes = proj.nodes()
    assert "gone" not in nodes, "a foreign-version checkpoint was trusted"
    assert nodes["slot-b"].parent_slot == "slot-a", "the cold rebuild did not run"


def test_an_unparseable_checkpoint_rebuilds_rather_than_raising():
    _write_unit("s-parent", "slot-a")
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    proj = SessionTreeProjection()
    proj.ensure_seeded()
    assert "slot-a" in proj.nodes()


def test_a_malformed_row_is_dropped_not_coerced():
    """A row this cannot READ costs a re-read of that unit's head. A row COERCED into
    the wrong shape would instead be folded as though it had been read."""
    _write_unit("good", "slot-a")
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ver": CHECKPOINT_VERSION,
                "records": [
                    {"sid": "good", "slot": "slot-a", "at": 1},
                    {"sid": "bad-at", "slot": "slot-b", "at": "not-an-int"},
                    {"slot": "no-sid", "at": 1},
                    "not-a-dict",
                ],
            }
        ),
        encoding="utf-8",
    )
    proj = SessionTreeProjection()
    proj.ensure_seeded()
    nodes = proj.nodes()
    assert "slot-a" in nodes
    assert "slot-b" not in nodes


# ── the tail replay ────────────────────────────────────────────────────────


def test_tail_replay_reads_heads_only_for_the_delta(monkeypatch):
    """The replay pays for the GAP, not for the history.

    Three units are on disk and two are in the checkpoint, so exactly ONE head may be
    read. A replay that re-read the store would read three.
    """
    _write_unit("s-parent", "slot-a")
    _write_unit("s-child", "slot-b", parent="slot-a")

    seed = SessionTreeProjection()
    seed.apply(_rec("s-parent", "slot-a"))
    seed.apply(_rec("s-child", "slot-b", created=2, parent="slot-a"))
    assert seed.flush_checkpoint() is True

    # Written AFTER the checkpoint -- the crash-gap case the replay exists for.
    _write_unit("s-late", "slot-c", parent="slot-a")

    spy = _StoreSpy(monkeypatch)
    proj = SessionTreeProjection()
    proj.ensure_seeded()

    assert proj.nodes()["slot-c"].parent_slot == "slot-a"
    assert spy.read_head == 1, f"expected one head read for the delta, got {spy.read_head}"
    assert spy.unit_dirs == 0, "the replay called the scanner's unit walk"


def test_a_resumed_opener_does_not_retract_its_creator_edge():
    """A resume appends its OWN ``session/opened``, and that entry need not repeat the
    creator the first one named. Replacing outright would drop the edge on resume and
    leave the session rendering as a root until the process re-seeds from disk.

    Not the reverse of a real change: the crew log has no entry meaning "this session
    was not opened by anyone after all", so nothing legitimately retracts a citation.
    """
    proj = SessionTreeProjection()
    proj.apply(_rec("s-child", "slot-b", created=1, parent="slot-a"))
    proj.apply(_rec("s-parent", "slot-a"))
    assert proj.nodes()["slot-b"].parent_slot == "slot-a"

    # The resume: same session, same slot, no creator named this time.
    proj.apply(_rec("s-child", "slot-b", created=2, parent=None))

    assert proj.nodes()["slot-b"].parent_slot == "slot-a", "the resume retracted the edge"


def test_a_removal_during_the_scan_is_not_resurrected_by_the_seed(monkeypatch):
    """The cold scan runs OFF the lock, so a unit can be deleted after it was listed.

    Installing the scan's copy would put that record back and nothing would take it out
    again: the emitter fires once per opening, and the ``forget`` that accompanied the
    deletion ran against a state that was still empty, so it had nothing to pop.
    """
    from kiro_crew.crew_log.session_tree import SessionTree, TreeReading, fold_tree

    _write_unit_unheld("s-parent", "slot-a")
    _write_unit_unheld("s-child", "slot-b", parent="slot-a")

    proj = SessionTreeProjection()
    parent = _rec("s-parent", "slot-a")
    child = _rec("s-child", "slot-b", created=2, parent="slot-a")

    def scan_then_delete(_self, live_sids=()):
        # The deletion lands AFTER this scan listed the unit. That ordering is the whole
        # race: the reading below still carries a record whose unit is already deleted.
        proj.forget("s-child")
        return TreeReading(
            nodes=fold_tree([parent, child]),
            incomplete=False,
            records=(parent, child),
        )

    monkeypatch.setattr(SessionTree, "reading", scan_then_delete)
    proj.ensure_seeded()

    nodes = proj.nodes()
    assert "slot-b" not in nodes, "a unit removed during the scan was put back by the seed"
    assert "slot-a" in nodes


def test_tail_replay_drops_a_removed_unit(monkeypatch):
    _write_unit("s-parent", "slot-a")
    # Lease-free: this unit's DIRECTORY is deleted below, and Windows refuses to unlink
    # a file another handle still has open -- the lease a live handle holds is exactly
    # such a file.
    _write_unit_unheld("s-gone", "slot-b", parent="slot-a")

    seed = SessionTreeProjection()
    seed.apply(_rec("s-parent", "slot-a"))
    seed.apply(_rec("s-gone", "slot-b", created=2, parent="slot-a"))
    assert seed.flush_checkpoint() is True

    import shutil

    from kiro_crew.crew_log.store import crew_log_root

    shutil.rmtree(crew_log_root(lg.KIND_SESSION) / _store_name("s-gone"))

    proj = SessionTreeProjection()
    proj.ensure_seeded()

    nodes = proj.nodes()
    assert "slot-b" not in nodes, "a unit that is gone from disk stayed in the fold"
    assert "slot-a" in nodes


def test_a_read_after_seeding_does_no_io_at_all(monkeypatch):
    """The regression that matters: seeding is once per process, reads are free."""
    _write_unit("s-parent", "slot-a")
    _write_unit("s-child", "slot-b", parent="slot-a")

    proj = SessionTreeProjection()
    proj.ensure_seeded()

    spy = _StoreSpy(monkeypatch)
    for _ in range(50):
        proj.nodes()
        proj.reading()
        proj.ensure_seeded()

    assert spy.total == 0, "a poll-shaped read touched the store"


def test_a_moved_data_home_re_seeds_instead_of_serving_the_old_store(tmp_path, monkeypatch):
    """The fold is the image of ONE store; a home that moves must not inherit it."""
    _write_unit("s-parent", "slot-a")
    proj = SessionTreeProjection()
    proj.ensure_seeded()
    assert "slot-a" in proj.nodes()

    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "other-home"))
    _write_unit("s-other", "slot-z")
    proj.ensure_seeded()

    nodes = proj.nodes()
    assert "slot-a" not in nodes, "the previous store's records were served for a new home"
    assert "slot-z" in nodes


def test_no_checkpoint_seeds_from_one_cold_scan():
    _write_unit("s-parent", "slot-a")
    _write_unit("s-child", "slot-b", parent="slot-a")
    assert not _checkpoint_path().exists()

    proj = SessionTreeProjection()
    proj.ensure_seeded()

    assert proj.nodes()["slot-b"].parent_slot == "slot-a"


def test_an_empty_store_seeds_to_an_empty_tree():
    proj = SessionTreeProjection()
    proj.ensure_seeded()
    assert proj.nodes() == {}
    assert proj.reading().incomplete is False


# ── the emitter hook ───────────────────────────────────────────────────────


def test_the_emitter_applies_one_record_per_opened_entry(monkeypatch):
    """The hook fires once per ``session/opened``, carrying the parent when there is
    one, and the record it applies is built from what was just WRITTEN."""
    applied: list[OpenedRecord] = []
    real_apply = SessionTreeProjection.apply

    def spy_apply(self, record):
        applied.append(record)
        return real_apply(self, record)

    monkeypatch.setattr(SessionTreeProjection, "apply", spy_apply)

    _write_unit("s-parent", "slot-a")
    assert len(applied) == 1
    assert applied[0].sid == "s-parent"
    assert applied[0].slot == "slot-a"
    assert applied[0].parent_slot is None

    _write_unit("s-child", "slot-b", parent="slot-a")
    assert len(applied) == 2, "the hook did not fire exactly once for the second entry"
    assert applied[1].sid == "s-child"
    assert applied[1].parent_slot == "slot-a"

    # The live projection now holds the edge with no scan behind it.
    assert stp.projection().nodes()["slot-b"].parent_slot == "slot-a"


def test_a_warm_reuse_writes_no_entry_and_applies_nothing(monkeypatch):
    """The emitter is silent on a warm reuse of a session that already has a log, so
    the hook must be silent too: once per opened ENTRY, not once per claim."""
    applied: list[OpenedRecord] = []
    real_apply = SessionTreeProjection.apply

    def spy_apply(self, record):
        applied.append(record)
        return real_apply(self, record)

    monkeypatch.setattr(SessionTreeProjection, "apply", spy_apply)

    _write_unit("s-a", "slot-a")
    assert len(applied) == 1

    _write_unit("s-a", "slot-a")
    assert len(applied) == 1, "a warm reuse re-applied a record"


def test_the_hook_does_not_fire_for_other_entry_kinds(monkeypatch):
    applied: list[OpenedRecord] = []
    real_apply = SessionTreeProjection.apply

    def spy_apply(self, record):
        applied.append(record)
        return real_apply(self, record)

    monkeypatch.setattr(SessionTreeProjection, "apply", spy_apply)

    _write_unit("s-a", "slot-a")
    assert len(applied) == 1

    handle = CrewLog.open(lg.KIND_SESSION, "s-a")
    handle.append("turn/started", {"turn": 1, "actor": "user", "depth": 0}, src=GATEWAY)
    handle.append("session/closed", {"reason": "done"}, src=GATEWAY)
    assert len(applied) == 1, "a non-opened entry advanced the projection"


def test_the_record_is_applied_only_after_the_append_landed(monkeypatch):
    """Durability first, then memory: when the fold is advanced, the entry that
    justifies it is ALREADY on disk. Ordered the other way, an append that then failed
    would leave a creator edge no log records."""
    seen_on_disk: list[bool] = []
    real_apply = SessionTreeProjection.apply

    def spy_apply(self, record):
        from kiro_crew.crew_log.store import crew_log_root

        directory = crew_log_root(lg.KIND_SESSION) / _store_name(record.sid)
        segment = crew_store.oldest_segment(directory)
        body = segment.read_text(encoding="utf-8") if segment is not None else ""
        seen_on_disk.append("session/opened" in body)
        return real_apply(self, record)

    monkeypatch.setattr(SessionTreeProjection, "apply", spy_apply)

    _write_unit("s-a", "slot-a")

    assert seen_on_disk == [True], "the projection was advanced before the append landed"


def test_a_projection_failure_does_not_fail_the_session_open(monkeypatch):
    """An append that already succeeded must not be reported as failed because the
    memory image of it did not land."""

    def boom(self, record):
        raise RuntimeError("projection is broken")

    monkeypatch.setattr(SessionTreeProjection, "apply", boom)

    _write_unit("s-a", "slot-a")  # must not raise

    from kiro_crew.crew_log.store import crew_log_root

    assert (crew_log_root(lg.KIND_SESSION) / _store_name("s-a")).is_dir()


# ── removal ────────────────────────────────────────────────────────────────


def test_removing_a_unit_forgets_its_record():
    """``remove_unit`` is the one point a unit is established as gone, so the fold
    drops it there rather than waiting for a cold start."""
    _write_unit_unheld("s-parent", "slot-a")
    _write_unit_unheld("s-child", "slot-b", parent="slot-a")

    proj = stp.projection()
    proj.ensure_seeded()
    assert "slot-b" in proj.nodes()

    status = _remove_unit("s-child")

    assert status == crew_store.REMOVE_REMOVED
    assert "slot-b" not in proj.nodes(), "a removed unit's edge survived in memory"
    assert "slot-a" in proj.nodes()


def test_a_removal_that_did_not_happen_keeps_the_record():
    """``REMOVE_ABSENT`` is not a removal, so nothing is forgotten on the strength of
    it -- the fold must not drop an edge because a sweep aimed at the wrong id."""
    _write_unit_unheld("s-parent", "slot-a")
    proj = stp.projection()
    proj.ensure_seeded()

    status = _remove_unit("never-existed")

    assert status == crew_store.REMOVE_ABSENT
    assert "slot-a" in proj.nodes()
