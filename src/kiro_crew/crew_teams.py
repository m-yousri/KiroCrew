"""Crewmate teams: ``$KIROCREW_HOME/crew-teams/teams.json``.

Named ``crew_teams`` because ``kiro_crew.teams`` is the Microsoft Teams channel
package; a same-named module would be shadowed by it.

A team is a name plus an ordered list of crewmates (exact crew names). It is
the manager's grouping of the roster, nothing more: no memory, no chat, no
prompt of its own. The Crewmates page groups the roster by team and opens a
team view for each; everything in that view is derived from the members'
existing threads, activity and presence.

Every mutation runs read -> mutate -> rewrite under TWO locks: a process-wide
``threading.Lock`` (``_WRITE_LOCK``) and, inside it, an advisory file lock on
``crew-teams/.lock`` (``platform_compat.file_lock``). The gateway's writers (the
owner routes, the crew-delete route, the package-sync prune) run off-loop in
worker threads; ``kirocrew agents delete`` is a writer in ANOTHER process. Two
interleaved rewrites of one document would otherwise have the second silently
drop the first's change, and only the file lock reaches the CLI. Writers hold
the lock for a sub-second read plus an atomic rename, so the default ceiling
applies; a contended acquire past it fails closed (``OSError``, answered as
``teams_write_failed``) rather than writing unserialized.

Two invariants the store enforces so no UI has to:

* **A crewmate is on at most one team.** Assigning a crewmate to a team removes
  it from whichever team held it, in the same write, so two readers can never
  see it in two groups.
* **The file is one document.** Every mutation rewrites the whole list
  atomically (unique temp file + rename), so a torn record is never observable.

The file lives in its own data-home directory, ``crew-teams/``, and NOT under
``trust/``: ``trust`` is a declared sandbox READ-WRITE exception (in-sandbox code
appends to the audit log there), so a record under it stays writable by a
sandboxed command that builds the path at runtime -- the ``crew-panels`` lesson
in ``sandbox.py``. ``crew-teams`` is instead masked from every sandboxed process
(``sandbox._CREW_HIDDEN_LEAVES``, pre-created before each spawn so the mask is
never vacuous) and fenced from agent file tools (``security._CREW_SECRET_LEAVES``).
Only the gateway, on a human dashboard action, opens it. A crewmate's own tools
therefore cannot re-team itself or its siblings, and the team view is a human
surface built on human choices.

Member NAMES, not slugs, are stored: slugification is lossy (two names can
share one slug), and the roster is keyed and selected by exact name.
"""

from __future__ import annotations

import contextlib
import json
import logging
import secrets
import threading
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from kiro_crew import platform_compat
from kiro_crew.atomic_write import atomic_write, fsync_dir, read_bytes_with_retry
from kiro_crew.config.paths import data_home

logger = logging.getLogger(__name__)

#: Data-home directory holding the document. A DIRECTORY leaf, not a file, so the
#: mask also covers the sibling temp ``atomic_write`` renames into place. The same
#: string is listed in ``sandbox._CREW_HIDDEN_LEAVES``,
#: ``sandbox._CREW_PRECREATE_HIDDEN_DIR_LEAVES`` and ``security._CREW_SECRET_LEAVES``.
TEAMS_DIR_NAME = "crew-teams"

#: The one document inside it.
TEAMS_FILE_NAME = "teams.json"

#: Document schema version, written on every save. Matches the convention of
#: the other gateway-owned documents so a later shape change keys on it instead
#: of inferring v1 from absence. A document carrying a NEWER version is
#: refused (``TeamsUnreadable``) rather than half-read.
TEAMS_SCHEMA_VERSION = 1

#: Serializes every read -> mutate -> rewrite of the document in this process.
#: The cross-process half is the advisory lock on :data:`LOCK_FILE_NAME`.
_WRITE_LOCK = threading.Lock()

#: Lock file beside the document; content is never meaningful, only the lock.
LOCK_FILE_NAME = ".lock"

#: Hard cap on a team name. Enforced on write and refused loudly, never
#: truncated: a name is what the roster header shows.
TEAM_NAME_MAX_CHARS = 80

#: Hard cap on how many teams one roster can hold. A bound on the document,
#: not a product limit anyone is expected to reach.
TEAMS_MAX = 200

#: Hard cap on the crewmates one team lists. Same posture as ``TEAMS_MAX``.
TEAM_MEMBERS_MAX = 500

#: Bytes a document may occupy. Enforced on WRITE (a save that would exceed it
#: is refused with ``teams_too_large`` and the document on disk is untouched)
#: and on READ (a larger file was not written by this module). Sized above the
#: largest document the caps allow -- see :func:`max_document_bytes` and the
#: test that pins the inequality -- so a valid sequence of writes can never
#: produce a file the read refuses.
TEAMS_FILE_MAX_BYTES = 40_000_000


def max_document_bytes() -> int:
    """Upper bound on a document every cap admits, in UTF-8 JSON bytes.

    Names are capped in CHARACTERS: a team name at ``TEAM_NAME_MAX_CHARS`` and a
    crew name at the 64 the agent-name grammar allows, each up to 4 bytes per
    character and up to 6 for a JSON escape. The arithmetic is deliberately
    loose (every entry at its widest, every character escaped) because the point
    is the inequality with :data:`TEAMS_FILE_MAX_BYTES`, not the exact size.
    """
    per_char = 6  # a JSON escape of a BMP character is 6 bytes; 4-byte UTF-8 is less
    member = 64 * per_char + 4  # quotes, comma, space
    team = 64 + TEAM_NAME_MAX_CHARS * per_char + TEAM_MEMBERS_MAX * member  # keys + id + name
    return 64 + TEAMS_MAX * team


class TeamError(ValueError):
    """A team operation was refused; ``code`` is the machine-readable reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TeamsUnreadable(RuntimeError):
    """``crew-teams/teams.json`` exists but cannot be read or parsed.

    Propagates rather than degrading to "no teams": a read that answered an
    empty list here would let the next write erase every team the user made.
    """


@dataclass
class Team:
    id: str
    name: str
    members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "members": list(self.members)}


def teams_path() -> Path:
    """Absolute path to the teams document in its masked directory.

    Does NOT create the directory; :func:`write_teams` does on demand (the
    sandbox launcher pre-creates it empty before every spawn, so on a host
    that has spawned an agent it already exists).
    """
    return (data_home() / TEAMS_DIR_NAME / TEAMS_FILE_NAME).resolve()


@contextlib.contextmanager
def _document_lock() -> Iterator[None]:
    """Hold both locks around one read -> mutate -> rewrite.

    The directory is created here (owner-only, same as :func:`write_teams`)
    because the lock file has to exist before the document does: the first
    team the owner makes is itself a locked write.
    """
    path = teams_path()
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            platform_compat.restrict_dir_to_owner(path.parent)
        except OSError:
            logger.debug("could not tighten mode on %s", path.parent, exc_info=True)
        with platform_compat.open_lock_file(path.parent / LOCK_FILE_NAME) as fd:
            with platform_compat.file_lock(fd, exclusive=True, wait=True):
                yield


def _new_team_id() -> str:
    # 12 hex chars: URL-safe, not guessable from the name, and short enough to
    # ride the page's ``?team=`` query parameter.
    return secrets.token_hex(6)


def validate_team_name(name: object) -> str:
    """Return the stripped name, or raise :class:`TeamError`."""
    if not isinstance(name, str):
        raise TeamError("invalid_team_name", "team name must be a string")
    stripped = name.strip()
    if not stripped:
        raise TeamError("invalid_team_name", "team name must not be empty")
    if len(stripped) > TEAM_NAME_MAX_CHARS:
        raise TeamError("team_name_too_long", f"team name exceeds {TEAM_NAME_MAX_CHARS} characters")
    # JSON allows escaped lone surrogates; UTF-8 does not. Refuse them here
    # rather than letting the write raise mid-flight.
    try:
        stripped.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TeamError(
            "invalid_team_name", "team name contains characters that cannot be encoded"
        ) from exc
    # Refuse CONTROL characters (Cc: C0/C1, the escape sequences a terminal acts
    # on) and the two Unicode line/paragraph separators, which break a one-line
    # header. Judged by category, not ``isprintable()``: that also rejects format
    # characters (Cf) such as the zero-width joiner inside an emoji sequence, and
    # the no-break space (Zs), both legitimate in a name a user pasted.
    if any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in stripped):
        raise TeamError("invalid_team_name", "team name contains control characters")
    return stripped


def _parse_teams(raw: bytes) -> list[Team]:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TeamsUnreadable(f"{TEAMS_FILE_NAME} is not valid JSON") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("teams"), list):
        raise TeamsUnreadable(f"{TEAMS_FILE_NAME} does not hold a team list")
    version = doc.get("version", TEAMS_SCHEMA_VERSION)
    if not isinstance(version, int) or version > TEAMS_SCHEMA_VERSION:
        raise TeamsUnreadable(
            f"{TEAMS_FILE_NAME} has schema version {version!r}, newer than this build reads"
        )
    teams: list[Team] = []
    seen_ids: set[str] = set()
    seen_members: set[str] = set()
    for item in doc["teams"]:
        if not isinstance(item, dict):
            raise TeamsUnreadable(f"{TEAMS_FILE_NAME} holds a non-object team")
        team_id = item.get("id")
        name = item.get("name")
        members = item.get("members", [])
        if not isinstance(team_id, str) or not team_id or team_id in seen_ids:
            raise TeamsUnreadable(f"{TEAMS_FILE_NAME} holds a team with a bad id")
        if not isinstance(name, str) or not isinstance(members, list):
            raise TeamsUnreadable(f"{TEAMS_FILE_NAME} holds a malformed team")
        clean: list[str] = []
        for member in members:
            if not isinstance(member, str):
                raise TeamsUnreadable(f"{TEAMS_FILE_NAME} holds a non-string member")
            # The one-team invariant is enforced on every write; a document that
            # violates it was not written by this module. Reading it as-is would
            # render one crewmate twice, so the FIRST team keeps it -- the same
            # first-wins rule the write applies when it moves a crewmate.
            if member in seen_members or member in clean:
                continue
            clean.append(member)
        seen_ids.add(team_id)
        seen_members.update(clean)
        teams.append(Team(id=team_id, name=name, members=clean))
    return teams


def read_teams() -> list[Team]:
    """Every team, in the user's order. Absent file reads as no teams.

    Raises :class:`TeamsUnreadable` for a file that exists but cannot be read
    or parsed -- see the class docstring for why that is not an empty list.
    """
    path = teams_path()
    try:
        raw = read_bytes_with_retry(path)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise TeamsUnreadable(f"could not read {path}") from exc
    if len(raw) > TEAMS_FILE_MAX_BYTES:
        raise TeamsUnreadable(f"{TEAMS_FILE_NAME} is larger than a team list can be")
    return _parse_teams(raw)


def write_teams(teams: list[Team]) -> None:
    """Persist the whole team list atomically (human write path only).

    fsync like the member bindings: a team the dashboard confirmed must not
    silently vanish to a crash. ``OSError`` propagates so the handler can tell
    the user the save did not land.
    """
    path = teams_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Owner-only like every other gateway-owned leaf; best-effort tightening,
    # the sandbox mask and the file-tool fence are the real boundary.
    try:
        platform_compat.restrict_dir_to_owner(path.parent)
    except OSError:
        logger.debug("could not tighten mode on %s", path.parent, exc_info=True)
    payload = {"version": TEAMS_SCHEMA_VERSION, "teams": [t.to_dict() for t in teams]}
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > TEAMS_FILE_MAX_BYTES:
        # Refused before anything lands: a document the read would refuse must
        # never be written, or every later route answers ``teams_unreadable``.
        raise TeamError("teams_too_large", "the team list is too large to store")
    atomic_write(path, encoded, fsync=True)
    # The rename that publishes the file and, on first save, the directory's own
    # entry live in directory metadata a power-off can still lose.
    fsync_dir(path.parent)


def prune_unknown(teams: list[Team], known: set[str]) -> list[Team]:
    """*teams* with every member not in *known* dropped -- a read-side VIEW.

    Every crew-removal path calls :func:`drop_member` (the dashboard delete
    inside the registry lock, the package-sync prune for the names it actually
    deleted, ``kirocrew agents delete`` from its own process under the file
    lock). This is the second line: a drop that failed best-effort, or a
    document edited by a build that predates one of those hooks, must still
    never be ANSWERED as membership. The document is left as written; the next
    membership write re-validates it anyway.
    """
    return [Team(id=t.id, name=t.name, members=[m for m in t.members if m in known]) for t in teams]


def _validate_members(members: object, known: set[str]) -> list[str]:
    if not isinstance(members, list):
        raise TeamError("invalid_members", "members must be a list of crewmate names")
    if len(members) > TEAM_MEMBERS_MAX:
        raise TeamError("too_many_members", f"a team lists at most {TEAM_MEMBERS_MAX} crewmates")
    out: list[str] = []
    for member in members:
        if not isinstance(member, str) or not member:
            raise TeamError("invalid_members", "members must be a list of crewmate names")
        if member not in known:
            raise TeamError("unknown_member", f"no crewmate named {member!r}")
        if member not in out:
            out.append(member)
    return out


def _detach(teams: list[Team], members: list[str], *, keep: str | None) -> None:
    """Remove *members* from every team except *keep* (the one-team invariant)."""
    for team in teams:
        if team.id == keep:
            continue
        team.members = [m for m in team.members if m not in members]


def _find(teams: list[Team], team_id: str) -> Team:
    for team in teams:
        if team.id == team_id:
            return team
    raise TeamError("team_not_found", f"no team with id {team_id!r}")


def create_team(name: object, members: object, *, known: set[str]) -> Team:
    """Append a team; moves the listed crewmates out of their current team.

    *known* is the registered crew names every listed member must be in.
    """
    clean_name = validate_team_name(name)
    clean_members = _validate_members(members, known)
    with _document_lock():
        teams = read_teams()
        if len(teams) >= TEAMS_MAX:
            raise TeamError("too_many_teams", f"at most {TEAMS_MAX} teams")
        team_id = _new_team_id()
        while any(t.id == team_id for t in teams):  # pragma: no cover - 48 random bits
            team_id = _new_team_id()
        _detach(teams, clean_members, keep=None)
        team = Team(id=team_id, name=clean_name, members=clean_members)
        teams.append(team)
        write_teams(teams)
    return team


def update_team(
    team_id: str,
    *,
    name: object | None = None,
    members: object | None = None,
    known: set[str],
) -> Team:
    """Rename and/or re-member one team. Omitted fields are left unchanged."""
    if name is None and members is None:
        raise TeamError("nothing_to_update", "name or members required")
    clean_name = validate_team_name(name) if name is not None else None
    clean_members = _validate_members(members, known) if members is not None else None
    with _document_lock():
        teams = read_teams()
        team = _find(teams, team_id)
        if clean_name is not None:
            team.name = clean_name
        if clean_members is not None:
            _detach(teams, clean_members, keep=team.id)
            team.members = clean_members
        write_teams(teams)
    return team


def delete_team(team_id: str) -> None:
    """Remove a team; its crewmates simply have no team afterwards."""
    with _document_lock():
        teams = read_teams()
        team = _find(teams, team_id)
        teams.remove(team)
        write_teams(teams)


def drop_member(name: str) -> bool:
    """Remove a crewmate from whichever team lists it (a deleted crew).

    Returns True when a write happened. Best-effort by contract: the caller is
    the crew-delete path, whose config write has already committed, and a team
    list that still names a gone crewmate is tolerated by every reader (the
    roster simply has no row for it). So an unreadable document AND a failed
    rewrite (disk full, a permission change on ``crew-teams/``) are both logged and
    swallowed here -- neither may turn a completed crew delete into a 500.
    """
    try:
        with _document_lock():
            teams = read_teams()
            changed = False
            for team in teams:
                if name in team.members:
                    team.members = [m for m in team.members if m != name]
                    changed = True
            if changed:
                write_teams(teams)
    except (TeamsUnreadable, OSError):
        logger.warning("could not drop %r from its team", name, exc_info=True)
        return False
    return changed
