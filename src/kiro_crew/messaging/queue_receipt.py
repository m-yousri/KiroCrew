"""Mid-turn queue receipts — the single collapsing "⏳ Queued (N): …" bubble.

A message that arrives while a turn is running is either folded into that turn
as a steer or queued for after it. Queued messages get ONE receipt bubble that
is edited in place as the burst grows, then flipped to a durable record when the
turn drains it ("▶️ Now answering") or cancelled ("🛑 Cancelled"). Two channels
grew the same subsystem independently -- Telegram and Discord, ~560 duplicated
lines -- and this module is the half of it that is genuinely channel-neutral.

What lives here and what does NOT:

* HERE -- the receipt registry, its lock, and the three lifecycle transitions
  (create/grow, flip-to-answering, finalize-cancelled). These are pure
  bookkeeping over an opaque message id, and every line of them was identical
  across the two channels apart from the address type and the send call.
* NOT here -- ``_handle_busy`` and ``_drain_queue``. They re-enter the channel's
  own ``handle_message`` (whose signature differs per channel: route/chat_id/
  thread vs user_id/channel_id/thread_id) and they own the ``_active_renderers``
  registry. Sharing them would need a ``run_turn`` callback that buys nothing
  and couples this module to turn execution.

Channels reach the transitions through :class:`ReceiptSurface`, whose address is
bound at CONSTRUCTION -- so nothing below ever sees a ``chat_id``, a ``thread``
or a ``channel_id``, which is what let the five address-shaped divergences
between the two copies collapse to zero.

Dependency direction is ``<channel> -> messaging`` (never the reverse), matching
``messaging/dispatch.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Verbatim items shown in a receipt before "…and N more". A large mid-turn
#: burst would otherwise grow the rendered receipt past a channel's message
#: limit; the count prefix still reflects the true total.
RECEIPT_MAX_ITEMS = 5

#: Instant, no-extra-bubble acknowledgement that a mid-turn steer was accepted
#: and folded into the running turn (not merely "seen" — 👀 reads as passive).
STEER_ACK_EMOJI = "🫡"

#: Upper bound on how many queued messages collapse into a single combined turn.
#: A single human will not realistically burst past this mid-turn; anything beyond
#: stays queued and drains after the next turn. Lives here rather than in each
#: dispatcher because it bounds the same collapse in every channel that carries
#: the queue, and three copies had already drifted apart by comment alone.
MAX_COLLAPSE = 50

#: What a receipt SHOWS for a message whose only content was an upload. An
#: attachment-only message has no text, and a blank line in the bubble reads as a
#: message the queue lost. Lives here because every channel that ingests files
#: needs the same substitution in both receipt transitions.
ATTACHMENT_PLACEHOLDER = "[attachment]"


def short(text: str, limit: int = 40) -> str:
    """Collapse whitespace and truncate for compact receipt display."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def receipt_text(
    texts: list[str],
    *,
    answering: bool = False,
    cancelled: bool = False,
) -> str:
    """Render the single collapsing receipt for ``texts`` (order preserved).

    Only the first :data:`RECEIPT_MAX_ITEMS` are listed verbatim; the count
    prefix still reflects the true total.
    """
    count = len(texts)
    items = " · ".join(f"“{short(t)}”" for t in texts[:RECEIPT_MAX_ITEMS])
    if count > RECEIPT_MAX_ITEMS:
        items += f" · …and {count - RECEIPT_MAX_ITEMS} more"
    if cancelled:
        return f"🛑 Cancelled ({count}): {items}"
    if answering:
        return f"▶️ Now answering ({count}): {items}"
    return f"⏳ Queued ({count}): {items}"


@dataclass
class QueueReceipt:
    """The single, in-place receipt bubble tracking messages queued mid-turn.

    ``msg_id`` is deliberately opaque (``Any``): Telegram message ids are ints
    and Discord's are strings, the two are never interleaved in one process, and
    a generic parameter would add ceremony without catching a real mixup -- the
    id is only ever handed straight back to the surface that produced it.

    ``surface`` is that producer, held for the whole life of the receipt, because
    a session-wide transition has to reach a bubble in a conversation it is not
    being called from: ``/stop`` clears the queue for the WHOLE session, and the
    only thing that can address another conversation's bubble is the surface that
    posted it. It also makes "an id is only editable by its own surface"
    structural rather than a convention each caller has to honour.
    """

    msg_id: Any
    surface: "ReceiptSurface"
    texts: list[str] = field(default_factory=list)

    #: The FINAL record this receipt still has to show, set when that edit did
    #: not land. Its presence is what makes the receipt terminal: the messages it
    #: listed have left the queue -- answered or discarded -- so it must never
    #: again be grown, flipped, or counted as something queued, and
    #: :meth:`has_receipt` reports it as absent. The entry is kept only because it
    #: is the bubble's only handle, and the body is kept with it because the retry
    #: has to write the record that was intended, not one recomputed later from a
    #: different transition. The next thing that conversation does retries it once
    #: and then lets go, so an edit that keeps failing cannot hold the key.
    final_body: str | None = None


class ReceiptSurface(Protocol):
    """One conversation's receipt bubble, with its address already bound.

    Implementations close over whatever addresses their channel needs (Telegram
    binds ``chat_id`` AND the forum ``thread``; Discord binds ``channel_id``), so
    forum routing and channel addressing stay entirely channel-local.
    """

    #: Channel name for log lines only ("telegram" / "discord").
    label: str

    #: Names the CONVERSATION this surface is bound to, and so the conversation a
    #: receipt created through it belongs to. It is the registry's key alongside
    #: the session key, because the session key is too coarse: under
    #: ``messaging.dm_scope = "unified"`` every allow-listed person's direct DM
    #: collapses into one ``unified:{agent}`` bucket, so two people in two
    #: separate chats share it. The same reasoning already keys the approval and
    #: choice registries per conversation rather than per session.
    #:
    #: Only ever COMPARED and logged here, never parsed, so its spelling stays
    #: channel-local -- but for that reason it must be an OPAQUE conversation id
    #: (a chat/channel/conversation/room id), never an email or a display name.
    #:
    #: It is deliberately not "the whole bound address". A channel may bind more
    #: than the conversation for sending -- Telegram binds the forum ``thread``,
    #: Webex the thread ``parent_id`` -- and it may legitimately bind less when it
    #: only needs to edit (Telegram's ``/stop`` surface carries no ``thread``,
    #: because ``editMessageText`` is not threaded). Keying on the bound address
    #: would make those two spellings of one conversation fail to find each other.
    receipt_key: str

    async def send_receipt(self, body: str) -> Any | None:
        """Post a new receipt bubble. Returns an opaque message id, or None."""

    async def edit_receipt(self, msg_id: Any, body: str) -> bool:
        """Rewrite the receipt in place. Returns whether the edit LANDED.

        Reporting it is load-bearing, not informational: a terminal transition
        retires the registry entry, which is the bubble's only handle, and a
        channel that merely says nothing on failure gets that entry destroyed over
        a bubble still reading "⏳ Queued". Every channel client already answers
        this question -- each returns a bool and documents it -- so an
        implementation passes that answer straight through rather than inventing
        one. Raising is equally acceptable and treated identically; what is not
        acceptable is swallowing the client's False and returning None.
        """


class ReceiptQueue:
    """Owns the per-conversation receipt registry, its lock, and the transitions.

    A receipt is registered under ``(session_key, surface.receipt_key)`` -- the
    conversation its bubble was posted in, WITHIN one session -- and not under the
    session key alone. A receipt is a message in one chat, and its ``msg_id`` is
    meaningless in any other; the session key does not identify a chat, because
    ``messaging.dm_scope = "unified"`` deliberately collapses every allow-listed
    person's direct DM into one bucket. Keyed on the session alone, a second
    person queueing mid-turn found the FIRST person's receipt and edited it
    through their own surface -- an id that does not exist in their chat -- so the
    edit raised, was swallowed, and they got no receipt at all. The session key
    stays in the key because it carries the ``/new`` generation, so a bumped
    generation cannot inherit the previous one's live bubble.

    The lock is deliberately CALLER-HELD and exposed as :attr:`lock` rather than
    taken inside each method. Holding it across BOTH the enqueue and the receipt
    bookkeeping is what makes the subsystem race-free against the end-of-turn
    drain, which takes the same lock across dequeue + flip: the drain either sees
    a message queued WITH its receipt or sees neither yet -- never a half state
    that would orphan a bubble. ``/stop`` holds it across clear_queue + finalize
    for the same reason. Hiding the lock inside these methods would silently
    reintroduce that race, which is why the ``_locked`` suffixes stay in the
    public names: ugly, and load-bearing.
    """

    def __init__(self) -> None:
        self._receipts: dict[tuple[str, str], QueueReceipt] = {}
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        """The lock callers MUST hold across compound operations (see class doc)."""
        return self._lock

    def has_receipt(self, session_key: str, receipt_key: str) -> bool:
        """Whether a live receipt exists for this conversation in this session.

        A retained terminal entry is NOT live: its record is written, or could not
        be, and either way its messages have left the queue. It answers False so
        this conversation's next message opens a fresh bubble instead of growing a
        dead one.
        """
        receipt = self._receipts.get((session_key, receipt_key))
        return receipt is not None and receipt.final_body is None

    def _live_keys(self, session_key: str) -> list[tuple[str, str]]:
        """Keys in this session whose receipts still stand for QUEUED messages."""
        return [
            key
            for key, receipt in self._receipts.items()
            if key[0] == session_key and receipt.final_body is None
        ]

    async def _retry_terminal(self, key: tuple[str, str], receipt: QueueReceipt) -> None:
        """Give a stranded terminal record one more attempt, then release the key.

        Writes the body that transition MEANT to show, carried on the receipt, so a
        cancel is retried as a cancel and a flip as a flip. Called when the
        conversation acts again, which is the first moment the platform may have
        recovered. One attempt, not a loop: the entry exists to rescue a transient
        failure, and an edit that keeps failing must not keep this conversation's
        key -- its next bubble needs it.
        """
        if receipt.final_body is not None:
            await self._edit(receipt, receipt.final_body, "record-retry")
        del self._receipts[key]

    def _warn_unresolved(self, session_key: str, surface: ReceiptSurface, transition: str) -> None:
        """Report a transition that found no receipt for the conversation it names.

        A session with NO live receipt at all is the ordinary case -- nothing was
        queued -- and says nothing, or every uneventful turn would warn. Live
        receipts under OTHER conversations is the diagnosable one: this turn is
        flipping a conversation that never opened a receipt while someone else's
        bubble is still sitting on "⏳ Queued". On Telegram, Discord and Webex the
        end-of-turn drain builds its surface from the inbound that FINISHED the
        turn rather than from each queued message's own origin, which is the way
        to reach this in normal operation, so the log names the issue tracking
        that. Those queue entries are untouched, so nothing is lost: the bubble is
        stale until that conversation's own turn drains it.
        """
        stranded = len(self._live_keys(session_key))
        if not stranded:
            return
        logger.warning(
            "%s: queue receipt %s found no receipt for conversation %s; "
            "%d receipt(s) under other conversations in this session stay on "
            "'Queued' (see #12574: the drain may run under the turn opener's "
            "envelope rather than each queued message's own)",
            surface.label,
            transition,
            surface.receipt_key,
            stranded,
        )

    async def _edit(self, receipt: QueueReceipt, body: str, transition: str) -> bool:
        """Rewrite one receipt through the surface that posted it.

        Always that surface, never the caller's: the id and the address that can
        reach it are one thing, and a session-wide transition is called from a
        conversation that is not the bubble's. Debug, and only debug, because with
        the two bound together what is left is a transient API failure on a
        cosmetic edit, not a receipt reaching the wrong chat.

        Returns whether the edit LANDED, because a caller that drops the registry
        entry is destroying the only handle that bubble has. Swallowing the failure
        and reporting nothing let a terminal edit be presumed done: the entry went,
        the bubble kept reading "⏳ Queued" with nothing left able to rewrite it,
        and the next message opened a second bubble beside it. A caller that
        retires a receipt MUST therefore branch on this, not on reaching the line
        after the edit.

        A surface that RETURNS False is treated exactly like one that raises, and
        this is the common case rather than the exotic one: every channel client
        answers a non-2xx with False instead of an exception -- a rate limit, or
        Webex's cap of ten edits per message, after which the API replies 400. Read
        only the exception, and those ordinary failures are indistinguishable from
        success, which is precisely the state that strands a bubble with no retry
        left to rescue it.
        """
        try:
            landed = await receipt.surface.edit_receipt(receipt.msg_id, body)
        except Exception:
            logger.debug(
                "%s: queue receipt %s failed", receipt.surface.label, transition, exc_info=True
            )
            return False
        if landed is False:
            # Not `if not landed`: a surface that returns None has not reported a
            # failure, and reading its silence as one would keep every receipt on
            # a channel that has not adopted the contract yet.
            logger.debug("%s: queue receipt %s was not accepted", receipt.surface.label, transition)
            return False
        return True

    async def create_or_grow_locked(
        self, session_key: str, surface: ReceiptSurface, display_text: str
    ) -> None:
        """Create the receipt, or append to it and edit in place.

        ``display_text`` is what the receipt SHOWS, which is not always the raw
        message: a file-capable channel substitutes :data:`ATTACHMENT_PLACEHOLDER`
        for an attachment-only message so the bubble is not blank. Caller MUST hold
        :attr:`lock`, and MUST have already enqueued the message under that same
        hold.

        A message from a conversation that has no receipt yet opens its OWN
        bubble, even when another conversation in the same session already has
        one: two people sharing a unified session key do not share a chat.

        A conversation whose last record failed to land has a terminal entry
        still on its key. That entry is not grown -- the messages it lists are
        gone, so appending to it would put already-answered text back under
        "⏳ Queued" beside the new message -- but this is the first evidence the
        platform may be answering again, so its record is retried once here before
        the key is handed to the new bubble.
        """
        entry = (session_key, surface.receipt_key)
        receipt = self._receipts.get(entry)
        if receipt is not None and receipt.final_body is not None:
            await self._retry_terminal(entry, receipt)
            receipt = None
        if receipt is None:
            msg_id = await surface.send_receipt(receipt_text([display_text]))
            if msg_id is not None:
                self._receipts[entry] = QueueReceipt(
                    msg_id=msg_id, surface=surface, texts=[display_text]
                )
            return
        receipt.texts.append(display_text)
        await self._edit(receipt, receipt_text(receipt.texts), "grow")

    async def flip_answering_locked(
        self,
        session_key: str,
        surface: ReceiptSurface,
        answered: list[str],
        deferred: int = 0,
    ) -> None:
        """Flip the receipt to a durable "▶️ Now answering" record.

        Resolves the receipt by the conversation ``surface`` names, so it flips the
        bubble that conversation actually owns and never another's, and shows it only
        the part of ``answered`` that is ITS OWN. ``answered`` is the whole turn, and
        on a session-wide dequeue a turn carries co-tenants' messages, so showing it
        whole would print another person's queued text in this person's chat.

        When NONE of it matches, the receipt is not finalized on the strength of
        that alone, because two different things produce it. One is a channel
        whose rendering differs from what the bubble displayed, and then the
        receipt's own texts are the right body -- never the turn's. The other is
        that these messages were simply not answered: the drain left them past a
        cap, or their text was ambiguous and so was never attributable in the
        first place. Only the first may be finalized, and the two are told apart
        by evidence already in hand -- the queue is provably empty when the drain
        re-enqueued nothing and no text was ambiguous. Otherwise this bubble stays
        live, exactly as a co-tenant in the same state does, and gets no privilege
        for having opened the turn. A stale "⏳ Queued" is corrected by this
        conversation's own next turn; "▶️ Now answering" printed over a message
        still waiting is a lie the record keeps.

        ``deferred`` is what the drain re-enqueued past a cap, and it is reported
        for the whole drain rather than per conversation. It is shown as-is only
        while this conversation is ALONE on the session key, where the two are the
        same number. Once a co-tenant holds a receipt the count spans both, and
        printing it would tell this person how many messages the OTHER person has
        waiting -- so the body then states only what the split proved this
        conversation did not get answered.

        The session's OTHER receipts are then reconciled against ``answered``,
        because a drain that dequeues by session key alone (Telegram, Discord)
        consumes their entries too. Whether a receipt's messages were consumed is
        read off the receipt itself rather than guessed from the channel: its
        ``texts`` are exactly what it displayed, so one whose every text this turn
        answered stands for nothing any more and is finalized in its OWN chat,
        showing its OWN texts. One with a text left over is NOT finalized -- a
        message of its own is still waiting -- but it does shrink to just that
        remainder, so it never keeps showing text this turn already answered as
        still queued. Counting is by multiset, over a tally of EVERY receipt in
        the session including this conversation's own, so two conversations that
        happened to send the same words are not mistaken for each other. While
        the turn answered at least as many copies of a text as the session holds,
        that accounting is exact and each conversation takes its own. Short of
        that, display text stops being an identity at all and the registry
        declines to use it: a text the session wants more copies of than were
        answered, and that more than one conversation is displaying, is claimed by
        none of them -- this conversation included, which gets no privilege for
        having opened the turn -- leaving those bubbles live rather than
        finalizing whichever the dict happened to yield first, and never
        presenting as answered a copy that may have been a co-tenant's. Leaving
        one live is self-correcting -- that conversation's own next turn resolves
        it -- where finalizing the wrong chat is a lie that never is. Exact
        per-entry attribution would need the dequeued entry's own origin, which
        the drains do not carry; this reconciliation is what the registry can say
        honestly without it.

        NO receipt is retired until its record is ON the bubble, never merely
        because the edit was attempted -- this conversation's own included. The
        registry entry is that bubble's only handle, so dropping it on a transient
        platform failure strands it reading "⏳ Queued" permanently while the next
        message opens a second bubble beside it. A kept entry is TERMINAL and
        carries the record it owes in ``final_body``, because its messages were
        consumed: growing it would put already-answered text back under "Queued"
        beside the next message, and recomputing the body at retry time would write
        whatever transition happened to run then rather than the one that failed.
        So it is never grown, never flipped, not counted as queued, and reported
        absent by :meth:`has_receipt`; the next thing that conversation does writes
        that record once and releases the key.
        :meth:`finish_cancelled_locked` keeps its entry the same way, and there the
        messages are gone rather than answered -- ``clear_queue`` discarded them --
        which is the same reason a kept entry must not stay live.

        When the keyed receipt is
        not there at all -- its send failed, so it was never recorded -- the
        reconciliation does not run: without its texts there is no way to take its
        own share out of the tally first, and this conversation's messages would be
        credited to whoever shows the same words. That state is already unexplained
        and logged; acting on it is how one conversation's record ends up closed
        over another's message.

        Caller MUST hold :attr:`lock` across dequeue + this call.
        """
        remaining = Counter(answered)

        def split(
            texts: list[str], blocked: frozenset[str] = frozenset()
        ) -> tuple[list[str], list[str]]:
            """Partition ``texts`` into (answered this turn, still waiting)."""
            taken: list[str] = []
            waiting: list[str] = []
            for text in texts:
                if text not in blocked and remaining[text] > 0:
                    remaining[text] -= 1
                    taken.append(text)
                else:
                    waiting.append(text)
            return taken, waiting

        entry = (session_key, surface.receipt_key)
        receipt = self._receipts.get(entry)
        if receipt is not None and receipt.final_body is not None:
            await self._retry_terminal(entry, receipt)
            receipt = None
        if receipt is None:
            self._warn_unresolved(session_key, surface, "flip")
            return
        del self._receipts[entry]
        others = self._live_keys(session_key)
        # One tally over every receipt in the session, this conversation's
        # included, taken before any split spends a copy. A text is ambiguous
        # only when the session wants MORE copies of it than the turn answered:
        # with enough to go round, the multiset accounting is exact and each
        # conversation takes its own. Short of that, the copies cannot be
        # attributed by text, and this conversation gets no privilege for having
        # opened the turn -- claiming one would spend what a co-tenant's own
        # message accounts for, presenting a text this turn may never have
        # answered as answered. Holders are counted per conversation, so a
        # conversation that repeated itself is never ambiguous with itself.
        demand: Counter[str] = Counter()
        holders: Counter[str] = Counter()
        for texts in [receipt.texts, *(self._receipts[k].texts for k in others)]:
            demand.update(texts)
            holders.update(set(texts))
        ambiguous = frozenset(
            text
            for text, held_by in holders.items()
            if held_by > 1 and remaining[text] < demand[text]
        )
        mine, left = split(receipt.texts, ambiguous)
        # Nothing of this conversation's was attributable. The body below would
        # then fall back to the WHOLE bubble and call it answered, which is true
        # only when the queue is provably empty: the drain re-enqueued nothing and
        # no text here was ambiguous, leaving a rendering mismatch as the sole
        # explanation. Short of that a message of this conversation's may still be
        # waiting -- deferred past the cap, or holding a text that was never
        # attributable -- so the bubble is left exactly as it is, the same outcome
        # a co-tenant in this state gets below. Opening the turn earns no
        # privilege here. Stale reads "⏳ Queued" until this conversation's own
        # next turn resolves it; the alternative is a permanent record claiming a
        # message was answered while it sits in the queue.
        unattributable = bool(deferred) or any(text in ambiguous for text in left)
        if not mine and left and unattributable:
            self._receipts[entry] = receipt
            body = None
        else:
            body = receipt_text(mine or receipt.texts, answering=True)
        # ``deferred`` counts the whole drain, so it is this conversation's own
        # number only while this conversation is alone on the key. With a co-tenant
        # present it spans both, and the honest per-conversation number is what the
        # split PROVED was not answered -- an ambiguous leftover proves nothing
        # either way, so it is not counted -- and only when ``mine`` matched,
        # because an empty ``mine`` makes ``left`` the whole bubble.
        held = (
            deferred
            if not others
            else (sum(1 for text in left if text not in ambiguous) if mine else 0)
        )
        if body is not None:
            if held:
                body += f" · +{held} deferred"
            # Retire it only once the record is actually ON the bubble, and keep
            # the body with the entry when it does not land -- see the co-tenant
            # branch below for why the entry is the bubble's only handle, and why
            # a kept entry has to be terminal rather than growable.
            if not await self._edit(receipt, body, "flip"):
                receipt.final_body = body
                self._receipts[entry] = receipt
        for key in others:
            other = self._receipts[key]
            taken, waiting = split(other.texts, ambiguous)
            if waiting:
                if taken:
                    other.texts[:] = waiting
                    await self._edit(other, receipt_text(other.texts), "shrink-consumed")
                continue
            # Retire it only once the record is actually ON the bubble. The entry
            # is that bubble's only handle, so dropping it on a transient edit
            # failure strands it reading "⏳ Queued" for good. Kept, it is TERMINAL
            # and carries the record it owes: its messages were consumed, so
            # growing it would put answered text back under "Queued" beside the
            # next message, and recomputing the body later would write whatever
            # that transition happened to be instead of this one.
            final = receipt_text(other.texts, answering=True)
            if await self._edit(other, final, "flip-consumed"):
                del self._receipts[key]
            else:
                other.final_body = final

    async def finish_cancelled_locked(self, session_key: str, surface: ReceiptSurface) -> None:
        """Finalize EVERY receipt in this session to a "🛑 Cancelled" record.

        Every one, not only the conversation that typed ``/stop``, because the
        clear this pairs with is session-wide: ``clear_queue(session_key)`` drops
        the held messages of every conversation sharing the key, so a bubble left
        reading "⏳ Queued" stands over messages that are already gone -- exactly
        the lie this record exists to prevent. Each is edited through its OWN
        surface, which is why the receipt keeps one -- the caller's surface cannot
        address anyone else's chat. ``surface`` is still taken because it names
        the channel this ``/stop`` arrived on.

        A receipt is retired only once its record is ON the bubble. When the edit
        fails the entry is KEPT, because it is that bubble's only handle and
        dropping it leaves it reading "⏳ Queued" over messages ``clear_queue`` has
        already discarded -- the exact lie this record exists to prevent, made
        permanent. Kept, it is terminal and carries this "🛑 Cancelled" body in
        ``final_body``: those messages are gone, so it is never grown, never
        flipped, and never counted as queued, and the retry writes the cancel that
        was intended rather than whatever a later transition would compute. The
        next thing this conversation does writes it once and releases the key.

        A bubble that ALREADY owes a record from an earlier transition is skipped
        rather than relabelled. Its messages left the queue when that transition
        ran, not when this ``/stop`` did, so writing "🛑 Cancelled" over a flip's
        owed "▶️ Now answering" would state the opposite of what happened and do it
        permanently. Its own conversation's next action still retries what it owes.

        Caller MUST hold :attr:`lock` across clear_queue + this call.
        """
        for key in list(self._receipts):
            if key[0] != session_key:
                continue
            receipt = self._receipts[key]
            if receipt.final_body is not None:
                # This bubble already owes a record from an earlier transition, and
                # its messages left the queue then rather than now -- a flip whose
                # edit failed owes "Now answering". ``clear_queue`` did not discard
                # THOSE messages, so relabelling them "Cancelled" would state the
                # opposite of what happened, permanently. Left alone, that
                # conversation's own next action still retries the record it owes.
                continue
            body = receipt_text(receipt.texts, cancelled=True)
            if await self._edit(receipt, body, "cancel-finalize"):
                del self._receipts[key]
            else:
                receipt.final_body = body
