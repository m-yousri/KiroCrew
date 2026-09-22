"""The shared mid-turn queue receipt: lifecycle, lock contract, and a ratchet.

Telegram and Discord grew this subsystem independently and kept ~560 duplicated
lines of it. The channel-neutral half now lives in
``messaging/queue_receipt.py``; these tests pin that channel-neutral behaviour
once, and add the
mechanism that stops a third channel from starting a third copy.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from typing import Any

import kiro_crew.messaging.queue_receipt as Q
from kiro_crew.messaging.queue_receipt import (
    ATTACHMENT_PLACEHOLDER,
    RECEIPT_MAX_ITEMS,
    ReceiptQueue,
    receipt_text,
)


class _Surface:
    """Records what a channel would have put on the wire."""

    label = "fake"

    def __init__(
        self,
        *,
        send_id: Any = 7,
        edit_raises: bool = False,
        edit_returns_false: bool = False,
        receipt_key: str = "conv-1",
    ) -> None:
        self._send_id = send_id
        self.edit_raises = edit_raises
        #: A real client answers a non-2xx with False rather than raising -- a rate
        #: limit, or Webex's cap of ten edits per message. The registry must treat
        #: that identically to an exception.
        self.edit_returns_false = edit_returns_false
        self.receipt_key = receipt_key
        self.sent: list[str] = []
        self.edits: list[tuple[Any, str]] = []

    async def send_receipt(self, body: str) -> Any | None:
        self.sent.append(body)
        return self._send_id

    async def edit_receipt(self, msg_id: Any, body: str) -> bool:
        self.edits.append((msg_id, body))
        if self.edit_raises:
            raise RuntimeError("edit failed mid-flush")
        return not self.edit_returns_false


class TestReceiptText:
    def test_queued_grows_with_the_count(self) -> None:
        assert receipt_text(["a"]).startswith("⏳ Queued (1):")
        assert receipt_text(["a", "b"]).startswith("⏳ Queued (2):")

    def test_past_the_cap_the_tail_is_summarised_not_dropped(self) -> None:
        texts = [f"m{i}" for i in range(RECEIPT_MAX_ITEMS + 3)]
        out = receipt_text(texts)
        # The count is the TRUE total even though only the cap is listed.
        assert f"({len(texts)})" in out
        assert "…and 3 more" in out

    def test_the_three_states_are_distinguishable(self) -> None:
        assert "Now answering" in receipt_text(["a"], answering=True)
        assert "Cancelled" in receipt_text(["a"], cancelled=True)


class TestLifecycle:
    def test_create_then_grow_edits_one_bubble(self) -> None:
        q, s = ReceiptQueue(), _Surface(send_id=42)

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "first")
                await q.create_or_grow_locked("s", s, "second")

        asyncio.run(go())
        assert len(s.sent) == 1, "a second message would orphan the first bubble"
        assert s.edits == [(42, receipt_text(["first", "second"]))]

    def test_flip_drops_the_entry_so_the_next_burst_opens_a_fresh_bubble(self) -> None:
        q, s = ReceiptQueue(), _Surface()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.flip_answering_locked("s", s, ["a"])
                assert not q.has_receipt("s", s.receipt_key)
                await q.create_or_grow_locked("s", s, "b")

        asyncio.run(go())
        assert len(s.sent) == 2, "post-flip burst must start a NEW receipt"

    def test_deferred_remainder_is_stated_not_implied(self) -> None:
        q, s = ReceiptQueue(), _Surface()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.flip_answering_locked("s", s, ["a"], deferred=4)

        asyncio.run(go())
        assert "+4 deferred" in s.edits[-1][1]

    def test_cancel_finalises_with_the_full_queued_list(self) -> None:
        q, s = ReceiptQueue(), _Surface()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.create_or_grow_locked("s", s, "b")
                await q.finish_cancelled_locked("s", s)

        asyncio.run(go())
        assert "Cancelled (2)" in s.edits[-1][1]
        assert not q.has_receipt("s", s.receipt_key)

    def test_a_failing_edit_never_escapes(self) -> None:
        """Receipt upkeep is cosmetic; it must not fail the turn around it."""
        q, s = ReceiptQueue(), _Surface(edit_raises=True)

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.create_or_grow_locked("s", s, "b")  # edit raises
                await q.flip_answering_locked("s", s, ["a", "b"])  # raises too

        asyncio.run(go())  # must not raise

    def test_a_send_that_returns_no_id_records_no_receipt(self) -> None:
        """No id means no bubble to edit later -- storing one would 404 forever."""
        q, s = ReceiptQueue(), _Surface(send_id=None)

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")

        asyncio.run(go())
        assert not q.has_receipt("s", s.receipt_key)


class TestConversationIdentity:
    """Two conversations sharing one session key must not share one receipt.

    ``messaging.dm_scope = "unified"`` collapses every allow-listed person's
    direct DM onto a single ``unified:{agent}`` session key on purpose, so the
    session key is NOT a chat. A receipt is a message in one chat, and its id is
    meaningless in any other -- which is what these pin.
    """

    SESSION = "unified:kirocrew"

    def test_a_second_conversation_opens_its_own_receipt(self) -> None:
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "hi")
                await q.create_or_grow_locked(self.SESSION, b, "yo")

        asyncio.run(go())
        assert len(a.sent) == 1, "the first sender's receipt is posted once"
        assert len(b.sent) == 1, "the SECOND sender must get a receipt of their own"
        assert b.edits == [], "a first message has no earlier bubble of its own to edit"

    def test_the_second_sender_is_never_shown_the_first_senders_text(self) -> None:
        """The failure this pins is a disclosure, not just a missing bubble.

        Keyed on the session alone, the second sender's arrival appended to the
        FIRST sender's receipt and pushed the combined body -- both people's
        message text -- at the second sender's chat.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "my salary is")
                await q.create_or_grow_locked(self.SESSION, b, "unrelated question")

        asyncio.run(go())
        addressed_to_b = b.sent + [body for _, body in b.edits]
        assert not any("my salary is" in body for body in addressed_to_b)
        addressed_to_a = a.sent + [body for _, body in a.edits]
        assert not any("unrelated question" in body for body in addressed_to_a)

    def test_the_first_receipt_survives_the_second_senders_arrival(self) -> None:
        """A's bubble must still be live, so A's own next message grows it."""
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "first")
                await q.create_or_grow_locked(self.SESSION, b, "other person")
                await q.create_or_grow_locked(self.SESSION, a, "second")

        asyncio.run(go())
        assert len(a.sent) == 1, "A's bubble was replaced instead of grown"
        assert [body for _, body in a.edits] == [receipt_text(["first", "second"])]

    def test_a_flip_only_resolves_the_conversation_that_owns_the_receipt(self) -> None:
        """B's flip may never take A's receipt as its own record.

        Shown on a conversation-scoped dequeue (Teams, Webex), where A's messages
        are still queued so A's receipt must survive untouched.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                # B's turn drains and flips its OWN message. A's is untouched, so
                # the reconciliation must not claim A's text.
                await q.flip_answering_locked(self.SESSION, b, ["b held"], 1)
                # A's receipt must still be live, so A's next message GROWS it.
                await q.create_or_grow_locked(self.SESSION, a, "more")

        asyncio.run(go())
        assert b.edits == [], "B has no receipt to flip; A's is not B's to take"
        assert len(a.sent) == 1 and a.edits, "A's receipt was taken by B's flip"

    def test_a_cancel_finalises_every_conversation_whose_queue_it_cleared(self) -> None:
        """``/stop`` clears the queue for the WHOLE session, so it closes out all.

        ``clear_queue(session_key)`` drops the held messages of every conversation
        sharing the key. Leaving anyone's bubble on "Queued" over messages that no
        longer exist is the lie the receipt exists to prevent, so each one is
        finalized -- in its OWN chat, through its own surface, because the chat
        that typed ``/stop`` cannot address the other.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b held")
                # B types /stop. A's messages are cleared too, so A must be told.
                await q.finish_cancelled_locked(self.SESSION, b)

        asyncio.run(go())
        assert "Cancelled" in a.edits[-1][1], "A's messages were dropped with no record"
        assert "Cancelled" in b.edits[-1][1]
        # Each record carries only its own chat's messages.
        assert "b held" not in a.edits[-1][1]
        assert "a held" not in b.edits[-1][1]

    def test_a_cancelled_session_keeps_no_live_receipt_behind(self) -> None:
        """Nothing may survive the clear, or the next burst grows a dead bubble."""
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b held")
                await q.finish_cancelled_locked(self.SESSION, b)
                await q.create_or_grow_locked(self.SESSION, a, "after")

        asyncio.run(go())
        assert len(a.sent) == 2, "A's post-cancel message must open a FRESH receipt"

    def test_one_conversation_still_collapses_into_a_single_bubble(self) -> None:
        """The split is by CONVERSATION, not by surface object or by sender.

        Two people talking in one group share a chat, so they share its bubble --
        and each mid-turn message rebuilds the surface. Splitting on anything
        finer than the conversation would post a bubble per message. The edit goes
        through the surface that POSTED the bubble, not the one the later message
        arrived on, because only the poster's id and address belong together.
        """
        first = _Surface(send_id="mid-1", receipt_key="room-7")
        again = _Surface(send_id="mid-2", receipt_key="room-7")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, first, "a")
                await q.create_or_grow_locked(self.SESSION, again, "b")

        asyncio.run(go())
        assert again.sent == [], "a second bubble orphans the first"
        assert again.edits == [], "the later surface never addresses another's bubble"
        assert "Queued (2)" in first.edits[-1][1]

    def test_a_session_wide_dequeue_leaves_no_receipt_live_over_consumed_messages(
        self,
    ) -> None:
        """A drain that dequeues by session key alone consumes everyone's entries.

        Telegram and Discord do exactly that, so B's messages are gone once A's
        turn drains. Leaving B's receipt live would sit it on "Queued" and then
        grow it with text that has already been answered, so B's bubble is flipped
        too -- showing B's OWN messages, in B's OWN chat.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b held")
                # The drain dequeued both and collapsed them into ONE turn, so
                # `answered` carries both texts -- which is how the registry knows
                # B's message was consumed too.
                await q.flip_answering_locked(self.SESSION, a, ["a held", "b held"])
                # B's next message must open a FRESH bubble, not grow a dead one.
                await q.create_or_grow_locked(self.SESSION, b, "b again")

        asyncio.run(go())
        assert "Now answering" in b.edits[0][1], "B's consumed receipt was left on Queued"
        assert "b held" in b.edits[0][1] and "a held" not in b.edits[0][1]
        assert len(b.sent) == 2, "B's later message grew a receipt over answered text"
        assert "Queued (1)" in b.sent[1], "the fresh bubble must not carry the old text"

    def test_a_conversation_scoped_dequeue_leaves_the_still_queued_alone(self) -> None:
        """Teams and Webex re-enqueue other conversations, so those stay live.

        Flipping a receipt whose messages are still sitting in the queue would
        claim they are being answered when nothing has read them.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b held")
                # deferred=1: B's entry was re-enqueued, so B's receipt stays
                # live -- flipping it would claim a queued message is answered.
                await q.flip_answering_locked(self.SESSION, a, ["a held"], 1)
                # Still live, so B's next message GROWS B's bubble.
                await q.create_or_grow_locked(self.SESSION, b, "b again")

        asyncio.run(go())
        assert not any("Now answering" in body for _, body in b.edits)
        assert len(b.sent) == 1, "B's receipt was replaced while B's message still waited"
        assert "Queued (2)" in b.edits[-1][1]

    def test_a_capped_drain_finalises_the_consumed_and_spares_the_waiting(self) -> None:
        """A drain past a cap consumes some conversations and defers others.

        Telegram and Discord dequeue the whole session and re-enqueue the overflow,
        so one co-tenant's messages can be fully answered while another's are back
        in the queue. The one that was answered must not be left on "Queued" --
        its next message would grow a bubble carrying already-answered text -- and
        the one still waiting must not be finalized.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        c = _Surface(send_id="mid-c", receipt_key="chat-C")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b held")
                await q.create_or_grow_locked(self.SESSION, c, "c held")
                # The turn answered A's and B's; C's was re-enqueued past the cap.
                await q.flip_answering_locked(self.SESSION, a, ["a held", "b held"], 1)
                # C's message is still queued, so C's bubble must still grow.
                await q.create_or_grow_locked(self.SESSION, c, "c again")

        asyncio.run(go())
        assert "b held" not in a.edits[-1][1], "B's message was shown in A's chat"
        assert "a held" in a.edits[-1][1]
        assert "Now answering" in b.edits[-1][1], "B's answered messages were left on Queued"
        assert "b held" in b.edits[-1][1] and "a held" not in b.edits[-1][1]
        assert not any(
            "Now answering" in body for _, body in c.edits
        ), "C's message is still queued; claiming it is being answered is a lie"
        assert len(c.sent) == 1 and "Queued (2)" in c.edits[-1][1]

    def test_a_co_tenants_deferred_count_is_not_printed_in_this_persons_chat(self) -> None:
        """The drain's deferred count spans the session, so it is not A's to show.

        Telegram and Discord report how many entries they re-enqueued for the WHOLE
        drain. On a unified DM those can be entirely someone else's, so printing the
        number in the answering person's finalized receipt tells them how many
        messages the OTHER person still has waiting.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b one")
                await q.create_or_grow_locked(self.SESSION, b, "b two")
                # A's one message was answered; both of B's were re-enqueued.
                await q.flip_answering_locked(self.SESSION, a, ["a held"], 2)

        asyncio.run(go())
        assert "Now answering" in a.edits[-1][1]
        assert "deferred" not in a.edits[-1][1], "B's queued count was shown in A's chat"

    def test_the_answering_conversations_own_remainder_is_still_stated(self) -> None:
        """Hiding the session-wide count must not hide A's OWN remainder.

        A's bubble is the only record that a message of A's is still held, so a cap
        that left one behind is still called out -- with A's own number, which is
        what the split proved this turn did not answer, never the drain's total.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a one")
                await q.create_or_grow_locked(self.SESSION, a, "a two")
                await q.create_or_grow_locked(self.SESSION, b, "b one")
                await q.create_or_grow_locked(self.SESSION, b, "b two")
                # The turn answered only A's first message. A's second and both of
                # B's went back, so the drain reports 3 -- of which exactly 1 is A's.
                await q.flip_answering_locked(self.SESSION, a, ["a one"], 3)

        asyncio.run(go())
        assert "+1 deferred" in a.edits[-1][1]
        assert "+3 deferred" not in a.edits[-1][1], "the drain's session-wide total leaked"

    def test_a_missing_keyed_receipt_finalises_nobody_else(self) -> None:
        """A receipt whose send failed was never recorded, so its share is unknown.

        Reconciliation works by taking the answering conversation's own texts out
        of the tally first. With no receipt there is nothing to take out, so that
        conversation's words stay in the tally and are credited to whoever happens
        to show the same words -- closing a bubble over a message still queued.
        """
        a = _Surface(send_id=None, receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                # A's receipt send fails, so A has no registry entry at all.
                await q.create_or_grow_locked(self.SESSION, a, "same words")
                await q.create_or_grow_locked(self.SESSION, b, "same words")
                await q.flip_answering_locked(self.SESSION, a, ["same words"], 1)

        asyncio.run(go())
        assert not any(
            "Now answering" in body for _, body in b.edits
        ), "B was finalized on A's text while B's own message is still queued"

    def test_a_text_two_conversations_share_is_claimed_by_neither(self) -> None:
        """Identical display text cannot say which conversation was consumed.

        Two people each send an attachment, so both bubbles read the same
        placeholder. A capped drain consumes ONE of them. Nothing in the registry
        says whose, so finalizing either is a coin flip that lands on the wrong
        chat half the time. Both stay live instead: a live receipt is resolved by
        that conversation's own next turn, where a wrongly finalized one is a
        permanent lie. Exact attribution would need each dequeued entry's own
        origin, which the drains do not carry.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        c = _Surface(send_id="mid-c", receipt_key="chat-C")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a asked")
                await q.create_or_grow_locked(self.SESSION, b, "[attachment]")
                await q.create_or_grow_locked(self.SESSION, c, "[attachment]")
                # One of the two attachments got through; the registry cannot say which.
                await q.flip_answering_locked(self.SESSION, a, ["a asked", "[attachment]"], 1)

        asyncio.run(go())
        for who, surf in (("B", b), ("C", c)):
            assert not any(
                "Now answering" in body for _, body in surf.edits
            ), f"{who} was finalized on a text it may not have owned"

    def test_the_flipped_receipt_is_never_shown_the_whole_turns_text(self) -> None:
        """The turn's ``answered`` list is not one conversation's property.

        A session-wide dequeue hands ``flip_answering`` every message the turn
        consumed, co-tenants' included. Editing the keyed bubble to that whole list
        prints another person's queued words in this person's chat -- the exact
        disclosure the per-conversation key exists to stop.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a secret")
                await q.create_or_grow_locked(self.SESSION, b, "b secret")
                # One drain consumed both; A's conversation finished the turn.
                await q.flip_answering_locked(self.SESSION, a, ["a secret", "b secret"])

        asyncio.run(go())
        assert "b secret" not in a.edits[-1][1], "B's message was shown in A's chat"
        assert "a secret" in a.edits[-1][1]
        assert "a secret" not in b.edits[-1][1], "A's message was shown in B's chat"
        assert "b secret" in b.edits[-1][1]

    def test_a_half_consumed_receipt_drops_the_answered_text_and_keeps_the_waiting(
        self,
    ) -> None:
        """A cap can split ONE conversation: some of its messages answered, some not.

        The receipt must stay live, because a message of its own really is still
        queued -- but it must stop listing the text this turn already answered.
        Leaving the whole list up means the bubble reads "Queued" over something
        that was answered, and its next message grows a receipt carrying that
        answered text forward for good.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        c = _Surface(send_id="mid-c", receipt_key="chat-C")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, c, "c first")
                await q.create_or_grow_locked(self.SESSION, c, "c second")
                # The cap let A's and C's FIRST through; C's second went back.
                await q.flip_answering_locked(self.SESSION, a, ["a held", "c first"], 1)
                await q.create_or_grow_locked(self.SESSION, c, "c third")

        asyncio.run(go())
        body = c.edits[-1][1]
        assert "c first" not in body, "an answered message was left showing as queued"
        assert "c second" in body and "c third" in body
        assert "Now answering" not in body, "C still has a message queued"
        assert len(c.sent) == 1, "the receipt must be shrunk, not reopened"

    def test_identical_words_from_two_people_are_not_mistaken_for_each_other(self) -> None:
        """Reconciliation counts by multiset, and nets out the flipped receipt first.

        Two people both say "ok". The turn answers one of them. Without taking the
        flipped receipt's own share out of the tally first, the other person's
        identical word looks answered too, and their bubble is closed over a
        message still sitting in the queue.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "ok")
                await q.create_or_grow_locked(self.SESSION, b, "ok")
                # Only A's "ok" was answered; B's was re-enqueued past the cap.
                await q.flip_answering_locked(self.SESSION, a, ["ok"], 1)

        asyncio.run(go())
        assert not any(
            "Now answering" in body for _, body in b.edits
        ), "B's queued message was declared answered because A used the same word"

    def test_the_answering_conversation_cannot_claim_a_co_tenants_duplicate(self) -> None:
        """The flipped receipt gets no privilege over a short-supply duplicate.

        A and B both say "ok"; A also says "a2". The turn answered "a2" and ONE
        "ok" -- and which chat that copy came from is not knowable from the text.
        Splitting the flipped receipt first and unfiltered made A simply take it,
        so A's own bubble asserted a message was answered that may still be
        sitting in B's queue.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "ok")
                await q.create_or_grow_locked(self.SESSION, a, "a2")
                await q.create_or_grow_locked(self.SESSION, b, "ok")
                await q.flip_answering_locked(self.SESSION, a, ["a2", "ok"])

        asyncio.run(go())
        flip = a.edits[-1][1]
        assert "Now answering (1)" in flip, f"A claimed a copy it cannot prove is its own: {flip}"
        assert "a2" in flip
        # An ambiguous leftover proves nothing, so it must not be counted as A's
        # own deferred message either -- that would assert the mirror-image lie.
        assert "deferred" not in flip, f"an unattributable copy was reported as deferred: {flip}"
        # Neither claimed it, so B keeps its bubble and its own next turn resolves it.
        assert q.has_receipt(self.SESSION, "chat-B")

    def test_enough_answered_copies_still_attribute_exactly(self) -> None:
        """Ambiguity is short SUPPLY, not a shared word.

        Two people both say "ok" and the turn answered both copies. There is
        nothing to guess: one copy each. Refusing every duplicate outright would
        strand both bubbles on "Queued" over messages that are demonstrably gone.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "ok")
                await q.create_or_grow_locked(self.SESSION, b, "ok")
                await q.flip_answering_locked(self.SESSION, a, ["ok", "ok"])

        asyncio.run(go())
        assert any(
            "Now answering" in body for _, body in b.edits
        ), "B's answered message was not recorded"
        assert not q.has_receipt(self.SESSION, "chat-B")
        assert not q.has_receipt(self.SESSION, "chat-A")

    def test_a_co_tenant_whose_terminal_edit_failed_stays_resolvable(self) -> None:
        """A receipt is retired only once its record is ON the bubble.

        B's messages were consumed by A's session-wide drain, so B's bubble must
        become a record. The platform refuses that edit. Dropping the entry
        destroyed the bubble's only handle: it read "⏳ Queued" for good, and B's
        next message opened a second bubble beside it. Kept, the entry is a retry
        handle -- and NOT a live receipt, or B's next message grows it and one
        bubble shows an answered message and a queued one as a single queue.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B", edit_raises=True)
        q = ReceiptQueue()
        live_after_failure: list[bool] = []

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a said")
                await q.create_or_grow_locked(self.SESSION, b, "b said")
                await q.flip_answering_locked(self.SESSION, a, ["a said", "b said"])
                live_after_failure.append(q.has_receipt(self.SESSION, "chat-B"))
                # The platform recovers and B sends again.
                b.edit_raises = False
                await q.create_or_grow_locked(self.SESSION, b, "b again")

        asyncio.run(go())
        assert not live_after_failure[0], "a record still owed is not a live receipt"
        retry = b.edits[-1][1]
        assert "Now answering" in retry, f"the record B was owed was never written: {b.edits}"
        assert "b again" not in retry, "the answered record was grown with a new message"
        assert len(b.sent) == 2, "B's new message did not get its own bubble"
        assert "b said" not in b.sent[1], "the fresh bubble inherited answered text"
        # A's edit landed, so A's entry is gone -- the two are judged separately.
        assert not q.has_receipt(self.SESSION, "chat-A")

    def test_the_turn_opener_is_not_finalised_over_its_own_deferred_message(self) -> None:
        """Opening the turn does not mean your message was answered.

        A and B share a unified DM key. The drain fills its cap with B's message
        and puts A's back in the queue, then flips through A's surface because A's
        inbound finished the turn. None of A's text is in ``answered``, and the
        fallback would print A's whole bubble as "Now answering" and drop it --
        over a message still sitting in the queue, which is the one record this
        bubble exists to keep honest.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a held")
                await q.create_or_grow_locked(self.SESSION, b, "b said")
                # The cap took B's and deferred A's: one entry re-enqueued.
                await q.flip_answering_locked(self.SESSION, a, ["b said"], 1)

        asyncio.run(go())
        assert not a.edits, f"A's bubble was rewritten over a queued message: {a.edits}"
        assert q.has_receipt(self.SESSION, "chat-A"), "A's still-queued message lost its receipt"
        assert any("Now answering" in body for _, body in b.edits), "B's record is missing"
        assert not q.has_receipt(self.SESSION, "chat-B")

    def test_a_wholly_ambiguous_bubble_is_not_finalised_by_the_turn_opener(self) -> None:
        """An unattributable text is not attributed to whoever opened the turn.

        Two people each send an attachment, so both bubbles read the same
        placeholder and the turn answered one copy. Which one cannot be known from
        the text. The co-tenant's bubble correctly stays live; the opener's must
        too, or the guess is settled in favour of whoever happened to be keyed.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, ATTACHMENT_PLACEHOLDER)
                await q.create_or_grow_locked(self.SESSION, b, ATTACHMENT_PLACEHOLDER)
                await q.flip_answering_locked(self.SESSION, a, [ATTACHMENT_PLACEHOLDER])

        asyncio.run(go())
        assert not any(
            "Now answering" in body for _, body in [*a.edits, *b.edits]
        ), "an unattributable copy was presented as answered"
        assert q.has_receipt(self.SESSION, "chat-A"), "the opener's bubble was finalised on a guess"
        assert q.has_receipt(self.SESSION, "chat-B")

    def test_a_reported_failure_keeps_the_record_as_surely_as_a_raised_one(self) -> None:
        """A channel that answers False has not written the record.

        This is the ORDINARY failure, not the exotic one: every channel client
        reports a non-2xx as False rather than raising -- a rate limit, or Webex's
        cap of ten edits per message. Read only the exception and that is
        indistinguishable from success, so the entry is retired over a bubble still
        reading "⏳ Queued", with the retry that would have rescued it never created.
        """
        s = _Surface(edit_returns_false=True)
        q = ReceiptQueue()
        live_after_failure: list[bool] = []

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.flip_answering_locked("s", s, ["a"])
                live_after_failure.append(q.has_receipt("s", s.receipt_key))
                s.edit_returns_false = False
                await q.create_or_grow_locked("s", s, "b")

        asyncio.run(go())
        assert not live_after_failure[0], "a record still owed is not a live receipt"
        records = [body for _, body in s.edits if "Now answering" in body]
        assert len(records) == 2, f"a reported failure was presumed written: {s.edits}"
        assert len(s.sent) == 2, "the new message did not get its own bubble"

    def test_a_surface_that_reports_nothing_is_not_read_as_a_failure(self) -> None:
        """Silence is not a reported failure.

        The contract asks a surface to answer, but a channel that has not adopted
        it yet returns None. Reading that as False would keep every receipt in the
        registry for ever and open a second bubble beside each one.
        """

        class _Quiet(_Surface):
            async def edit_receipt(self, msg_id: Any, body: str) -> None:  # type: ignore[override]
                self.edits.append((msg_id, body))
                return None

        s = _Quiet()
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.flip_answering_locked("s", s, ["a"])
                # A retained entry would be RETRIED here, adding a second edit.
                await q.create_or_grow_locked("s", s, "b")

        asyncio.run(go())
        assert len(s.edits) == 1, f"silence was read as a failure and retried: {s.edits}"
        assert len(s.sent) == 2, "the new message did not get its own bubble"

    def test_a_stop_does_not_relabel_a_record_another_transition_owes(self) -> None:
        """``/stop`` speaks for the messages IT discarded, not for answered ones.

        A's flip failed, so its bubble owes "Now answering" over messages that were
        answered. B then types ``/stop``, which is session-wide. Recomputing every
        entry's body would write "Cancelled" over A's answered messages -- the
        opposite of what happened, and permanent.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A", edit_returns_false=True)
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "a said")
                await q.flip_answering_locked(self.SESSION, a, ["a said"])  # edit fails
                await q.create_or_grow_locked(self.SESSION, b, "b said")
                await q.finish_cancelled_locked(self.SESSION, b)

        asyncio.run(go())
        assert not any(
            "Cancelled" in body for _, body in a.edits
        ), f"an answered message was relabelled cancelled: {a.edits}"
        assert any("Cancelled" in body for _, body in b.edits), "B's own record is missing"

    def test_the_turn_openers_failed_record_stays_resolvable(self) -> None:
        """The opener's entry is its bubble's only handle too, and is not growable.

        Its messages really were answered, so a fresh bubble for its next message
        would be truthful -- but only once the record is ON this one. Dropped on a
        transient failure, the bubble reads "⏳ Queued" for good with nothing left
        able to rewrite it. Kept as a LIVE receipt instead, the next mid-turn
        message grows it, and one bubble presents an answered message and a queued
        one as the same queue. So it is kept, terminal, carrying the record it owes.
        """
        s = _Surface(edit_raises=True)
        q = ReceiptQueue()
        live_after_failure: list[bool] = []

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.flip_answering_locked("s", s, ["a"])
                live_after_failure.append(q.has_receipt("s", s.receipt_key))
                # The platform recovers and the same conversation sends again.
                s.edit_raises = False
                await q.create_or_grow_locked("s", s, "b")

        asyncio.run(go())
        assert not live_after_failure[0], "a record still owed is not a live receipt"
        records = [body for _, body in s.edits if "Now answering" in body]
        assert len(records) == 2, f"the owed record was attempted {len(records)}x, not retried"
        assert "b" not in records[-1], "the answered record was grown with a new message"
        assert len(s.sent) == 2, "the new message did not get its own bubble"
        assert q.has_receipt("s", s.receipt_key), "the fresh bubble is not live"

    def test_a_cancelled_record_that_failed_never_reenters_consumption(self) -> None:
        """A kept cancelled entry is a retry handle, not a queued message.

        ``clear_queue`` already discarded these messages. The entry is held back
        only so the "🛑 Cancelled" edit can be retried; treated as live, a later
        drain would flip that bubble to "Now answering" over messages that no
        longer exist anywhere.
        """
        s = _Surface(edit_raises=True)
        q = ReceiptQueue()
        live: list[bool] = []

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.finish_cancelled_locked("s", s)  # edit fails
                live.append(q.has_receipt("s", s.receipt_key))
                await q.flip_answering_locked("s", s, ["a"])

        asyncio.run(go())
        assert live == [False], "a cancelled entry was reported as a live receipt"
        assert not any(
            "Now answering" in body for _, body in s.edits
        ), "a discarded message was recorded as answered"

    def test_a_cancelled_record_is_retried_once_then_yields_its_key(self) -> None:
        """The next thing the conversation does is the first chance to recover.

        The retry is one attempt, not a loop: an edit that keeps failing must not
        keep the key, because this conversation's new bubble needs it.
        """
        s = _Surface(edit_raises=True)
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("s", s, "a")
                await q.finish_cancelled_locked("s", s)  # edit fails
                s.edit_raises = False
                await q.create_or_grow_locked("s", s, "b")

        asyncio.run(go())
        cancels = [body for _, body in s.edits if "Cancelled" in body]
        assert (
            len(cancels) == 2
        ), f"the record was attempted {len(cancels)}x, not retried: {s.edits}"
        assert len(s.sent) == 2, "the new message did not get its own bubble"
        assert "a" not in s.sent[1], "the fresh bubble inherited the cancelled text"
        assert q.has_receipt("s", s.receipt_key)

    def test_a_discarded_message_cannot_make_a_live_conversations_text_ambiguous(self) -> None:
        """A terminal entry is not a co-tenant holding a queued message.

        A's ``/stop`` discarded its message and the record failed to land, so its
        entry survives only as a retry handle. B then queues the same word and its
        own turn answers it. Counted as live, A's gone message makes B's word look
        like a duplicate in short supply -- so B's bubble is left "⏳ Queued" over a
        message that was demonstrably answered, and the retry handle has started
        deciding other conversations' records.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A", edit_raises=True)
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "ok")
                await q.finish_cancelled_locked(self.SESSION, a)  # A's edit fails
                await q.create_or_grow_locked(self.SESSION, b, "ok")
                await q.flip_answering_locked(self.SESSION, b, ["ok"])

        asyncio.run(go())
        assert any(
            "Now answering" in body for _, body in b.edits
        ), "a discarded message blocked a live conversation's record"
        assert not q.has_receipt(self.SESSION, "chat-B")
        assert not any(
            "Now answering" in body for _, body in a.edits
        ), "a discarded message was recorded as answered"

    def test_two_threads_of_one_room_are_two_conversations(self) -> None:
        """A Webex space's threads do not separate on the session key.

        A space is keyed ``space:{id}``, so two threads share it. Their bubbles
        live in different threads, so a room-only receipt key would append thread
        B's text to thread A's bubble and give B nothing.
        """
        t1 = _Surface(send_id="mid-1", receipt_key="room-7:thread-1")
        t2 = _Surface(send_id="mid-2", receipt_key="room-7:thread-2")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked("webex:space:7", t1, "in thread one")
                await q.create_or_grow_locked("webex:space:7", t2, "in thread two")

        asyncio.run(go())
        assert len(t2.sent) == 1, "the second thread got no receipt of its own"
        assert not any("in thread one" in b for b in t2.sent + [e[1] for e in t2.edits])

    def test_a_transition_that_resolves_nothing_while_another_holds_a_receipt_warns(
        self, caplog
    ) -> None:
        """A receipt left sitting on "Queued" is a real loss, so it is reported.

        Nothing resolves, which is safe -- no edit reaches a chat it does not
        belong to -- but silence would make the stranded bubble invisible.
        """
        a = _Surface(send_id="mid-a", receipt_key="chat-A")
        b = _Surface(send_id="mid-b", receipt_key="chat-B")
        q = ReceiptQueue()

        async def go() -> None:
            async with q.lock:
                await q.create_or_grow_locked(self.SESSION, a, "held")
                await q.flip_answering_locked(self.SESSION, b, ["held"])

        with caplog.at_level(logging.WARNING, logger=Q.__name__):
            asyncio.run(go())
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "a stranded receipt must not be silent"
        # Naming the tracked cause, so an operator reading this does not have to
        # go digging for why a drain reached a conversation that owns no receipt.
        assert "#12574" in warnings[-1].getMessage()

    def test_an_ordinary_transition_with_nothing_queued_stays_quiet(self, caplog) -> None:
        """Most turns queue nothing at all; warning on those would be noise."""
        q, s = ReceiptQueue(), _Surface(receipt_key="chat-A")

        async def go() -> None:
            async with q.lock:
                await q.flip_answering_locked("s", s, [])
                await q.finish_cancelled_locked("s", s)

        with caplog.at_level(logging.WARNING, logger=Q.__name__):
            asyncio.run(go())
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


class TestLockIsCallerHeld:
    def test_the_transitions_do_not_take_the_lock_themselves(self) -> None:
        """The atomicity contract, asserted rather than documented.

        Callers hold the lock ACROSS enqueue+receipt (and dequeue+flip), which is
        what makes the subsystem race-free against the drain. If a future change
        moved the acquire inside these methods, this deadlocks -- so the bounded
        wait is the assertion, not a timeout guard.
        """
        q, s = ReceiptQueue(), _Surface()

        async def go() -> None:
            async with q.lock:
                await asyncio.wait_for(q.create_or_grow_locked("s", s, "a"), timeout=2)
                await asyncio.wait_for(q.flip_answering_locked("s", s, ["a"]), timeout=2)
                await asyncio.wait_for(q.finish_cancelled_locked("s", s), timeout=2)

        asyncio.run(go())


def _dispatchers() -> list[Path]:
    pkg = Path(Q.__file__).resolve().parent.parent
    found = sorted(pkg.glob("*/transport_dispatch.py"))
    assert len(found) >= 5, f"expected the dispatcher set, found {found}"
    return found


class TestRatchet:
    def test_no_channel_keeps_its_own_receipt_registry_or_lock(self) -> None:
        """A third copy of this subsystem must fail here, not in production."""
        offenders: dict[str, list[str]] = {}
        for path in _dispatchers():
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
            names = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr
                in {
                    "_queue_receipts",
                    "_receipt_lock",
                }
            }
            if names:
                offenders[path.parent.name] = sorted(names)
        assert not offenders, (
            "these channels carry a private receipt registry/lock instead of the "
            f"shared ReceiptQueue, so the lock discipline can drift again: {offenders}"
        )

    def test_every_channel_with_a_queue_uses_the_shared_one(self) -> None:
        missing = []
        for path in _dispatchers():
            src = path.read_text(encoding="utf-8")
            if "_enqueue_with_receipt" in src and "ReceiptQueue" not in src:
                missing.append(path.parent.name)
        assert not missing, f"{missing} implement a mid-turn queue without the shared ReceiptQueue"
