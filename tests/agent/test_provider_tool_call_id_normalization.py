"""Tests for provider-minted tool-call id normalization (#130363).

Some OpenAI-compatible providers mint ids (``chatcmpl-tool-<hex>``) that they
accept when produced but reject with a 502 when an assistant message replays
two or more of them in one turn. The ids are persisted verbatim, so every
later request re-sends the poisoned payload and the session wedges
permanently. ``normalize_provider_tool_call_ids`` rewrites such batches at
mint time to deterministic replacements so the stored row and every replay
agree.
"""

import hashlib
from types import SimpleNamespace

from agent.message_sanitization import normalize_provider_tool_call_ids


def _tc(cid, fn_name="terminal"):
    return SimpleNamespace(
        id=cid,
        function=SimpleNamespace(name=fn_name, arguments="{}"),
    )


def _expected(old_id):
    return f"call_{hashlib.sha256(old_id.encode('utf-8')).hexdigest()[:12]}"


class TestNormalizeProviderToolCallIds:
    def test_single_provider_id_left_byte_identical(self):
        # A lone provider id is accepted by the provider; rewriting it would
        # churn prompt-cache prefixes for no benefit.
        tc = _tc("chatcmpl-tool-abc123")
        assert normalize_provider_tool_call_ids([tc]) is False
        assert tc.id == "chatcmpl-tool-abc123"

    def test_parallel_provider_ids_rewritten_deterministically(self):
        tc1 = _tc("chatcmpl-tool-aaa")
        tc2 = _tc("chatcmpl-tool-bbb")
        assert normalize_provider_tool_call_ids([tc1, tc2]) is True
        assert tc1.id == _expected("chatcmpl-tool-aaa")
        assert tc2.id == _expected("chatcmpl-tool-bbb")
        assert tc1.id.startswith("call_") and tc2.id.startswith("call_")
        assert tc1.id != tc2.id

    def test_rewrite_is_byte_stable_across_mints(self):
        first = [_tc("chatcmpl-tool-aaa"), _tc("chatcmpl-tool-bbb")]
        second = [_tc("chatcmpl-tool-aaa"), _tc("chatcmpl-tool-bbb")]
        normalize_provider_tool_call_ids(first)
        normalize_provider_tool_call_ids(second)
        assert [tc.id for tc in first] == [tc.id for tc in second]

    def test_mixed_batch_untouched(self):
        # Trigger condition requires ALL ids in the turn to share the prefix.
        tc1 = _tc("chatcmpl-tool-aaa")
        tc2 = _tc("call_1")
        assert normalize_provider_tool_call_ids([tc1, tc2]) is False
        assert tc1.id == "chatcmpl-tool-aaa"
        assert tc2.id == "call_1"

    def test_plain_ids_untouched(self):
        tc1 = _tc("call_0")
        tc2 = _tc("call_1")
        assert normalize_provider_tool_call_ids([tc1, tc2]) is False
        assert (tc1.id, tc2.id) == ("call_0", "call_1")

    def test_three_parallel_provider_ids_all_rewritten(self):
        tcs = [_tc(f"chatcmpl-tool-{c * 3}") for c in "abc"]
        assert normalize_provider_tool_call_ids(tcs) is True
        assert all(tc.id.startswith("call_") for tc in tcs)
        assert len({tc.id for tc in tcs}) == 3

    def test_composite_id_preserves_response_item_half(self):
        tc1 = _tc("chatcmpl-tool-aaa|fc_111")
        tc2 = _tc("chatcmpl-tool-bbb|fc_222")
        assert normalize_provider_tool_call_ids([tc1, tc2]) is True
        assert tc1.id == f"{_expected('chatcmpl-tool-aaa')}|fc_111"
        assert tc2.id == f"{_expected('chatcmpl-tool-bbb')}|fc_222"

    def test_call_id_field_rewritten_when_present(self):
        tc1 = SimpleNamespace(id="chatcmpl-tool-aaa", call_id="chatcmpl-tool-aaa",
                              function=SimpleNamespace(name="t", arguments="{}"))
        tc2 = SimpleNamespace(id="chatcmpl-tool-bbb", call_id="chatcmpl-tool-bbb",
                              function=SimpleNamespace(name="t", arguments="{}"))
        assert normalize_provider_tool_call_ids([tc1, tc2]) is True
        assert tc1.call_id == _expected("chatcmpl-tool-aaa")
        assert tc2.call_id == _expected("chatcmpl-tool-bbb")

    def test_empty_and_none_inputs(self):
        assert normalize_provider_tool_call_ids([]) is False
        assert normalize_provider_tool_call_ids(None) is False

    def test_warning_logged_once_per_turn(self, caplog):
        import logging

        tc1 = _tc("chatcmpl-tool-aaa")
        tc2 = _tc("chatcmpl-tool-bbb")
        with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
            normalize_provider_tool_call_ids([tc1, tc2])
        rewrite_warnings = [r for r in caplog.records if "Rewrote" in r.message]
        assert len(rewrite_warnings) == 1
        assert "130363" in rewrite_warnings[0].message
