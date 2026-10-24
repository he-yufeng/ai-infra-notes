import json
import unittest
from decimal import Decimal

from notes.examples.kimi_k31.protocol import (
    ChatStream,
    ProtocolError,
    append_tool_results,
    parse_usage,
    price_usage,
)


def event(value):
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def chunk(delta=None, finish=None, usage=None, choices=True):
    item = {"model": "kimi-k3.1-pro", "id": "chatcmpl-example", "choices": []}
    if choices:
        item["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        item["usage"] = usage
    return item


USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 112,
    "total_tokens": 1112,
    "completion_tokens_details": {"reasoning_tokens": 90},
    "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 256},
}


def tool_stream():
    return b"".join(
        [
            event(
                chunk(
                    {
                        "role": "assistant",
                        "reasoning_content": "Read the specification.",
                    }
                )
            ),
            event(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_example",
                                "type": "function",
                                "function": {
                                    "name": "get_part_spec",
                                    "arguments": '{"part_',
                                },
                            }
                        ]
                    }
                )
            ),
            event(
                chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'id":"plate"}'}}
                        ]
                    },
                    "tool_calls",
                )
            ),
            event(chunk(usage=USAGE, choices=False)),
            b"data: [DONE]\n\n",
        ]
    )


class UsageTests(unittest.TestCase):
    def test_disjoint_categories_and_reasoning(self):
        usage = parse_usage(USAGE)
        self.assertEqual(usage.input_other, 144)
        self.assertEqual(usage.input_cache_read, 600)
        self.assertEqual(usage.input_cache_write, 256)
        self.assertEqual(usage.output_tokens, 112)
        self.assertEqual(usage.reasoning_tokens, 90)
        self.assertEqual(usage.input_total, 1000)
        self.assertEqual(usage.raw, USAGE)
        self.assertIsNot(usage.raw, USAGE)

    def test_official_write_only_example(self):
        usage = parse_usage(
            {
                "prompt_tokens": 373,
                "completion_tokens": 112,
                "prompt_tokens_details": {"cache_write_tokens": 256},
            }
        )
        self.assertEqual((usage.input_other, usage.input_cache_write), (117, 256))

    def test_absent_details_and_legacy_read(self):
        self.assertIsNone(parse_usage(None))
        self.assertEqual(
            parse_usage({"prompt_tokens": 10, "completion_tokens": 2}).input_other, 10
        )
        self.assertEqual(
            parse_usage(
                {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 3}
            ).input_other,
            7,
        )

    def test_rejects_inconsistent_or_partial_usage(self):
        invalid = [
            {},
            {"prompt_tokens": 1},
            {"prompt_tokens": True, "completion_tokens": 1},
            {"prompt_tokens": -1, "completion_tokens": 1},
            {"prompt_tokens": 1.5, "completion_tokens": 1},
            {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 2},
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 8},
            {"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": []},
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_tokens_details": False,
            },
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "cached_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                parse_usage(raw)

    def test_cost_does_not_count_reasoning_twice(self):
        cost = price_usage(parse_usage(USAGE), "kimi-k3.1-pro", "5m")
        self.assertTrue(cost["complete"])
        self.assertEqual(Decimal(cost["known_usd"]), Decimal("0.002040"))

    def test_unknown_usage_and_unknown_write_price_stay_unknown(self):
        missing = price_usage(None, "kimi-k3.1-pro", "5m")
        self.assertFalse(missing["complete"])
        self.assertIsNone(missing["known_usd"])
        long_ttl = price_usage(parse_usage(USAGE), "kimi-k3.1-pro", "1h")
        self.assertFalse(long_ttl["complete"])
        self.assertEqual(Decimal(long_ttl["known_usd"]), Decimal("0.001528"))
        self.assertIn("cache_write", long_ttl["unknown"])


class StreamTests(unittest.TestCase):
    def test_fragmented_unicode_reasoning_and_empty_choices_usage(self):
        data = b"".join(
            [
                b": heartbeat\r\n\r\n",
                event(chunk({"role": "assistant", "reasoning_content": "检查孔位"})),
                event(chunk({"content": '{"ok":true}'}, "stop")),
                event(chunk(usage=USAGE, choices=False)),
                b"data: [DONE]\n\n",
            ]
        )
        parser = ChatStream("kimi-k3.1-pro")
        for byte in data:
            parser.feed(bytes([byte]))
        result = parser.finish()
        self.assertEqual(result.assistant["reasoning_content"], "检查孔位")
        self.assertEqual(json.loads(result.assistant["content"]), {"ok": True})
        self.assertEqual(result.usage.input_cache_write, 256)
        self.assertEqual(result.response_model, "kimi-k3.1-pro")

    def test_usage_chunks_are_snapshots_not_addends(self):
        parser = ChatStream("kimi-k3.1-pro")
        parser.feed(
            event(
                chunk(
                    {"content": "ok"},
                    "stop",
                    {"prompt_tokens": 5, "completion_tokens": 2},
                )
            )
        )
        parser.feed(
            event(
                chunk(usage={"prompt_tokens": 5, "completion_tokens": 4}, choices=False)
            )
        )
        parser.feed(b"data: [DONE]\n\n")
        self.assertEqual(parser.finish().usage.output_tokens, 4)

    def test_usage_counters_cannot_move_backwards(self):
        parser = ChatStream("kimi-k3.1-pro")
        parser.feed(
            event(
                chunk(
                    {"content": "ok"},
                    "stop",
                    {"prompt_tokens": 1000, "completion_tokens": 4096},
                )
            )
        )
        with self.assertRaises(ProtocolError):
            parser.feed(
                event(
                    chunk(
                        usage={"prompt_tokens": 0, "completion_tokens": 0},
                        choices=False,
                    )
                )
            )

    def test_choice_message_and_unknown_fields_are_rejected(self):
        for choice in [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            },
            {
                "index": 0,
                "delta": {"content": "answer"},
                "assistant_extension": "preserve me",
                "finish_reason": "stop",
            },
        ]:
            with self.subTest(choice=choice), self.assertRaises(ProtocolError):
                parser = ChatStream("kimi-k3.1-pro")
                parser.feed(event({"model": "kimi-k3.1-pro", "choices": [choice]}))

    def test_deep_usage_is_a_protocol_error_not_a_recursion_error(self):
        data = (
            b'data: {"model":"kimi-k3.1-pro","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"extra":'
            + b"[" * 600
            + b"0"
            + b"]" * 600
            + b"}}\n\n"
        )
        with self.assertRaises(ProtocolError):
            ChatStream("kimi-k3.1-pro").feed(data)

    def test_tool_arguments_and_thinking_survive_history_replay(self):
        parser = ChatStream("kimi-k3.1-pro")
        for offset in range(0, len(tool_stream()), 7):
            parser.feed(tool_stream()[offset : offset + 7])
        result = parser.finish()
        history = [{"role": "user", "content": "Read the plate specification."}]
        append_tool_results(history, result, {"plate": {"hole_pitch_mm": 60}})
        self.assertEqual(history[1], result.assistant)
        self.assertIsNot(history[1], result.assistant)
        self.assertEqual(history[1]["reasoning_content"], "Read the specification.")
        self.assertEqual(history[2]["tool_call_id"], "call_example")
        self.assertEqual(json.loads(history[2]["content"]), {"hole_pitch_mm": 60})

    def test_no_usage_is_not_zero(self):
        parser = ChatStream("kimi-k3.1-pro")
        parser.feed(event(chunk({"content": "ok"}, "stop")) + b"data: [DONE]\n\n")
        self.assertIsNone(parser.finish().usage)

    def test_truncation_and_disconnect_are_not_completion(self):
        for finish, done in [("length", True), ("stop", False), (None, True)]:
            with (
                self.subTest(finish=finish, done=done),
                self.assertRaises(ProtocolError),
            ):
                parser = ChatStream("kimi-k3.1-pro")
                parser.feed(event(chunk({"content": "partial"}, finish)))
                if done:
                    parser.feed(b"data: [DONE]\n\n")
                parser.finish()

    def test_wrong_model_is_rejected(self):
        parser = ChatStream("kimi-k3.1-pro")
        with self.assertRaises(ProtocolError):
            parser.feed(event({"model": "other-model", "choices": []}))

    def test_unknown_assistant_fields_are_not_silently_discarded(self):
        parser = ChatStream("kimi-k3.1-pro")
        with self.assertRaises(ProtocolError):
            parser.feed(event(chunk({"new_reasoning_field": "important"})))

    def test_malformed_empty_fields_are_not_treated_as_missing(self):
        item = chunk()
        item["choices"][0]["delta"] = []
        parser = ChatStream("kimi-k3.1-pro")
        with self.assertRaises(ProtocolError):
            parser.feed(event(item))

    def test_unknown_tool_metadata_is_not_silently_discarded(self):
        parser = ChatStream("kimi-k3.1-pro")
        with self.assertRaises(ProtocolError):
            parser.feed(
                event(
                    chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_example",
                                    "signature": "future-field",
                                    "function": {
                                        "name": "get_part_spec",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    )
                )
            )

    def test_error_event_is_not_success(self):
        parser = ChatStream("kimi-k3.1-pro")
        with self.assertRaises(ProtocolError):
            parser.feed(
                event({"error": {"code": "provider_error", "message": "not logged"}})
            )

    def test_unknown_tool_does_not_change_history(self):
        parser = ChatStream("kimi-k3.1-pro")
        parser.feed(tool_stream().replace(b"get_part_spec", b"run_shell"))
        history = [{"role": "user", "content": "Read the plate."}]
        with self.assertRaises(ProtocolError):
            append_tool_results(history, parser.finish(), {"plate": {}})
        self.assertEqual(len(history), 1)

    def test_oversize_stream_is_rejected(self):
        parser = ChatStream("kimi-k3.1-pro", max_bytes=100)
        with self.assertRaises(ProtocolError):
            parser.feed(b"x" * 101)


if __name__ == "__main__":
    unittest.main()
