import asyncio
import json
import unittest
from decimal import Decimal

import httpx

from notes.examples.kimi_k31.client import (
    BudgetExceeded,
    Limits,
    NativeChatClient,
    RequestFailed,
    RunBudget,
)


class Chunks(httpx.AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts

    async def __aiter__(self):
        for part in self.parts:
            yield part


class NeverEnding(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b": connected\n\n"
        await asyncio.Event().wait()


def response_bytes():
    events = [
        {
            "model": "kimi-k3.1-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "model": "kimi-k3.1-pro",
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        },
    ]
    return [
        *(("data: " + json.dumps(item) + "\n\n").encode() for item in events),
        b"data: [DONE]\n\n",
    ]


class BudgetTests(unittest.TestCase):
    def test_reserves_before_request_and_never_refunds_unknown_work(self):
        budget = RunBudget(Limits(max_requests=2))
        budget.reserve(b"{}", 4096)
        budget.reserve(b"{}", 4096)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(b"{}", 4096)
        self.assertEqual(budget.requests, 2)
        self.assertGreater(budget.reserved_usd, 0)

    def test_bytes_output_money_and_elapsed_limits(self):
        now = [0.0]
        budget = RunBudget(Limits(), clock=lambda: now[0])
        for body, output in [(b"x" * 16385, 1), (b"{}", 4097)]:
            with self.subTest(output=output), self.assertRaises(BudgetExceeded):
                budget.reserve(body, output)
        cheap = RunBudget(Limits(max_usd=Decimal("0.01")))
        with self.assertRaises(BudgetExceeded):
            cheap.reserve(b"{}", 4096)
        now[0] = 601
        with self.assertRaises(BudgetExceeded):
            budget.reserve(b"{}", 1)
        self.assertEqual(budget.requests, 0)

    def test_limits_cannot_exceed_authorized_caps(self):
        for kwargs in [
            {"max_requests": 7},
            {"max_completion_tokens": 4097},
            {"max_body_bytes": 16385},
            {"timeout_seconds": 601},
            {"max_usd": Decimal("2.01")},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Limits(**kwargs)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_payload_and_final_usage(self):
        seen = []

        async def handle(request):
            seen.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Chunks(response_bytes()),
            )

        budget = RunBudget(Limits(max_requests=1))
        async with NativeChatClient(
            "YOUR_API_KEY", budget, transport=httpx.MockTransport(handle)
        ) as client:
            result = await client.complete(
                [{"role": "user", "content": "Return JSON."}], json_output=True
            )
            self.assertEqual(json.loads(result.assistant["content"]), {"ok": True})
            self.assertEqual(result.usage.input_total, 8)
            self.assertTrue(client.records[0]["cost"]["complete"])
            self.assertNotIn("YOUR_API_KEY", json.dumps(client.records))
        payload = json.loads(seen[0].content)
        self.assertEqual(
            str(seen[0].url), "https://api.moonshot.ai/v1/chat/completions"
        )
        self.assertEqual(
            payload["prompt_cache_options"], {"mode": "implicit", "ttl": "5m"}
        )
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", payload)
        self.assertEqual(seen[0].headers["Authorization"], "Bearer YOUR_API_KEY")

    async def test_http_error_is_not_retried_and_does_not_log_body(self):
        calls = []

        async def handle(request):
            calls.append(request)
            return httpx.Response(
                429, json={"error": {"code": "rate_limit", "message": "YOUR_API_KEY"}}
            )

        async with NativeChatClient(
            "YOUR_API_KEY", RunBudget(Limits()), transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaises(RequestFailed) as caught:
                await client.complete([{"role": "user", "content": "Hello"}])
            self.assertEqual(caught.exception.status_code, 429)
            self.assertEqual(len(calls), 1)
            self.assertFalse(client.records[0]["request_completed"])
            self.assertIsNone(client.records[0]["cost"]["known_usd"])
            self.assertNotIn("YOUR_API_KEY", json.dumps(client.records))

    async def test_error_code_cannot_echo_the_credential(self):
        async def handle(request):
            return httpx.Response(401, json={"error": {"code": "YOUR_API_KEY"}})

        async with NativeChatClient(
            "YOUR_API_KEY", RunBudget(Limits()), transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaises(RequestFailed) as caught:
                await client.complete([{"role": "user", "content": "Hello"}])
            self.assertNotIn("YOUR_API_KEY", str(caught.exception))
            self.assertNotIn("YOUR_API_KEY", json.dumps(client.records))

    async def test_redirects_are_not_followed(self):
        calls = []

        async def handle(request):
            calls.append(str(request.url))
            return httpx.Response(
                302, headers={"location": "https://example.test/other"}
            )

        async with NativeChatClient(
            "YOUR_API_KEY", RunBudget(Limits()), transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaises(RequestFailed):
                await client.complete([{"role": "user", "content": "Hello"}])
        self.assertEqual(len(calls), 1)

    async def test_budget_blocks_before_transport(self):
        calls = []

        async def handle(request):
            calls.append(request)
            return httpx.Response(200)

        async with NativeChatClient(
            "YOUR_API_KEY", RunBudget(Limits()), transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaises(BudgetExceeded):
                await client.complete([{"role": "user", "content": "x" * 17000}])
        self.assertEqual(calls, [])

    async def test_done_terminates_without_waiting_for_socket_close(self):
        class DoneThenWait(httpx.AsyncByteStream):
            async def __aiter__(self):
                for data in response_bytes():
                    yield data
                await asyncio.Event().wait()

        async def handle(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=DoneThenWait(),
            )

        async with NativeChatClient(
            "YOUR_API_KEY",
            RunBudget(Limits(timeout_seconds=0.1)),
            transport=httpx.MockTransport(handle),
        ) as client:
            result = await client.complete(
                [{"role": "user", "content": "Return JSON."}]
            )
            self.assertEqual(result.finish_reason, "stop")

    async def test_outer_deadline_closes_an_unfinished_stream(self):
        async def handle(request):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=NeverEnding()
            )

        async with NativeChatClient(
            "YOUR_API_KEY",
            RunBudget(Limits(timeout_seconds=0.03)),
            transport=httpx.MockTransport(handle),
        ) as client:
            with self.assertRaises(TimeoutError):
                await client.complete([{"role": "user", "content": "Hello"}])
            self.assertFalse(client.records[0]["request_completed"])
            self.assertEqual(client.records[0]["error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
