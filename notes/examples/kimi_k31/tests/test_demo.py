import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from notes.examples.kimi_k31.__main__ import live, offline
from notes.examples.kimi_k31.client import NativeChatClient
from notes.examples.kimi_k31.fixtures import catalog_answer, catalog_call
from notes.examples.kimi_k31.geometry import DEFAULT_SPEC

ROOT = Path(__file__).resolve().parents[4]


class Bytes(httpx.AsyncByteStream):
    def __init__(self, value):
        self.value = value

    async def __aiter__(self):
        for offset in range(0, len(self.value), 19):
            yield self.value[offset : offset + 19]


def json_stream(value):
    items = [
        {
            "model": "kimi-k3.1-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "Preserve the requested constraints.",
                        "content": json.dumps(value),
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "model": "kimi-k3.1-pro",
            "choices": [],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        },
    ]
    return (
        b"".join(("data: " + json.dumps(item) + "\n\n").encode() for item in items)
        + b"data: [DONE]\n\n"
    )


class DemoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        (ROOT / ".artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix="test-k31-", dir=ROOT / ".artifacts"
        )
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def test_offline_writes_actual_stl_and_negative_results(self):
        report = offline(self.out / "offline")
        self.assertTrue(report["accepted"])
        self.assertEqual(report["live_requests"], 0)
        for state in report["geometry"]["states"]:
            self.assertTrue(state["stl_roundtrip"]["accepted"])
            self.assertGreater(
                (self.out / "offline" / state["stl"]).stat().st_size, 1000
            )
            self.assertTrue((self.out / "offline" / state["preview"]).exists())
        self.assertTrue(
            all(
                not item["verification"]["accepted"]
                for item in report["geometry"]["negative_cases"]
            )
        )

    async def test_five_request_workflow_preserves_history_and_edits(self):
        requests = []
        initial = DEFAULT_SPEC.to_dict()
        enlarged = {**initial, "hole_diameter_mm": 8}
        wider = {**enlarged, "width_mm": 70}
        replies = [
            catalog_call(),
            catalog_answer(),
            json_stream(initial),
            json_stream(enlarged),
            json_stream(wider),
        ]

        async def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Bytes(replies[len(requests) - 1]),
            )

        def client(key, budget, **kwargs):
            return NativeChatClient(
                key, budget, transport=httpx.MockTransport(handle), **kwargs
            )

        with patch(
            "notes.examples.kimi_k31.__main__.NativeChatClient", side_effect=client
        ):
            result = await live(
                self.out / "live", "YOUR_API_KEY", "kimi-k3.1-pro", "low"
            )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["accepted"])
        self.assertTrue(result["usage_complete"])
        self.assertEqual(len(requests), 5)
        replay = requests[1]["messages"]
        self.assertEqual(
            replay[-2]["reasoning_content"],
            "Read the synthetic catalog before answering.",
        )
        self.assertEqual(replay[-1]["tool_call_id"], "call_catalog")
        self.assertEqual(
            requests[4]["messages"][-2]["reasoning_content"],
            "Preserve the requested constraints.",
        )
        self.assertEqual(result["geometry"][-1]["parameters"], wider)
        self.assertNotIn(
            "YOUR_API_KEY", (self.out / "live" / "report.json").read_text()
        )

    async def test_client_close_failure_is_not_accepted(self):
        initial = DEFAULT_SPEC.to_dict()
        enlarged = {**initial, "hole_diameter_mm": 8}
        wider = {**enlarged, "width_mm": 70}
        replies = iter(
            [
                catalog_call(),
                catalog_answer(),
                json_stream(initial),
                json_stream(enlarged),
                json_stream(wider),
            ]
        )

        async def handle(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Bytes(next(replies)),
            )

        class BrokenClose(httpx.MockTransport):
            async def aclose(self):
                raise httpx.ReadError("synthetic close failure")

        def client(key, budget, **kwargs):
            return NativeChatClient(
                key, budget, transport=BrokenClose(handle), **kwargs
            )

        with patch(
            "notes.examples.kimi_k31.__main__.NativeChatClient", side_effect=client
        ):
            result = await live(
                self.out / "close-failure", "YOUR_API_KEY", "kimi-k3.1-pro", "low"
            )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["accepted"])
        self.assertTrue(result["task_outputs_accepted"])

    async def test_no_key_is_an_explicit_resource_block(self):
        result = await live(self.out / "no-key", None, "kimi-k3.1-pro", "low")
        self.assertEqual(result["status"], "blocked_resource")
        self.assertEqual(result["requests"], [])
        self.assertFalse(result["accepted"])

    def test_cli_defaults_to_offline_and_refuses_existing_output(self):
        target = self.out / "cli"
        command = [
            sys.executable,
            "-m",
            "notes.examples.kimi_k31",
            "--output",
            str(target),
        ]
        result = subprocess.run(
            command,
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "MOONSHOT_API_KEY": ""},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["live_requests"], 0)
        marker = target / "keep.txt"
        marker.write_text("keep this file")
        again = subprocess.run(
            command, check=False, cwd=ROOT, capture_output=True, text=True
        )
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(marker.read_text(), "keep this file")


if __name__ == "__main__":
    unittest.main()
