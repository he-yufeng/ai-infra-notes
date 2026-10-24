import json

MODEL = "kimi-k3.1-pro"


def _event(delta=None, finish=None, usage=None):
    value = {"id": "chatcmpl-synthetic", "model": MODEL, "choices": []}
    if delta is not None or finish is not None:
        value["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        value["usage"] = usage
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def catalog_call():
    return b"".join(
        [
            _event(
                {
                    "role": "assistant",
                    "reasoning_content": "Read the synthetic catalog before answering.",
                }
            ),
            _event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_catalog",
                            "type": "function",
                            "function": {
                                "name": "get_part_spec",
                                "arguments": '{"part_',
                            },
                        }
                    ]
                }
            ),
            _event(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": 'id":"plate"}'}}
                    ]
                },
                "tool_calls",
            ),
            _event(
                usage={
                    "prompt_tokens": 373,
                    "completion_tokens": 112,
                    "total_tokens": 485,
                    "prompt_tokens_details": {"cache_write_tokens": 256},
                    "completion_tokens_details": {"reasoning_tokens": 90},
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


def catalog_answer():
    return b"".join(
        [
            _event(
                {
                    "role": "assistant",
                    "reasoning_content": "The catalog provides the required pitch.",
                }
            ),
            _event({"content": '{"hole_pitch_mm":'}),
            _event({"content": "60}"}, "stop"),
            _event(
                usage={
                    "prompt_tokens": 1000,
                    "completion_tokens": 40,
                    "total_tokens": 1040,
                    "prompt_tokens_details": {
                        "cached_tokens": 600,
                        "cache_write_tokens": 256,
                    },
                    "completion_tokens_details": {"reasoning_tokens": 20},
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )
