import codecs
import copy
import json
import math
from dataclasses import dataclass
from decimal import Decimal


class ProtocolError(ValueError):
    pass


def _bounded_json(value):
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > 32 or visited > 10000:
            raise ProtocolError("JSON structure exceeds the supported limits")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ProtocolError("JSON object keys must be text")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ProtocolError("non-finite JSON number")
        elif item is not None and type(item) not in (str, int, bool):
            raise ProtocolError("unsupported JSON value")


def decode_json(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError("invalid or overly nested JSON") from exc
    _bounded_json(value)
    return value


@dataclass(frozen=True)
class Usage:
    input_total: int
    input_other: int
    input_cache_read: int
    input_cache_write: int
    output_tokens: int
    reasoning_tokens: int | None
    raw: dict


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{name} must be a nonnegative integer")
    return value


def _object(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be an object")
    return value


def _details(raw, name):
    return _object(raw.get(name), name)


def parse_usage(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProtocolError("usage must be an object")
    _bounded_json(raw)
    total = _count(raw.get("prompt_tokens"), "prompt_tokens")
    output = _count(raw.get("completion_tokens"), "completion_tokens")
    details = _details(raw, "prompt_tokens_details")
    cached = _count(
        details.get("cached_tokens", raw.get("cached_tokens", 0)), "cached_tokens"
    )
    if (
        "cached_tokens" in raw
        and _count(raw["cached_tokens"], "cached_tokens") != cached
    ):
        raise ProtocolError("conflicting cache-read counters")
    written = _count(details.get("cache_write_tokens", 0), "cache_write_tokens")
    if cached + written > total:
        raise ProtocolError("cache counters exceed total input")
    if (
        "total_tokens" in raw
        and _count(raw["total_tokens"], "total_tokens") != total + output
    ):
        raise ProtocolError("total_tokens does not match input plus output")
    reasoning = _details(raw, "completion_tokens_details").get("reasoning_tokens")
    if reasoning is not None and _count(reasoning, "reasoning_tokens") > output:
        raise ProtocolError("reasoning exceeds completion tokens")
    return Usage(
        total,
        total - cached - written,
        cached,
        written,
        output,
        reasoning,
        copy.deepcopy(raw),
    )


RATES = {
    "kimi-k3.1-pro": (Decimal(2), Decimal("0.20"), Decimal(10), Decimal(2)),
    "kimi-k3.1-pro-highspeed": (
        Decimal(4),
        Decimal("0.20"),
        Decimal(20),
        Decimal(4),
    ),
}


def price_usage(usage, model, cache_ttl):
    if model not in RATES:
        raise ProtocolError("unsupported model price")
    if cache_ttl not in ("5m", "1h"):
        raise ProtocolError("unsupported cache lifetime")
    result = {
        "currency": "USD",
        "price_date": "2026-10-24",
        "source": "https://platform.kimi.ai/",
        "known_usd": None,
        "complete": False,
        "unknown": [],
    }
    if usage is None:
        result["unknown"] = ["usage"]
        return result
    ordinary, cached, output, written = RATES[model]
    amount = (
        usage.input_other * ordinary
        + usage.input_cache_read * cached
        + usage.output_tokens * output
    )
    if usage.input_cache_write:
        if cache_ttl == "5m":
            amount += usage.input_cache_write * written
        else:
            result["unknown"].append("cache_write")
    result["known_usd"] = str(amount / Decimal(1_000_000))
    result["complete"] = not result["unknown"]
    return result


@dataclass(frozen=True)
class ChatResult:
    assistant: dict
    usage: Usage | None
    response_model: str
    response_id: str | None
    finish_reason: str


class ChatStream:
    def __init__(self, expected_model, max_bytes=2 * 1024 * 1024):
        self.expected_model = expected_model
        self.max_bytes = max_bytes
        self._bytes = 0
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self._buffer = ""
        self._data = []
        self._content = []
        self._reasoning = []
        self._reasoning_seen = False
        self._calls = {}
        self._usage = None
        self._model = None
        self._id = None
        self._finish = None
        self.done = False

    @property
    def usage_snapshot(self):
        return self._usage

    def feed(self, data):
        self._bytes += len(data)
        if self._bytes > self.max_bytes:
            raise ProtocolError("response exceeds byte limit")
        try:
            self._buffer += self._decoder.decode(data)
        except UnicodeDecodeError as exc:
            raise ProtocolError("invalid UTF-8 stream") from exc
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self._data:
                    self._event("\n".join(self._data))
                    self._data.clear()
            elif line.startswith("data:"):
                self._data.append(line[5:].removeprefix(" "))

    def _event(self, data):
        if self.done:
            raise ProtocolError("data after stream terminator")
        if data == "[DONE]":
            self.done = True
            return
        item = decode_json(data)
        if not isinstance(item, dict):
            raise ProtocolError("event must be an object")
        if "error" in item:
            raise ProtocolError("provider emitted an error event")
        if "model" in item:
            if item["model"] != self.expected_model:
                raise ProtocolError("response model differs from requested model")
            self._model = item["model"]
        if "id" in item:
            if not isinstance(item["id"], str) or (
                self._id is not None and item["id"] != self._id
            ):
                raise ProtocolError("inconsistent response ID")
            self._id = item["id"]
        if item.get("usage") is not None:
            usage = parse_usage(item["usage"])
            if self._usage is not None and (
                usage.input_total < self._usage.input_total
                or usage.output_tokens < self._usage.output_tokens
            ):
                raise ProtocolError("usage counters moved backwards")
            self._usage = usage
        choices = item.get("choices")
        if not isinstance(choices, list):
            raise ProtocolError("choices must be an array")
        if not choices:
            return
        if (
            len(choices) != 1
            or not isinstance(choices[0], dict)
            or type(choices[0].get("index")) is not int
            or choices[0]["index"] != 0
        ):
            raise ProtocolError("only one choice is supported")
        choice = choices[0]
        if set(choice) - {"index", "delta", "finish_reason", "logprobs"}:
            raise ProtocolError("unsupported choice field")
        if not isinstance(choice.get("delta"), dict):
            raise ProtocolError("choice.delta must be an object")
        delta = choice["delta"]
        if set(delta) - {"role", "content", "reasoning_content", "tool_calls"}:
            raise ProtocolError("unsupported assistant delta field")
        if delta.get("role") not in (None, "assistant"):
            raise ProtocolError("unexpected assistant role")
        if self._finish is not None and any(
            delta.get(key) for key in ("content", "reasoning_content", "tool_calls")
        ):
            raise ProtocolError("assistant data after finish reason")
        for key, target in [
            ("content", self._content),
            ("reasoning_content", self._reasoning),
        ]:
            value = delta.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ProtocolError(f"{key} must be text")
                target.append(value)
            if key == "reasoning_content" and key in delta:
                self._reasoning_seen = True
        calls = delta.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            raise ProtocolError("tool_calls must be an array")
        for call in calls:
            if not isinstance(call, dict) or set(call) - {
                "index",
                "id",
                "type",
                "function",
            }:
                raise ProtocolError("unsupported tool-call fields")
            index = call.get("index")
            if type(index) is not int or not 0 <= index < 8:
                raise ProtocolError("invalid tool-call index")
            saved = self._calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if call.get("type") not in (None, "function"):
                raise ProtocolError("unsupported tool-call type")
            if "id" in call:
                if not isinstance(call["id"], str) or not call["id"]:
                    raise ProtocolError("invalid tool-call ID")
                if saved["id"] is not None and saved["id"] != call["id"]:
                    raise ProtocolError("tool-call ID changed")
                saved["id"] = call["id"]
            function = _object(call.get("function"), "function")
            if set(function) - {"name", "arguments"}:
                raise ProtocolError("invalid function delta")
            for key in ("name", "arguments"):
                if key in function:
                    if not isinstance(function[key], str):
                        raise ProtocolError("function fields must be text")
                    saved["function"][key] += function[key]
        reason = choice.get("finish_reason")
        if reason is not None:
            if not isinstance(reason, str) or (
                self._finish is not None and self._finish != reason
            ):
                raise ProtocolError("inconsistent finish reason")
            self._finish = reason

    def finish(self):
        try:
            remainder = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ProtocolError("incomplete UTF-8 stream") from exc
        if remainder or self._buffer.strip() or self._data or not self.done:
            raise ProtocolError("stream ended before a complete terminator")
        if self._model is None:
            raise ProtocolError("response did not identify its model")
        if self._finish not in ("stop", "tool_calls"):
            raise ProtocolError(f"incomplete response: {self._finish}")
        assistant = {"role": "assistant", "content": "".join(self._content) or None}
        if self._reasoning_seen:
            assistant["reasoning_content"] = "".join(self._reasoning)
        if self._calls:
            if self._finish != "tool_calls" or sorted(self._calls) != list(
                range(len(self._calls))
            ):
                raise ProtocolError("incomplete tool-call sequence")
            calls = [self._calls[index] for index in sorted(self._calls)]
            ids = [call["id"] for call in calls]
            if any(not value for value in ids) or len(set(ids)) != len(ids):
                raise ProtocolError("missing or duplicate tool-call IDs")
            if any(not call["function"]["name"] for call in calls):
                raise ProtocolError("missing tool name")
            assistant["tool_calls"] = copy.deepcopy(calls)
        elif self._finish == "tool_calls":
            raise ProtocolError("tool-call finish without calls")
        return ChatResult(assistant, self._usage, self._model, self._id, self._finish)


READ_PART_TOOL = {
    "type": "function",
    "function": {
        "name": "get_part_spec",
        "description": "Read a synthetic part specification from the local example catalog.",
        "parameters": {
            "type": "object",
            "properties": {"part_id": {"type": "string", "enum": ["plate"]}},
            "required": ["part_id"],
            "additionalProperties": False,
        },
    },
}


def append_tool_results(history, result, catalog):
    if result.finish_reason != "tool_calls":
        raise ProtocolError("response did not request tools")
    pending = []
    for call in result.assistant["tool_calls"]:
        if call["function"]["name"] != "get_part_spec":
            raise ProtocolError("tool is not in the read-only catalog")
        args = decode_json(call["function"]["arguments"])
        if (
            not isinstance(args, dict)
            or set(args) != {"part_id"}
            or not isinstance(args["part_id"], str)
            or args["part_id"] not in catalog
        ):
            raise ProtocolError("invalid catalog request")
        pending.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(
                    catalog[args["part_id"]], ensure_ascii=False, allow_nan=False
                ),
            }
        )
    history.extend([copy.deepcopy(result.assistant), *pending])
