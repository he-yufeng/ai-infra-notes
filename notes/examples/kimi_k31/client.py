import asyncio
import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import httpx

from .protocol import RATES, ChatStream, ProtocolError, price_usage

ENDPOINT = "https://api.moonshot.ai/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


class RequestFailed(RuntimeError):
    def __init__(self, status_code, code="provider_error"):
        self.status_code = status_code
        self.code = code
        super().__init__(f"HTTP {status_code}: {code}")


@dataclass(frozen=True)
class Limits:
    max_requests: int = 6
    max_completion_tokens: int = 4096
    max_body_bytes: int = 16 * 1024
    timeout_seconds: float = 600
    max_usd: Decimal = Decimal(2)

    def __post_init__(self):
        for name, cap in [
            ("max_requests", 6),
            ("max_completion_tokens", 4096),
            ("max_body_bytes", 16 * 1024),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= cap:
                raise ValueError(f"{name} exceeds the supported budget")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 600
        ):
            raise ValueError("invalid wall-clock budget")
        if (
            not isinstance(self.max_usd, Decimal)
            or not self.max_usd.is_finite()
            or not 0 < self.max_usd <= 2
        ):
            raise ValueError("invalid monetary budget")


class RunBudget:
    def __init__(self, limits=None, clock=time.monotonic):
        self.limits = limits or Limits()
        self.clock = clock
        self.started = clock()
        self.requests = 0
        self.reserved_usd = Decimal(0)

    def remaining_seconds(self):
        remaining = self.limits.timeout_seconds - (self.clock() - self.started)
        if remaining <= 0:
            raise BudgetExceeded("wall-clock budget exhausted")
        return remaining

    def reserve(self, body, output_limit):
        self.remaining_seconds()
        if len(body) > self.limits.max_body_bytes:
            raise BudgetExceeded("request body exceeds the byte budget")
        if (
            type(output_limit) is not int
            or not 0 < output_limit <= self.limits.max_completion_tokens
        ):
            raise BudgetExceeded("completion limit exceeds the budget")
        if self.requests >= self.limits.max_requests:
            raise BudgetExceeded("request count exhausted")
        # Reserve using byte count plus framing margin and rates above either supported model.
        reservation = (
            Decimal(len(body) + 2048) * 10 + Decimal(output_limit) * 20
        ) / 1_000_000
        if self.reserved_usd + reservation > self.limits.max_usd:
            raise BudgetExceeded("conservative cost reservation exceeds the budget")
        self.requests += 1
        self.reserved_usd += reservation
        return reservation

    def snapshot(self):
        return {
            "requests": self.requests,
            "reserved_usd": str(self.reserved_usd),
            "elapsed_seconds": self.clock() - self.started,
            "max_requests": self.limits.max_requests,
            "max_body_bytes": self.limits.max_body_bytes,
            "max_completion_tokens": self.limits.max_completion_tokens,
            "timeout_seconds": self.limits.timeout_seconds,
            "max_usd": str(self.limits.max_usd),
        }


class NativeChatClient:
    def __init__(
        self, api_key, budget, model="kimi-k3.1-pro", effort="low", transport=None
    ):
        if (
            not isinstance(api_key, str)
            or not api_key
            or api_key.strip() != api_key
            or any(char in api_key for char in "\r\n")
        ):
            raise ValueError("a nonempty native API key is required")
        if api_key.startswith("sk-or-"):
            raise ValueError("an OpenRouter key cannot be sent to the native endpoint")
        if model not in RATES or effort not in ("low", "high", "max"):
            raise ValueError("unsupported model or reasoning effort")
        self._api_key = api_key
        self.budget = budget
        self.model = model
        self.effort = effort
        self._transport = transport
        self._http = None
        self.records = []

    async def __aenter__(self):
        self._http = httpx.AsyncClient(
            transport=self._transport or httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(90, connect=10, pool=10),
        )
        return self

    async def __aexit__(self, *args):
        try:
            await self._http.aclose()
        finally:
            self._http = None

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {self.redact(key): self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value

    async def _error_code(self, response):
        data = bytearray()
        async for block in response.aiter_bytes():
            data.extend(block[: max(0, 8192 - len(data))])
            if len(data) >= 8192:
                break
        try:
            value = json.loads(data)
            code = value.get("error", {}).get("code")
        except (ValueError, AttributeError, RecursionError):
            return "provider_error"
        if not isinstance(code, str) or self._api_key in code:
            return "provider_error"
        return code if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else "provider_error"

    async def complete(self, messages, tools=None, tool_choice=None, json_output=False):
        if self._http is None:
            raise RuntimeError("use the client as an async context manager")
        if not isinstance(messages, list) or not messages:
            raise ProtocolError("messages must be a nonempty list")
        if tool_choice is not None and (
            not tools or tool_choice not in ("auto", "none", "required")
        ):
            raise ProtocolError("unsupported tool choice")
        payload = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "reasoning_effort": self.effort,
            "max_completion_tokens": self.budget.limits.max_completion_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
        }
        if tools is not None:
            payload["tools"] = copy.deepcopy(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        reservation = self.budget.reserve(body, payload["max_completion_tokens"])
        record = {
            "request_index": self.budget.requests,
            "started_at": datetime.now(UTC).isoformat(),
            "endpoint": ENDPOINT,
            "model": self.model,
            "reasoning_effort": self.effort,
            "request_bytes": len(body),
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "completion_limit": payload["max_completion_tokens"],
            "cache_ttl": "5m",
            "reserved_usd": str(reservation),
            "request_completed": False,
        }
        self.records.append(record)
        started = self.budget.clock()
        parser = ChatStream(self.model)
        try:
            async with asyncio.timeout(self.budget.remaining_seconds()):
                async with self._http.stream(
                    "POST",
                    ENDPOINT,
                    content=body,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Accept-Encoding": "identity",
                    },
                    follow_redirects=False,
                ) as response:
                    record["http_status"] = response.status_code
                    if response.status_code != 200:
                        raise RequestFailed(
                            response.status_code, await self._error_code(response)
                        )
                    if not response.headers.get("content-type", "").startswith(
                        "text/event-stream"
                    ):
                        raise ProtocolError("expected an event-stream response")
                    async for block in response.aiter_bytes():
                        parser.feed(block)
                        if parser.done:
                            break
                result = parser.finish()
            record["request_completed"] = True
            record["response_model"] = result.response_model
            record["response_id"] = result.response_id
            record["finish_reason"] = result.finish_reason
            return result
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            if isinstance(exc, (ProtocolError, RequestFailed, BudgetExceeded)):
                record["error_detail"] = self.redact(str(exc))[:200]
            if isinstance(exc, RequestFailed):
                record["provider_error_code"] = exc.code
            raise
        finally:
            record["elapsed_seconds"] = self.budget.clock() - started
            usage = parser.usage_snapshot
            record["raw_usage"] = copy.deepcopy(usage.raw) if usage else None
            record["usage_is_final"] = bool(parser.done and usage is not None)
            record["cost"] = price_usage(
                usage if parser.done else None, self.model, "5m"
            )
            redacted = self.redact(record)
            record.clear()
            record.update(redacted)
