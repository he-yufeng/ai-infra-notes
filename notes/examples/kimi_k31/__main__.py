import argparse
import asyncio
import copy
import json
import os
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import trimesh

from .client import BudgetExceeded, Limits, NativeChatClient, RequestFailed, RunBudget
from .fixtures import catalog_answer, catalog_call
from .geometry import (
    DEFAULT_SPEC,
    GeometryError,
    PlateSpec,
    build_plate,
    validate_revision,
    verify_plate,
)
from .protocol import (
    READ_PART_TOOL,
    ChatStream,
    ProtocolError,
    append_tool_results,
    decode_json,
    price_usage,
)
from .render import render_mesh


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


def artifact(out, name, spec, mode):
    mesh = build_plate(spec)
    verification = verify_plate(mesh, spec)
    stl, preview = f"{name}.stl", f"{name}.png"
    mesh.export(out / stl)
    reloaded = trimesh.load_mesh(out / stl, process=True)
    roundtrip = verify_plate(reloaded, spec)
    render_mesh(reloaded, spec, out / preview, mode)
    return {
        "parameters": spec.to_dict(),
        "verification": verification,
        "stl_roundtrip": roundtrip,
        "stl": stl,
        "preview": preview,
        "accepted": verification["accepted"] and roundtrip["accepted"],
    }


def read_fixture(data):
    parser = ChatStream("kimi-k3.1-pro")
    for offset in range(0, len(data), 11):
        parser.feed(data[offset : offset + 11])
    return parser.finish()


def offline(out):
    out.mkdir()
    history = [
        {"role": "user", "content": "Read the plate catalog and return its hole pitch."}
    ]
    first = read_fixture(catalog_call())
    append_tool_results(history, first, {"plate": DEFAULT_SPEC.to_dict()})
    second = read_fixture(catalog_answer())
    history.append(copy.deepcopy(second.assistant))
    protocol_ok = json.loads(second.assistant["content"]) == {"hole_pitch_mm": 60}
    usage = [asdict(first.usage), asdict(second.usage)]
    costs = [
        price_usage(first.usage, "kimi-k3.1-pro", "5m"),
        price_usage(second.usage, "kimi-k3.1-pro", "5m"),
    ]
    enlarged = replace(DEFAULT_SPEC, hole_diameter_mm=8)
    wider = replace(enlarged, width_mm=70)
    validate_revision(DEFAULT_SPEC, enlarged, {"hole_diameter_mm": 8})
    validate_revision(enlarged, wider, {"width_mm": 70})
    states = [
        artifact(out, name, spec, "Offline reference: fixed parameters, no model call")
        for name, spec in [
            ("initial", DEFAULT_SPEC),
            ("larger_holes", enlarged),
            ("wider_plate", wider),
        ]
    ]
    negatives = []
    for name, mesh in [
        ("missing_holes", trimesh.creation.box(extents=(100, 60, 8))),
        (
            "wrong_pitch_same_volume",
            build_plate(replace(DEFAULT_SPEC, hole_pitch_mm=50)),
        ),
        ("wrong_width", build_plate(replace(DEFAULT_SPEC, width_mm=61))),
    ]:
        negatives.append(
            {"case": name, "verification": verify_plate(mesh, DEFAULT_SPEC)}
        )
    report = {
        "mode": "offline",
        "source": "Synthetic SSE fixtures and fixed geometry parameters, not model outputs",
        "live_requests": 0,
        "protocol": {
            "accepted": protocol_ok,
            "history": history,
            "usage": usage,
            "illustrative_costs": costs,
        },
        "geometry": {"states": states, "negative_cases": negatives},
    }
    report["accepted"] = (
        protocol_ok
        and all(state["accepted"] for state in states)
        and all(not item["verification"]["accepted"] for item in negatives)
    )
    write_json(out / "report.json", report)
    return report


def parse_answer(result):
    if result.finish_reason != "stop" or not isinstance(
        result.assistant.get("content"), str
    ):
        raise ProtocolError("expected a completed JSON answer")
    value = decode_json(result.assistant["content"])
    if not isinstance(value, dict):
        raise ProtocolError("answer must be an object")
    return value


async def live(out, api_key, model, effort):
    out.mkdir()
    limits = Limits()
    config = {
        "model": model,
        "reasoning_effort": effort,
        "max_requests": limits.max_requests,
        "max_completion_tokens": limits.max_completion_tokens,
        "max_body_bytes": limits.max_body_bytes,
        "timeout_seconds": limits.timeout_seconds,
        "max_usd": str(limits.max_usd),
        "implicit_retries": 0,
        "cache_ttl": "5m",
        "input_scope": "synthetic public plate parameters only",
    }
    write_json(out / "run-config.json", config)
    if not api_key or api_key.startswith("sk-or-"):
        reason = (
            "MOONSHOT_API_KEY is not configured"
            if not api_key
            else "MOONSHOT_API_KEY is not a native platform key"
        )
        report = {
            "mode": "live",
            "status": "blocked_resource",
            "reason": reason,
            "requests": [],
            "accepted": False,
        }
        write_json(out / "report.json", report)
        return report
    budget = RunBudget(limits)
    report = {
        "mode": "live",
        "status": "incomplete",
        "accepted": False,
        "task_outputs_accepted": False,
        "geometry": [],
        "protocol": None,
    }
    client = NativeChatClient(api_key, budget, model=model, effort=effort)
    try:
        async with asyncio.timeout(budget.remaining_seconds()), client:
            history = [
                {
                    "role": "system",
                    "content": "Use the read-only catalog tool. Then return only a JSON object containing hole_pitch_mm.",
                },
                {
                    "role": "user",
                    "content": "Read part plate and report its hole pitch in millimeters.",
                },
            ]
            call = await client.complete(
                history, tools=[READ_PART_TOOL], tool_choice="required"
            )
            append_tool_results(history, call, {"plate": DEFAULT_SPEC.to_dict()})
            answer = await client.complete(
                history, tools=[READ_PART_TOOL], tool_choice="none", json_output=True
            )
            protocol_ok = parse_answer(answer) == {"hole_pitch_mm": 60}
            history.append(copy.deepcopy(answer.assistant))
            report["protocol"] = {"accepted": protocol_ok, "history": history}
            if not protocol_ok:
                raise ProtocolError("catalog answer failed verification")
            design_history = [
                {
                    "role": "system",
                    "content": "Return only a JSON object with exactly length_mm, width_mm, thickness_mm, hole_diameter_mm, hole_pitch_mm. All values are numbers in millimeters. Do not return code, extra keys or markdown.",
                }
            ]
            prompts = [
                "Create a plate 100 mm long, 60 mm wide and 8 mm thick, with two 6 mm diameter through-holes. Their centers are 60 mm apart and symmetric about the plate center.",
                "Change only the hole diameter to 8 mm. Keep every other parameter unchanged.",
                "Change only the plate width to 70 mm. Keep every other parameter unchanged.",
            ]
            previous = None
            for index, prompt in enumerate(prompts):
                budget.remaining_seconds()
                design_history.append({"role": "user", "content": prompt})
                result = await client.complete(design_history, json_output=True)
                spec = PlateSpec.from_dict(parse_answer(result))
                if previous is None:
                    if spec != DEFAULT_SPEC:
                        raise GeometryError(
                            "initial parameters do not match the requested plate"
                        )
                else:
                    validate_revision(
                        previous,
                        spec,
                        {"hole_diameter_mm": 8} if index == 1 else {"width_mm": 70},
                    )
                state = artifact(
                    out,
                    f"model_state_{index}",
                    spec,
                    "Live model parameters; geometry generated and checked locally",
                )
                report["geometry"].append(state)
                design_history.append(copy.deepcopy(result.assistant))
                if not state["accepted"]:
                    raise GeometryError("generated geometry failed verification")
                previous = spec
            budget.remaining_seconds()
            report["design_history"] = design_history
            report["task_outputs_accepted"] = True
        budget.remaining_seconds()
        report["accepted"] = True
        report["status"] = "completed"
    except RequestFailed as exc:
        report["status"] = (
            "blocked_resource" if exc.status_code in (401, 402, 403, 429) else "failed"
        )
        report["http_status"] = exc.status_code
        report["provider_error_code"] = exc.code
        report["error_type"] = type(exc).__name__
    except (
        ProtocolError,
        GeometryError,
        BudgetExceeded,
        httpx.HTTPError,
        TimeoutError,
    ) as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
    finally:
        report["requests"] = client.records
        report["budget"] = budget.snapshot()
        report["usage_complete"] = bool(client.records) and all(
            item.get("usage_is_final") for item in client.records
        )
        report["cost_estimates_complete"] = bool(client.records) and all(
            item.get("cost", {}).get("complete") for item in client.records
        )
        if report["accepted"] and not (
            report["usage_complete"] and report["cost_estimates_complete"]
        ):
            report["accepted"] = False
            report["status"] = "incomplete_accounting"
        report["scope"] = (
            "Protocol and parameter-edit smoke test, not BenchCAD, a model ranking or manufacturing validation."
        )
        report = client.redact(report)
        write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Offline-first Kimi Chat and parameterized geometry examples."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make at most six bounded native API requests using MOONSHOT_API_KEY.",
    )
    parser.add_argument(
        "--model",
        choices=["kimi-k3.1-pro", "kimi-k3.1-pro-highspeed"],
        default="kimi-k3.1-pro",
    )
    parser.add_argument("--effort", choices=["low", "high", "max"], default="low")
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output or Path(".artifacts") / f"kimi-k31-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    reference = offline(out / "offline")
    if not reference["accepted"]:
        print(json.dumps({"status": "offline_verification_failed", "output": str(out)}))
        return 1
    summary = {
        "offline_accepted": True,
        "output": str(out),
        "live_requested": args.live,
    }
    if args.live:
        report = asyncio.run(
            live(
                out / "live",
                os.environ.get("MOONSHOT_API_KEY"),
                args.model,
                args.effort,
            )
        )
        summary.update(
            {
                "live_status": report["status"],
                "live_requests": len(report["requests"]),
                "live_accepted": report["accepted"],
            }
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if report["accepted"] else 2
    summary["live_requests"] = 0
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
