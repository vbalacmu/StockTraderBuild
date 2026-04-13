"""LLM strategy agents via Ollama (parallel-safe: no shared mutable state per call)."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ollama


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_prompt(name: str) -> str:
    path = _project_root() / "prompts" / name
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in model output: {text[:500]!r}")
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model output: {e}; excerpt={text[start : start + 800]!r}") from e
    if not isinstance(obj, dict):
        raise ValueError("Model JSON must be an object")
    return obj


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2")


def _ollama_host() -> str | None:
    return os.environ.get("OLLAMA_HOST")


def _clamp_temp(t: float) -> float:
    return max(0.0, min(2.0, t))


def _temperature_strategy_a() -> float:
    if "OLLAMA_TEMPERATURE_STRATEGY_A" in os.environ:
        return _clamp_temp(float(os.environ["OLLAMA_TEMPERATURE_STRATEGY_A"]))
    if "OLLAMA_TEMPERATURE" in os.environ:
        return _clamp_temp(float(os.environ["OLLAMA_TEMPERATURE"]))
    return 0.32


def _temperature_strategy_b() -> float:
    if "OLLAMA_TEMPERATURE_STRATEGY_B" in os.environ:
        return _clamp_temp(float(os.environ["OLLAMA_TEMPERATURE_STRATEGY_B"]))
    if "OLLAMA_TEMPERATURE" in os.environ:
        return _clamp_temp(float(os.environ["OLLAMA_TEMPERATURE"]))
    return 0.52


def _ollama_options_for_strategy_role(role: str) -> dict[str, Any]:
    n = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))
    if role == "strategy_a":
        temp = _temperature_strategy_a()
    else:
        temp = _temperature_strategy_b()
    return {"num_predict": max(256, n), "temperature": temp}


def _ollama_options_evaluator() -> dict[str, Any]:
    n = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))
    if "OLLAMA_TEMPERATURE_EVALUATOR" in os.environ:
        temp = _clamp_temp(float(os.environ["OLLAMA_TEMPERATURE_EVALUATOR"]))
    elif "OLLAMA_TEMPERATURE" in os.environ:
        temp = _clamp_temp(float(os.environ["OLLAMA_TEMPERATURE"]))
    else:
        temp = 0.35
    return {"num_predict": max(256, n), "temperature": temp}


def _ollama_options() -> dict[str, Any]:
    """Default options when a single temperature is enough (e.g. re-exports, tests)."""
    n = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))
    temp = float(os.environ.get("OLLAMA_TEMPERATURE", "0.4"))
    return {"num_predict": max(256, n), "temperature": _clamp_temp(temp)}


def run_strategy_agent(
    *,
    role: str,
    strategy_name: str,
    prompt_filename: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    system = _load_prompt(prompt_filename)
    user = json.dumps({"market_data": market_data}, indent=2)

    kwargs: dict[str, Any] = {
        "model": _ollama_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": _ollama_options_for_strategy_role(role),
        "format": "json",
    }
    host = _ollama_host()
    if host:
        client = ollama.Client(host=host)
        resp = client.chat(**kwargs)
    else:
        resp = ollama.chat(**kwargs)

    content = resp["message"]["content"]
    parsed = _extract_json_object(content)
    decision = str(parsed.get("decision", "")).upper().strip()
    if decision not in {"BUY", "HOLD", "SELL"}:
        raise ValueError(f"Invalid decision from {role}: {decision!r}")
    conf = int(parsed.get("confidence", 0))
    conf = max(1, min(10, conf))
    justification = str(parsed.get("justification", "")).strip()
    if not justification:
        raise ValueError(f"Empty justification from {role}")

    return {
        "name": strategy_name,
        "decision": decision,
        "confidence": conf,
        "justification": justification,
    }


def run_strategies_parallel(
    market_data: dict[str, Any],
    strategy_a_name: str,
    strategy_b_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run A and B in parallel; same input, independent outputs."""

    def _run_a() -> dict[str, Any]:
        return run_strategy_agent(
            role="strategy_a",
            strategy_name=strategy_a_name,
            prompt_filename="strategy_a.txt",
            market_data=market_data,
        )

    def _run_b() -> dict[str, Any]:
        return run_strategy_agent(
            role="strategy_b",
            strategy_name=strategy_b_name,
            prompt_filename="strategy_b.txt",
            market_data=market_data,
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(_run_a)
        fb = ex.submit(_run_b)
        a = fa.result()
        b = fb.result()
    return a, b
