"""Evaluator: compares two strategy outputs; uses Ollama for narrative analysis."""

from __future__ import annotations

import json
from typing import Any

import ollama

from strategies import (
    _extract_json_object,
    _load_prompt,
    _ollama_host,
    _ollama_model,
    _ollama_options_evaluator,
)


def decisions_agree(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return str(a.get("decision", "")).upper() == str(b.get("decision", "")).upper()


def run_evaluator(
    *,
    market_data: dict[str, Any],
    strategy_a: dict[str, Any],
    strategy_b: dict[str, Any],
) -> dict[str, Any]:
    system = _load_prompt("evaluator.txt")
    agree = decisions_agree(strategy_a, strategy_b)
    payload = {
        "market_data": market_data,
        "strategy_a": {
            "name": strategy_a["name"],
            "decision": strategy_a["decision"],
            "confidence": strategy_a["confidence"],
            "justification": strategy_a["justification"],
        },
        "strategy_b": {
            "name": strategy_b["name"],
            "decision": strategy_b["decision"],
            "confidence": strategy_b["confidence"],
            "justification": strategy_b["justification"],
        },
        "decisions_match": agree,
    }
    user = json.dumps(payload, indent=2)

    kwargs: dict[str, Any] = {
        "model": _ollama_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": _ollama_options_evaluator(),
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
    agents_agree = bool(parsed.get("agents_agree", agree))
    # Keep flag consistent with actual decisions if model drifted
    if agents_agree != agree:
        agents_agree = agree
    analysis = str(parsed.get("analysis", "")).strip()
    if not analysis:
        raise ValueError("Evaluator returned empty analysis.")

    return {"agents_agree": agents_agree, "analysis": analysis}
