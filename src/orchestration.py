"""End-to-end pipeline: market data -> parallel strategies -> evaluator -> JSON records."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from evaluator import decisions_agree, run_evaluator
from market_data import fetch_market_data, market_data_summary_for_json
from strategies import run_strategies_parallel

STRATEGY_A_NAME = "Momentum Trader"
STRATEGY_B_NAME = "Value Contrarian"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def outputs_dir() -> Path:
    return _project_root() / "outputs"


def run_stock(ticker: str, *, run_date: str | None = None) -> dict[str, Any]:
    run_date = run_date or date.today().isoformat()
    md = fetch_market_data(ticker)
    summary = market_data_summary_for_json(md)

    strategy_a, strategy_b = run_strategies_parallel(
        summary,
        STRATEGY_A_NAME,
        STRATEGY_B_NAME,
    )
    evaluator = run_evaluator(
        market_data=summary,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
    )

    # Optional: enforce agree flag matches decisions
    evaluator = {
        "agents_agree": decisions_agree(strategy_a, strategy_b),
        "analysis": evaluator["analysis"],
    }

    out: dict[str, Any] = {
        "ticker": md.ticker,
        "run_date": run_date,
        "market_data_summary": {k: v for k, v in summary.items() if k != "ticker"},
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "evaluator": evaluator,
    }
    return out


def write_stock_json(record: dict[str, Any], directory: Path | None = None) -> Path:
    directory = directory or outputs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    ticker = record["ticker"]
    path = directory / f"{ticker}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def write_summary_json(records: list[dict[str, Any]], directory: Path | None = None) -> Path:
    directory = directory or outputs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    agreements = 0
    disagreements = 0
    for r in records:
        a = r["strategy_a"]["decision"]
        b = r["strategy_b"]["decision"]
        agree = a == b
        if agree:
            agreements += 1
        else:
            disagreements += 1
        results.append(
            {
                "ticker": r["ticker"],
                "a_decision": a,
                "b_decision": b,
                "agree": agree,
            }
        )
    summary = {
        "strategies": [STRATEGY_A_NAME, STRATEGY_B_NAME],
        "stocks_analyzed": [r["ticker"] for r in records],
        "total_agreements": agreements,
        "total_disagreements": disagreements,
        "results": results,
    }
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def run_all(tickers: list[str], *, run_date: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for t in tickers:
        records.append(run_stock(t.strip(), run_date=run_date))
    return records
