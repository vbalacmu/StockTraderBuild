"""Bonus: historical backtest — same LLM strategies on point-in-time data vs forward returns."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from market_data import _market_summary_from_hist, fetch_full_history, forward_close_return
from orchestration import STRATEGY_A_NAME, STRATEGY_B_NAME
from strategies import run_strategies_parallel


def _score_vs_forward(decision: str, forward_ret: float, *, hold_band: float) -> bool:
    d = decision.upper().strip()
    if d == "BUY":
        return forward_ret > 0.0
    if d == "SELL":
        return forward_ret < 0.0
    if d == "HOLD":
        return abs(forward_ret) <= hold_band
    return False


def _sample_positions(first_pos: int, last_pos: int, n_samples: int) -> list[int]:
    if last_pos <= first_pos or n_samples < 1:
        return []
    if n_samples == 1:
        return [first_pos]
    span = last_pos - first_pos
    raw = [first_pos + int(round(span * i / (n_samples - 1))) for i in range(n_samples)]
    out: list[int] = []
    for x in sorted(set(raw)):
        if first_pos <= x <= last_pos:
            out.append(x)
    return out


def run_historical_backtest(
    tickers: list[str],
    *,
    lookback_days: int = 400,
    n_samples: int = 8,
    forward_trading_days: int = 5,
    hold_band: float = 0.005,
) -> dict[str, Any]:
    """
    For each ticker, download a long daily series, then at evenly spaced past dates:
    rebuild indicators using only history through that date (no lookahead), run both
    Ollama strategies, and score decisions against realized close-to-close return over
    the next `forward_trading_days` sessions.
    """
    today = date.today()
    start = today - timedelta(days=lookback_days)
    rows: list[dict[str, Any]] = []
    a_hits = a_total = b_hits = b_total = 0

    for raw in tickers:
        t = raw.strip().upper()
        hist = fetch_full_history(t, start=start)
        hist = hist.sort_index()
        first_pos = 60
        last_pos = len(hist) - 1 - forward_trading_days
        positions = _sample_positions(first_pos, last_pos, n_samples)

        for pos in positions:
            sub = hist.iloc[: pos + 1]
            asof_ts = hist.index[pos]
            asof_date = asof_ts.date().isoformat() if hasattr(asof_ts, "date") else str(asof_ts)[:10]
            try:
                summary = _market_summary_from_hist(t, sub)
            except ValueError:
                continue

            fwd = forward_close_return(hist, asof_ts, forward_trading_days)
            if fwd is None:
                continue

            try:
                strat_a, strat_b = run_strategies_parallel(
                    summary,
                    STRATEGY_A_NAME,
                    STRATEGY_B_NAME,
                )
            except Exception as e:
                rows.append(
                    {
                        "ticker": t,
                        "asof_date": asof_date,
                        "error": str(e),
                    }
                )
                continue

            da, db = strat_a["decision"], strat_b["decision"]
            ca = _score_vs_forward(da, fwd, hold_band=hold_band)
            cb = _score_vs_forward(db, fwd, hold_band=hold_band)
            a_total += 1
            b_total += 1
            if ca:
                a_hits += 1
            if cb:
                b_hits += 1

            rows.append(
                {
                    "ticker": t,
                    "asof_date": asof_date,
                    "forward_trading_days": forward_trading_days,
                    "forward_close_return": round(fwd, 6),
                    "strategy_a_decision": da,
                    "strategy_b_decision": db,
                    "strategy_a_correct": ca,
                    "strategy_b_correct": cb,
                }
            )

    winner = "tie"
    if a_total and b_total:
        ar = a_hits / a_total
        br = b_hits / b_total
        if ar > br:
            winner = STRATEGY_A_NAME
        elif br > ar:
            winner = STRATEGY_B_NAME

    return {
        "bonus": "historical_backtest",
        "methodology": (
            "Point-in-time yfinance bars only through each asof date; same two Ollama agents as live run. "
            "Accuracy: BUY iff forward close-to-close return > 0; SELL iff < 0; HOLD iff |return| <= hold_band."
        ),
        "period": {
            "lookback_calendar_days": lookback_days,
            "samples_per_ticker_target": n_samples,
            "forward_horizon_trading_days": forward_trading_days,
            "hold_band_abs_return": hold_band,
            "run_generated": date.today().isoformat(),
        },
        "strategies": [STRATEGY_A_NAME, STRATEGY_B_NAME],
        "tickers": [x.strip().upper() for x in tickers],
        "strategy_a_hits": a_hits,
        "strategy_a_total": a_total,
        "strategy_a_hit_rate": round(a_hits / a_total, 4) if a_total else None,
        "strategy_b_hits": b_hits,
        "strategy_b_total": b_total,
        "strategy_b_hit_rate": round(b_hits / b_total, 4) if b_total else None,
        "more_accurate_overall": winner,
        "rows": rows,
    }


def write_backtest_json(payload: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "backtest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
