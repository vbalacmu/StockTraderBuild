"""CLI entry: fetch data, run two Ollama strategy agents + evaluator, save JSON outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest import run_historical_backtest, write_backtest_json
from orchestration import outputs_dir, run_all, write_stock_json, write_summary_json

# Mix: extended leader (NVDA), volatile growth (TSLA), defensive steady (JNJ), laggard / value tape (INTC).
DEFAULT_TICKERS = ["NVDA", "TSLA", "JNJ", "INTC"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="StockTrader two-strategy agent run (Ollama + yfinance).")
    p.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Stock symbols (default: NVDA TSLA JNJ INTC).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: ../outputs from src, i.e. stocktrader/outputs).",
    )
    p.add_argument(
        "--backtest",
        action="store_true",
        help="After the live run, also write outputs/backtest.json (bonus: historical backtest).",
    )
    p.add_argument(
        "--backtest-only",
        action="store_true",
        help="Only run the historical backtest (skip live per-ticker JSON run).",
    )
    p.add_argument("--backtest-samples", type=int, default=8, help="As-of dates per ticker (default: 8).")
    p.add_argument(
        "--backtest-forward",
        type=int,
        default=5,
        help="Forward horizon in trading days for scoring (default: 5).",
    )
    p.add_argument(
        "--backtest-lookback",
        type=int,
        default=400,
        help="Calendar days of history to download per ticker (default: 400).",
    )
    args = p.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if not tickers:
        print("No tickers provided.", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else outputs_dir()

    if args.backtest_only:
        try:
            bt = run_historical_backtest(
                tickers,
                lookback_days=args.backtest_lookback,
                n_samples=args.backtest_samples,
                forward_trading_days=args.backtest_forward,
            )
        except Exception as e:
            print(f"Backtest failed: {e}", file=sys.stderr)
            return 1
        path = write_backtest_json(bt, out_dir)
        print(f"Wrote {path}")
        return 0

    try:
        records = run_all(tickers)
    except Exception as e:
        print(f"Run failed: {e}", file=sys.stderr)
        return 1

    for r in records:
        path = write_stock_json(r, out_dir)
        print(f"Wrote {path}")

    sp = write_summary_json(records, out_dir)
    print(f"Wrote {sp}")

    if args.backtest:
        try:
            bt = run_historical_backtest(
                tickers,
                lookback_days=args.backtest_lookback,
                n_samples=args.backtest_samples,
                forward_trading_days=args.backtest_forward,
            )
        except Exception as e:
            print(f"Backtest failed: {e}", file=sys.stderr)
            return 1
        bp = write_backtest_json(bt, out_dir)
        print(f"Wrote {bp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
