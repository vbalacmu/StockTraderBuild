# StockTrader

Multi-agent stock snapshot experiment: two LLM strategy personas and one evaluator share the same Yahoo Finance–derived metrics and produce comparable **BUY / HOLD / SELL** recommendations with short rationales. The goal is **behavioral contrast** (how different prompts interpret the same numbers), not production trading or return optimization.

## Overview

1. **Market data** — `yfinance` pulls daily history; Python computes indicators (moving averages, RSI, drawdowns, volume tilt, etc.). No LLM in this layer.
2. **Strategies** — Two Ollama agents run **in parallel** on the same JSON snapshot, with separate system prompts (`prompts/strategy_a.txt`, `prompts/strategy_b.txt`). Neither sees the other’s output until both finish.
3. **Evaluator** — A third Ollama call compares the two structured outputs against the same snapshot (`prompts/evaluator.txt`) and writes consensus or disagreement text. The `agents_agree` field matches mechanical equality of the two decisions.
4. **Outputs** — One JSON file per ticker under `outputs/`, plus `summary.json`. Checked-in runs let you inspect results without rerunning the pipeline or keeping Ollama online.

## Strategies

- **Strategy A:** Momentum Trader  
- **Strategy B:** Value Contrarian  

## LLM

**Ollama** (local). Default model: `llama3.2` (override with `OLLAMA_MODEL`). Remote daemon: set `OLLAMA_HOST`.

## Framework

Plain Python in `src/`, including `concurrent.futures` for parallel strategy calls. No LangChain/LangGraph requirement.

## Install and run

```bash
cd stocktrader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull a model, then:

```bash
ollama pull llama3.2
cd src
python main.py
```

Default tickers: `NVDA`, `TSLA`, `JNJ`, `INTC`. Pass other symbols as arguments. Writes JSON under `outputs/`.

## Optional: historical backtest

Point-in-time replay + simple forward-return scoring (bonus extension). Example:

```bash
cd src
python main.py --backtest-only NVDA TSLA JNJ INTC --backtest-samples 8
```

Writes `outputs/backtest.json`. See `main.py --help` for lookback and horizon flags.

## Layout

| Path | Purpose |
|------|---------|
| `src/main.py` | CLI: live run, optional backtest |
| `src/market_data.py` | yfinance + indicators |
| `src/strategies.py` | Parallel Ollama strategy calls |
| `src/evaluator.py` | Ollama evaluator |
| `src/orchestration.py` | Pipeline wiring |
| `src/backtest.py` | Historical backtest |
| `prompts/` | Strategy A, B, and evaluator prompts |
| `outputs/` | Per-ticker JSON, `summary.json`, optional `backtest.json` |
| `report/` | `report.pdf`, `ai_use_appendix.pdf` |
