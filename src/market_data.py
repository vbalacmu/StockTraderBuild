"""Fetch Yahoo Finance data via yfinance and compute indicators for strategy agents."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class MarketDataResult:
    ticker: str
    market_data_summary: dict[str, Any]
    """Flat dict suitable for JSON `market_data_summary` and for LLM context."""


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss) or last_loss == 0:
        return None
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _volume_trend(volume: pd.Series, short: int = 5, long: int = 20) -> float | None:
    if len(volume) < long:
        return None
    short_ma = volume.iloc[-short:].mean()
    long_ma = volume.iloc[-long:].mean()
    if long_ma == 0:
        return None
    return float(short_ma / long_ma - 1.0)


def _market_summary_from_hist(t: str, hist: pd.DataFrame) -> dict[str, Any]:
    """Build the same summary dict as live fetch, from an OHLCV frame (no network)."""
    if hist is None or hist.empty or len(hist) < 60:
        raise ValueError(f"Insufficient rows for {t!r}: {0 if hist is None else len(hist)}")

    close = hist["Close"]
    volume = hist["Volume"]

    current_price = float(close.iloc[-1])
    price_30d_ago = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
    pct_change_30d = float((current_price / price_30d_ago - 1.0) * 100)

    daily_returns = close.pct_change().dropna()
    vol_30 = daily_returns.iloc[-30:] if len(daily_returns) >= 30 else daily_returns
    volatility_30d = float(vol_30.std()) if len(vol_30) > 1 else 0.0

    ma20 = float(close.rolling(20, min_periods=20).mean().iloc[-1])
    ma50 = float(close.rolling(50, min_periods=50).mean().iloc[-1])

    high_52w = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    low_52w = float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min())
    peak_90d = float(close.iloc[-90:].max())
    drawdown_from_90d_peak = float((current_price / peak_90d - 1.0) * 100)
    pct_from_52w_high = float((current_price / high_52w - 1.0) * 100)
    pct_from_52w_low = float((current_price / low_52w - 1.0) * 100)

    rsi_14 = _rsi(close, 14)
    v_trend = _volume_trend(volume)

    last_30_rets = daily_returns.iloc[-30:].tolist() if len(daily_returns) >= 30 else daily_returns.tolist()
    last_30_rets = [float(x) for x in last_30_rets]

    return {
        "ticker": t,
        "current_price": round(current_price, 4),
        "price_30d_ago": round(price_30d_ago, 4),
        "pct_change_30d": round(pct_change_30d, 3),
        "avg_daily_volume": int(round(volume.iloc[-20:].mean())),
        "volatility_30d": round(volatility_30d, 6),
        "moving_avg_20d": round(ma20, 4),
        "moving_avg_50d": round(ma50, 4),
        "high_52w": round(high_52w, 4),
        "low_52w": round(low_52w, 4),
        "pct_from_52w_high": round(pct_from_52w_high, 3),
        "pct_from_52w_low": round(pct_from_52w_low, 3),
        "drawdown_from_90d_peak_pct": round(drawdown_from_90d_peak, 3),
        "rsi_14": None if rsi_14 is None else round(rsi_14, 2),
        "volume_trend_short_vs_long": None if v_trend is None else round(v_trend, 4),
        "last_30_daily_returns": [round(r, 6) for r in last_30_rets],
    }


def fetch_market_data(ticker: str, history_days: int = 120) -> MarketDataResult:
    """
    Pull at least 90 trading days of history; default buffer for MA50 + RSI warmup.
    No LLM calls.
    """
    t = ticker.strip().upper()
    stock = yf.Ticker(t)
    hist = stock.history(period=f"{history_days}d", auto_adjust=True)

    if hist is None or hist.empty or len(hist) < 60:
        raise ValueError(f"Insufficient history for {t!r} (got {0 if hist is None else len(hist)} rows).")

    summary = _market_summary_from_hist(t, hist)
    return MarketDataResult(ticker=t, market_data_summary=summary)


def fetch_full_history(
    ticker: str,
    *,
    start: date | datetime | str,
    end: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Download daily history through `end` (inclusive of last bar); yfinance `end` is exclusive."""
    t = ticker.strip().upper()
    stock = yf.Ticker(t)
    if end is None:
        end_exclusive = date.today() + timedelta(days=1)
    else:
        end_inclusive = pd.Timestamp(end).date()
        end_exclusive = end_inclusive + timedelta(days=1)
    hist = stock.history(start=start, end=end_exclusive, auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"No history for {t!r}")
    return hist


def forward_close_return(
    hist: pd.DataFrame,
    asof_last_ts: pd.Timestamp,
    forward_trading_days: int,
) -> float | None:
    """Close-to-close return N trading sessions after asof close. None if not enough future bars."""
    if forward_trading_days < 1:
        raise ValueError("forward_trading_days must be >= 1")
    try:
        pos = hist.index.get_loc(asof_last_ts)
    except KeyError:
        idx = hist.index.get_indexer([asof_last_ts], method="pad")
        if idx.size == 0 or idx[0] == -1:
            return None
        pos = int(idx[0])
    if isinstance(pos, slice):
        pos = pos.stop - 1
    elif hasattr(pos, "__len__") and not isinstance(pos, str):
        pos = int(pos[-1])
    pos = int(pos)
    j = pos + forward_trading_days
    if j >= len(hist):
        return None
    c0 = float(hist["Close"].iloc[pos])
    c1 = float(hist["Close"].iloc[j])
    if c0 == 0:
        return None
    return (c1 / c0) - 1.0


def market_data_summary_for_json(result: MarketDataResult) -> dict[str, Any]:
    """Strip ticker if you want it only under top-level ticker; keep full summary for assignment."""
    d = dict(result.market_data_summary)
    return d
