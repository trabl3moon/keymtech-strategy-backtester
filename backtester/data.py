"""Load OHLCV price history from CSV, and generate synthetic data for demos and tests."""
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


def load_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Read a CSV with a date column and open/high/low/close(/volume). Column names are case-insensitive."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if date_column not in df.columns:
        raise ValueError(f"{path}: no '{date_column}' column. Found: {list(df.columns)}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = df[REQUIRED].isna().any(axis=1)
    if bad.any():
        df = df[~bad]
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[REQUIRED + ["volume"]].astype(float)


def synthetic(days: int = 1500, start: str = "2019-01-01", seed: int = 7, start_price: float = 100.0,
              drift: float = 0.08, vol: float = 0.22) -> pd.DataFrame:
    """Geometric Brownian motion daily bars with realistic open/high/low. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    rets = rng.normal((drift - vol**2 / 2) * dt, vol * np.sqrt(dt), days)
    close = start_price * np.exp(np.cumsum(rets))
    prev = np.concatenate([[start_price], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, vol * np.sqrt(dt) * 0.3, days))
    spread = np.abs(rng.normal(0, vol * np.sqrt(dt) * 0.6, days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(200_000, 2_000_000, days)
    idx = pd.bdate_range(start, periods=days, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx).round(4)
