"""Technical indicators. Each takes a price DataFrame and returns a Series aligned to it."""
import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    return df[column].rolling(period, min_periods=period).mean()


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    return df[column].ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Wilder's RSI."""
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def highest(df: pd.DataFrame, period: int, column: str = "high") -> pd.Series:
    """Highest value of the previous `period` bars (excludes the current bar, so breakouts work)."""
    return df[column].shift(1).rolling(period, min_periods=period).max()


def lowest(df: pd.DataFrame, period: int, column: str = "low") -> pd.Series:
    """Lowest value of the previous `period` bars (excludes the current bar)."""
    return df[column].shift(1).rolling(period, min_periods=period).min()


REGISTRY = {"sma": sma, "ema": ema, "rsi": rsi, "atr": atr, "highest": highest, "lowest": lowest}


def add_indicators(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """Add one column per entry in `specs`, e.g. {"sma_50": {"type": "sma", "period": 50}}."""
    df = df.copy()
    for name, spec in specs.items():
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in REGISTRY:
            raise ValueError(f"Unknown indicator type '{kind}' for '{name}'. Known: {sorted(REGISTRY)}")
        df[name] = REGISTRY[kind](df, **spec)
    return df
