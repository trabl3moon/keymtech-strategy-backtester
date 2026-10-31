"""Hand-checked cases. Every expected number below can be worked out on paper from the bars."""
import math

import pandas as pd
import pytest

from backtester import Backtester, metrics, synthetic
from backtester.indicators import rsi, sma


def bars(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows), name="date")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=0.0)


def cfg(**over):
    base = {
        "account": {"initial_capital": 10_000, "commission_pct": 0, "commission_fixed": 0, "slippage_pct": 0},
        "position_sizing": {"method": "fixed_qty", "qty": 10},
        "entry": {"conditions": [{"left": "close", "op": ">", "right": 100}]},
        "exit": {"conditions": [{"left": "close", "op": "<", "right": 100}]},
    }
    base.update(over)
    return base


def test_fills_at_next_open_no_lookahead():
    # Signal on bar 1 close (101 > 100) -> buy bar 2 open (102). Exit signal bar 3 close (99) -> sell bar 4 open (98).
    df = bars([(100, 100, 100, 100), (100, 101, 100, 101), (102, 103, 101, 102), (101, 101, 99, 99), (98, 98, 97, 97)])
    res = Backtester(cfg()).run(df)
    (t,) = res.trades
    assert (t.entry_price, t.exit_price, t.exit_reason) == (102, 98, "exit_signal")
    assert t.pnl == pytest.approx(-40)
    assert res.equity.iloc[-1] == pytest.approx(9960)


def test_stop_gap_fills_at_open_not_stop():
    # Buy at 102 with 5% stop = 96.9. Bar 3 opens at 90 (gap below stop) -> fill at 90, not 96.9.
    df = bars([(100, 100, 100, 100), (100, 101, 100, 101), (102, 103, 101, 102), (90, 91, 89, 90)])
    c = cfg(exit={}, exits={"stop_loss_pct": 5})
    (t,) = Backtester(c).run(df).trades
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == 90


def test_take_profit_and_fees_and_slippage():
    # 1% slippage: entry 102*1.01 = 103.02. Target 10% -> 113.322. Bar hits high 120 -> fill 113.322*0.99.
    df = bars([(100, 100, 100, 100), (100, 101, 100, 101), (102, 103, 101, 102), (110, 120, 109, 115)])
    c = cfg(exit={}, exits={"take_profit_pct": 10})
    c["account"].update(slippage_pct=1, commission_fixed=5)
    (t,) = Backtester(c).run(df).trades
    assert t.entry_price == pytest.approx(103.02)
    assert t.exit_price == pytest.approx(113.322 * 0.99)
    assert t.pnl == pytest.approx((113.322 * 0.99 - 103.02) * 10 - 10)


def test_open_position_closed_at_end_of_data():
    df = bars([(100, 100, 100, 100), (100, 101, 100, 101), (102, 103, 101, 105)])
    (t,) = Backtester(cfg(exit={})).run(df).trades
    assert (t.exit_reason, t.exit_price) == ("end_of_data", 105)


def test_risk_pct_sizing():
    # Equity 10,000, risk 2% = 200. Entry 102, stop 5% -> 96.9, risk/share 5.1 -> floor(39.2) = 39 shares.
    df = bars([(100, 100, 100, 100), (100, 101, 100, 101), (102, 103, 101, 102)])
    c = cfg(exit={}, exits={"stop_loss_pct": 5}, position_sizing={"method": "risk_pct", "risk_pct": 2})
    (t,) = Backtester(c).run(df).trades
    assert t.qty == 39


def test_never_spends_more_than_cash():
    df = bars([(100, 100, 100, 100), (100, 101, 100, 101), (102, 103, 101, 102)])
    c = cfg(exit={}, position_sizing={"method": "fixed_qty", "qty": 1000})
    c["account"]["initial_capital"] = 1000
    (t,) = Backtester(c).run(df).trades
    assert t.qty == 9  # floor(1000 / 102), not the 1000 asked for


def test_crosses_above_fires_once():
    df = bars([(c, c, c, c) for c in [10, 10, 10, 12, 13, 14, 9, 15]])
    c = cfg(indicators={"s3": {"type": "sma", "period": 3}},
            entry={"conditions": [{"left": "close", "op": "crosses_above", "right": "s3"}]}, exit={})
    res = Backtester(c).run(df)
    assert len(res.trades) == 1  # stays in after the first cross; no pyramiding


def test_metrics_on_known_curve():
    eq = pd.Series([100, 110, 99, 120], index=pd.bdate_range("2024-01-01", periods=4))
    m = metrics.compute(eq, [], 100)
    assert m["total_return_pct"] == 20
    assert m["max_drawdown_pct"] == pytest.approx(-10)


def test_indicators_match_reference():
    df = bars([(c, c, c, c) for c in range(1, 21)])
    assert sma(df, 5).iloc[-1] == pytest.approx(18)
    assert rsi(df, 14).iloc[-1] == 100  # only gains
    assert math.isnan(sma(df, 5).iloc[3])


def test_synthetic_is_deterministic_and_valid():
    a, b = synthetic(300, seed=3), synthetic(300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
