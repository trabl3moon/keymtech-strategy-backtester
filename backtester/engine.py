"""Bar-by-bar long-only backtest engine.

Timing rules (no look-ahead):
  * Entry and exit signals are evaluated on a bar's close.
  * Orders fill at the NEXT bar's open, plus slippage.
  * Stop-loss / take-profit are checked against each later bar's high/low.
    If the bar opens beyond the stop, the fill is the open (gap), not the stop price.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .rules import evaluate_group


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    qty: float
    stop: float | None
    target: float | None
    entry_fee: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_fee: float = 0.0
    exit_reason: str = ""

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty - self.entry_fee - self.exit_fee

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.qty + self.entry_fee
        return self.pnl / cost * 100 if cost else 0.0


@dataclass
class Result:
    equity: pd.Series
    trades: list = field(default_factory=list)
    data: pd.DataFrame | None = None


class Backtester:
    def __init__(self, config: dict):
        self.cfg = config
        acct = config.get("account", {})
        self.initial_capital = float(acct.get("initial_capital", 100_000))
        self.commission_pct = float(acct.get("commission_pct", 0.0)) / 100
        self.commission_fixed = float(acct.get("commission_fixed", 0.0))
        self.slippage_pct = float(acct.get("slippage_pct", 0.0)) / 100
        self.fractional = bool(acct.get("fractional_shares", False))
        self.sizing = config.get("position_sizing", {"method": "fixed_fraction", "fraction": 1.0})
        self.exits = config.get("exits", {})

    # --- helpers -----------------------------------------------------------
    def _fee(self, notional: float) -> float:
        return notional * self.commission_pct + self.commission_fixed

    def _round_qty(self, qty: float) -> float:
        return max(qty, 0.0) if self.fractional else float(np.floor(max(qty, 0.0)))

    def _stop_and_target(self, price: float, atr_value: float | None):
        stop = target = None
        if "stop_loss_pct" in self.exits:
            stop = price * (1 - self.exits["stop_loss_pct"] / 100)
        if "stop_loss_atr" in self.exits and atr_value and not np.isnan(atr_value):
            stop = price - self.exits["stop_loss_atr"] * atr_value
        if "take_profit_pct" in self.exits:
            target = price * (1 + self.exits["take_profit_pct"] / 100)
        if "take_profit_r" in self.exits and stop is not None:
            target = price + self.exits["take_profit_r"] * (price - stop)
        return stop, target

    def _size(self, cash: float, equity: float, price: float, stop: float | None) -> float:
        method = self.sizing.get("method", "fixed_fraction")
        if method == "fixed_qty":
            qty = float(self.sizing["qty"])
        elif method == "fixed_fraction":
            qty = equity * float(self.sizing.get("fraction", 1.0)) / price
        elif method == "risk_pct":
            if stop is None or stop >= price:
                raise ValueError("position_sizing.method = 'risk_pct' needs a stop-loss in [exits]")
            risk_cash = equity * float(self.sizing["risk_pct"]) / 100
            qty = risk_cash / (price - stop)
        else:
            raise ValueError(f"Unknown position sizing method '{method}'")
        # Never spend more cash than we have, including the fee.
        max_affordable = (cash - self.commission_fixed) / (price * (1 + self.commission_pct))
        return self._round_qty(min(qty, max_affordable))

    # --- main loop ---------------------------------------------------------
    def run(self, prices: pd.DataFrame) -> Result:
        df = add_indicators(prices, self.cfg.get("indicators", {}))
        entry_sig = evaluate_group(df, self.cfg.get("entry"))
        exit_sig = evaluate_group(df, self.cfg.get("exit"))
        atr_col = self.exits.get("atr_column")
        max_bars = self.exits.get("max_bars_in_trade")

        cash = self.initial_capital
        open_trade: Trade | None = None
        bars_held = 0
        pending_entry = pending_exit = False
        trades, equity = [], []

        for i, (date, bar) in enumerate(df.iterrows()):
            # 1) Fill orders queued on the previous bar's close, at this bar's open.
            if pending_exit and open_trade:
                cash += self._close(open_trade, date, bar["open"] * (1 - self.slippage_pct), "exit_signal")
                trades.append(open_trade)
                open_trade, pending_exit = None, False
            if pending_entry and open_trade is None:
                fill = bar["open"] * (1 + self.slippage_pct)
                atr_value = df[atr_col].iloc[i - 1] if atr_col else None
                stop, target = self._stop_and_target(fill, atr_value)
                qty = self._size(cash, cash, fill, stop)
                if qty > 0:
                    fee = self._fee(fill * qty)
                    cash -= fill * qty + fee
                    open_trade = Trade(date, fill, qty, stop, target, fee)
                    bars_held = 0
            pending_entry = False

            # 2) Intrabar stop / target checks for an open position.
            if open_trade:
                bars_held += 1
                hit = self._intrabar_exit(open_trade, bar)
                if hit:
                    price, reason = hit
                    cash += self._close(open_trade, date, price, reason)
                    trades.append(open_trade)
                    open_trade = None

            # 3) Signals on this bar's close queue orders for the next bar.
            if open_trade:
                if exit_sig.iloc[i] or (max_bars and bars_held >= max_bars):
                    pending_exit = True
            elif entry_sig.iloc[i]:
                pending_entry = True

            equity.append(cash + (open_trade.qty * bar["close"] if open_trade else 0.0))

        # Close anything still open on the final close so P&L is realised.
        if open_trade:
            last_date, last_bar = df.index[-1], df.iloc[-1]
            cash += self._close(open_trade, last_date, last_bar["close"], "end_of_data")
            trades.append(open_trade)
            equity[-1] = cash

        return Result(equity=pd.Series(equity, index=df.index, name="equity"), trades=trades, data=df)

    def _intrabar_exit(self, t: Trade, bar):
        # Check the stop first: when both levels fall inside one bar we assume the worse outcome.
        if t.stop is not None and bar["low"] <= t.stop:
            price = min(bar["open"], t.stop) * (1 - self.slippage_pct)
            return price, "stop_loss"
        if t.target is not None and bar["high"] >= t.target:
            return max(bar["open"], t.target) * (1 - self.slippage_pct), "take_profit"
        return None

    def _close(self, t: Trade, date, price: float, reason: str) -> float:
        """Fill the exit and return the cash it brings back."""
        t.exit_date, t.exit_price, t.exit_reason = date, price, reason
        t.exit_fee = self._fee(price * t.qty)
        return price * t.qty - t.exit_fee
