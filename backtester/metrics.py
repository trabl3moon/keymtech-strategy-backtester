"""Performance metrics computed from the equity curve and the trade list."""
import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown in percent from the running peak (0 or negative)."""
    return (equity / equity.cummax() - 1) * 100


def compute(equity: pd.Series, trades: list, initial_capital: float, periods_per_year: int = 252,
            risk_free_rate: float = 0.0) -> dict:
    final = float(equity.iloc[-1])
    total_return = (final / initial_capital - 1) * 100
    years = len(equity) / periods_per_year
    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else float("nan")

    rets = equity.pct_change().dropna()
    excess = rets - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    sharpe = float(excess.mean() / std * np.sqrt(periods_per_year)) if std and std > 0 else float("nan")
    downside = excess[excess < 0].std(ddof=1)
    sortino = float(excess.mean() / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else float("nan")

    pnls = np.array([t.pnl for t in trades]) if trades else np.array([])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    gross_loss = -losses.sum()
    exposure = sum((equity.index.get_loc(t.exit_date) - equity.index.get_loc(t.entry_date) + 1) for t in trades)

    return {
        "initial_capital": initial_capital,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(float(drawdown(equity).min()), 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 2) if len(pnls) else 0.0,
        "profit_factor": round(wins.sum() / gross_loss, 2) if gross_loss > 0 else float("inf") if len(wins) else 0.0,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "exposure_pct": round(exposure / len(equity) * 100, 2),
    }
