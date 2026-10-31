"""Command line entry point.

    python -m backtester --config configs/sma_crossover.toml
    python -m backtester --config configs/sma_crossover.toml --data data/MYSTOCK.csv --out reports/mystock
"""
import argparse
import sys
import tomllib
from pathlib import Path

from . import metrics, report
from .data import load_csv, synthetic
from .engine import Backtester


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="backtester", description="Config-driven stock / ETF strategy backtester")
    ap.add_argument("--config", required=True, help="TOML strategy file")
    ap.add_argument("--data", nargs="*", help="One or more OHLCV CSV files (overrides [data] in the config)")
    ap.add_argument("--out", help="Output folder (default: reports/<strategy name>)")
    args = ap.parse_args(argv)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    name = cfg.get("name", Path(args.config).stem)
    data_cfg = cfg.get("data", {})
    files = args.data or data_cfg.get("files", [])
    out_root = Path(args.out or f"reports/{name}")

    datasets = {Path(p).stem: load_csv(p, data_cfg.get("date_column", "date")) for p in files}
    if not datasets:
        print("No data files given; using synthetic demo data (seed 7).")
        datasets = {"SYNTHETIC": synthetic()}

    for symbol, prices in datasets.items():
        bt = Backtester(cfg)
        res = bt.run(prices)
        m = metrics.compute(res.equity, res.trades, bt.initial_capital,
                            cfg.get("report", {}).get("periods_per_year", 252),
                            cfg.get("report", {}).get("risk_free_rate", 0.0))
        out = out_root / symbol if len(datasets) > 1 else out_root
        path = report.write(out, f"{name} · {symbol}", res.equity, res.trades, m, cfg)
        print(f"{symbol}: return {m['total_return_pct']}% | max DD {m['max_drawdown_pct']}% | "
              f"Sharpe {m['sharpe']} | trades {m['trades']} | win {m['win_rate_pct']}%  ->  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
