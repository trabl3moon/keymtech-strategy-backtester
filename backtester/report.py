"""Write trades.csv, equity.csv, metrics.json and a self-contained report.html (no external assets)."""
import html
import json
from pathlib import Path

import pandas as pd

from .metrics import drawdown

LABELS = {
    "initial_capital": "Initial capital", "final_equity": "Final equity", "total_return_pct": "Total return %",
    "cagr_pct": "CAGR %", "max_drawdown_pct": "Max drawdown %", "sharpe": "Sharpe", "sortino": "Sortino",
    "trades": "Trades", "win_rate_pct": "Win rate %", "profit_factor": "Profit factor",
    "avg_win": "Avg win", "avg_loss": "Avg loss", "exposure_pct": "Time in market %",
}


def trades_frame(trades: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "entry_date": t.entry_date.date(), "entry_price": round(t.entry_price, 4), "qty": t.qty,
        "exit_date": t.exit_date.date(), "exit_price": round(t.exit_price, 4), "exit_reason": t.exit_reason,
        "fees": round(t.entry_fee + t.exit_fee, 2), "pnl": round(t.pnl, 2), "return_pct": round(t.return_pct, 2),
    } for t in trades])


def _svg_line(series: pd.Series, width: int, height: int, color: str, fill: bool = False, fill_to_top: bool = False) -> str:
    vals = series.to_numpy(dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    n = len(vals) - 1 or 1
    pts = [f"{i / n * width:.1f},{height - (v - lo) / span * height:.1f}" for i, v in enumerate(vals)]
    path = "M" + " L".join(pts)
    base = 0 if fill_to_top else height
    area = f'<path d="{path} L{width},{base} L0,{base} Z" fill="{color}" opacity=".15"/>' if fill else ""
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="chart">{area}'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>'
            f'<div class="axis"><span>{series.index[0].date()}</span><span>min {lo:,.2f} · max {hi:,.2f}</span>'
            f'<span>{series.index[-1].date()}</span></div>')


def write(out_dir: str | Path, name: str, equity: pd.Series, trades: list, metrics: dict, config: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tdf = trades_frame(trades)
    tdf.to_csv(out / "trades.csv", index=False)
    pd.DataFrame({"equity": equity.round(2), "drawdown_pct": drawdown(equity).round(2)}).to_csv(out / "equity.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cards = "".join(f'<div class="card"><div class="k">{LABELS.get(k, k)}</div><div class="v">{v:,}</div></div>'
                    for k, v in metrics.items())
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for r in tdf.itertuples(index=False)) if len(tdf) else '<tr><td colspan="9">No trades</td></tr>'
    head = "".join(f"<th>{c}</th>" for c in tdf.columns) if len(tdf) else ""
    cfg = html.escape(json.dumps(config, indent=2, default=str))

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(name)} backtest</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f8fb;color:#1b2430}}
main{{max-width:1100px;margin:auto;padding:24px 16px}} h1{{margin:0 0 4px}} .sub{{color:#5b6b7f;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.card{{background:#fff;border:1px solid #e1e7ef;border-radius:8px;padding:10px 12px}}
.k{{font-size:12px;color:#5b6b7f}} .v{{font-size:20px;font-weight:600}}
section{{background:#fff;border:1px solid #e1e7ef;border-radius:8px;padding:14px;margin-top:16px}}
.chart{{width:100%;height:220px;display:block}} .axis{{display:flex;justify-content:space-between;font-size:12px;color:#5b6b7f}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:6px 8px;border-bottom:1px solid #eef2f7;text-align:right}}
th{{background:#f1f5fa;position:sticky;top:0}} .scroll{{max-height:420px;overflow:auto}} pre{{font-size:12px;overflow:auto}}
</style></head><body><main>
<h1>{html.escape(name)}</h1><div class="sub">Backtest report · {equity.index[0].date()} → {equity.index[-1].date()} · {len(equity)} bars</div>
<div class="grid">{cards}</div>
<section><h3>Equity curve</h3>{_svg_line(equity, 1000, 220, "#1f6feb", fill=True)}</section>
<section><h3>Drawdown %</h3>{_svg_line(drawdown(equity), 1000, 140, "#d1242f", fill=True, fill_to_top=True)}</section>
<section><h3>Trade log ({len(tdf)})</h3><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h3>Configuration used</h3><pre>{cfg}</pre></section>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
