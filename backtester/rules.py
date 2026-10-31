"""Turn rule conditions from the config file into boolean Series.

A condition is a table like:
    { left = "close", op = "crosses_above", right = "sma_50" }
`left` and `right` can be any column name (price or indicator) or a number.
A rule group is a list of conditions combined with "all" (AND) or "any" (OR).
"""
import pandas as pd

OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "crosses_above": lambda a, b: (a > b) & (a.shift(1) <= _shift(b)),
    "crosses_below": lambda a, b: (a < b) & (a.shift(1) >= _shift(b)),
}


def _shift(x):
    return x.shift(1) if isinstance(x, pd.Series) else x


def _operand(df: pd.DataFrame, value):
    if isinstance(value, (int, float)):
        return value
    if value not in df.columns:
        raise ValueError(f"Rule refers to unknown column '{value}'. Available: {list(df.columns)}")
    return df[value]


def evaluate_condition(df: pd.DataFrame, cond: dict) -> pd.Series:
    op = cond["op"]
    if op not in OPS:
        raise ValueError(f"Unknown operator '{op}'. Known: {sorted(OPS)}")
    left = _operand(df, cond["left"])
    if not isinstance(left, pd.Series):
        left = pd.Series(left, index=df.index)
    result = OPS[op](left, _operand(df, cond["right"]))
    return result.fillna(False).astype(bool)


def evaluate_group(df: pd.DataFrame, group: dict | None) -> pd.Series:
    """`group` = {"mode": "all"|"any", "conditions": [...]}. An empty group never fires."""
    if not group or not group.get("conditions"):
        return pd.Series(False, index=df.index)
    signals = [evaluate_condition(df, c) for c in group["conditions"]]
    combined = signals[0]
    for s in signals[1:]:
        combined = (combined & s) if group.get("mode", "all") == "all" else (combined | s)
    return combined
