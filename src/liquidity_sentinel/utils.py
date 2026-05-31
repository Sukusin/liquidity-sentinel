from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATE_COL = "date"


@dataclass(frozen=True)
class StatusThresholds:
    green_max: float = 40.0
    yellow_max: float = 70.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def as_date(df: pd.DataFrame, col: str = DATE_COL) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.normalize()
    return out.sort_values(col).reset_index(drop=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def robust_mad_score(
    series: pd.Series,
    window: int = 756,
    min_periods: int = 30,
    lower_is_stress: bool = False,
    clip: float = 8.0,
) -> pd.Series:
    """Rolling MAD z-score.

    Positive values should mean higher liquidity stress. Use lower_is_stress=True
    for indicators where abnormally low values are a stress signal, e.g. OFZ cover.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    median = s.rolling(window=window, min_periods=min_periods).median()

    def mad(x: np.ndarray) -> float:
        med = np.nanmedian(x)
        value = np.nanmedian(np.abs(x - med))
        return float(value) if np.isfinite(value) and value > 1e-12 else np.nan

    mad_values = s.rolling(window=window, min_periods=min_periods).apply(mad, raw=True)
    scaled_mad = 1.4826 * mad_values
    score = (s - median) / scaled_mad
    if lower_is_stress:
        score = -score
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return score.clip(-clip, clip)


def positive_part(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def status_from_lsi(value: float, thresholds: StatusThresholds | None = None) -> str:
    thresholds = thresholds or StatusThresholds()
    if value < thresholds.green_max:
        return "GREEN"
    if value < thresholds.yellow_max:
        return "YELLOW"
    return "RED"


def status_ru(status: str) -> str:
    return {"GREEN": "Норма", "YELLOW": "Напряжение", "RED": "Стресс"}.get(status, status)


def daily_frame(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({DATE_COL: pd.date_range(start=start, end=end, freq="D")})


def to_daily(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    fill: str = "ffill",
    limit: int | None = None,
    full_index: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Convert an event/monthly table to a daily frame."""
    d = as_date(df, date_col).set_index(date_col)
    if full_index is None:
        index = pd.date_range(d.index.min(), d.index.max(), freq="D")
    else:
        index = pd.DatetimeIndex(full_index)
    d = d.reindex(index)
    if fill == "ffill":
        d = d.ffill(limit=limit)
    elif fill == "zero":
        d = d.fillna(0.0)
    elif fill == "interpolate":
        d = d.interpolate(limit_direction="both")
    d = d.reset_index().rename(columns={"index": date_col})
    return d


def assert_required_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL in df.columns:
        df = as_date(df)
    return df
