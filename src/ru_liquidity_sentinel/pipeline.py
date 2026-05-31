from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aggregation import LogisticLSIAggregator, build_labels
from .data_sources import load_raw_data
from .modules import compute_m1_reserves, compute_m2_repo, compute_m3_ofz, compute_m4_tax, compute_m5_treasury
from .utils import DATE_COL, as_date, project_root, write_csv


def build_signal_frame(data_dir: Path) -> pd.DataFrame:
    raw = load_raw_data(data_dir)
    m1 = compute_m1_reserves(raw["reserves"])
    m2 = compute_m2_repo(raw["repo"])
    m5 = compute_m5_treasury(raw["treasury"])

    start = max(m1[DATE_COL].min(), m2[DATE_COL].min(), m5[DATE_COL].min())
    end = min(m1[DATE_COL].max(), m2[DATE_COL].max(), m5[DATE_COL].max())
    full_index = pd.date_range(start=start, end=end, freq="D")
    m3 = compute_m3_ofz(raw["ofz"], full_index=full_index)
    m4 = compute_m4_tax(raw["tax_calendar"], full_index=full_index)

    frame = pd.DataFrame({DATE_COL: full_index})
    for part in [m1, m2, m3, m4, m5, raw["ground_truth"]]:
        part = as_date(part)
        frame = frame.merge(part, on=DATE_COL, how="left")

    # Missing event signals are no signal, not missing stress.
    signal_cols = [c for c in frame.columns if c.startswith(("m1_", "m2_", "m3_", "m4_", "m5_"))]
    frame[signal_cols] = frame[signal_cols].fillna(0.0)
    return frame


def fit_and_score(frame: pd.DataFrame, train_until: str = "2021-12-31") -> tuple[pd.DataFrame, LogisticLSIAggregator]:
    labels = build_labels(frame)
    train_mask = frame[DATE_COL] <= pd.Timestamp(train_until)
    # Keep the first major stress episode for calibration and use 2022/2023 as out-of-sample checks.
    model = LogisticLSIAggregator(tax_dampening=0.75)
    model.fit(frame.loc[train_mask].reset_index(drop=True), labels.loc[train_mask].reset_index(drop=True))
    scored = model.predict(frame)
    return scored, model


def run_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    train_until: str = "2021-12-31",
) -> dict[str, Path]:
    root = project_root()
    data_dir = data_dir or root / "data" / "raw"
    output_dir = output_dir or root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    signals = build_signal_frame(data_dir)
    scored, model = fit_and_score(signals, train_until=train_until)
    full = signals.merge(scored, on=DATE_COL, how="left")

    paths = {
        "signals": output_dir / "signals.csv",
        "lsi_history": output_dir / "lsi_history.csv",
        "coefficients": output_dir / "model_coefficients.csv",
    }
    write_csv(signals, paths["signals"])
    write_csv(full, paths["lsi_history"])
    write_csv(model.coefficients_table(), paths["coefficients"])
    return paths
