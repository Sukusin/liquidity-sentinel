from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aggregation import LogisticLSIAggregator
from .pipeline import build_signal_frame, fit_and_score
from .sample_data import STRESS_EPISODES
from .utils import DATE_COL, project_root, status_from_lsi, write_csv


def backtest_episodes(lsi_history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, (start, end) in STRESS_EPISODES.items():
        mask = (lsi_history[DATE_COL] >= pd.Timestamp(start)) & (lsi_history[DATE_COL] <= pd.Timestamp(end))
        sample = lsi_history.loc[mask]
        if sample.empty:
            rows.append({"episode": name, "start": start, "end": end, "found": False})
            continue
        max_row = sample.loc[sample["lsi"].idxmax()]
        rows.append(
            {
                "episode": name,
                "start": start,
                "end": end,
                "found": True,
                "mean_lsi": round(float(sample["lsi"].mean()), 2),
                "max_lsi": round(float(sample["lsi"].max()), 2),
                "max_lsi_date": max_row[DATE_COL].date().isoformat(),
                "days_yellow_or_red": int((sample["lsi"] >= 40).sum()),
                "days_red": int((sample["lsi"] >= 70).sum()),
                "peak_status": status_from_lsi(float(sample["lsi"].max())),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_analysis(frame: pd.DataFrame, model: LogisticLSIAggregator) -> pd.DataFrame:
    base = model.predict(frame)
    rows: list[dict[str, object]] = []
    for module in ["M1_reserves", "M2_repo", "M3_ofz", "M5_treasury"]:
        for multiplier in [0.8, 1.2]:
            scenario = model.predict(frame, weight_multipliers={module: multiplier})
            rows.append(
                {
                    "module": module,
                    "multiplier": multiplier,
                    "mean_lsi_delta": round(float((scenario["lsi"] - base["lsi"]).mean()), 3),
                    "max_abs_lsi_delta": round(float((scenario["lsi"] - base["lsi"]).abs().max()), 3),
                    "red_days_delta": int((scenario["lsi"] >= 70).sum() - (base["lsi"] >= 70).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_backtest(data_dir: Path | None = None, output_dir: Path | None = None, train_until: str = "2021-12-31") -> dict[str, Path]:
    root = project_root()
    data_dir = data_dir or root / "data" / "raw"
    output_dir = output_dir or root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_signal_frame(data_dir)
    scored, model = fit_and_score(frame, train_until=train_until)
    full = frame.merge(scored, on=DATE_COL, how="left")
    backtest = backtest_episodes(full)
    sensitivity = sensitivity_analysis(frame, model)

    paths = {
        "backtest": output_dir / "backtest_report.csv",
        "sensitivity": output_dir / "sensitivity_analysis.csv",
    }
    write_csv(backtest, paths["backtest"])
    write_csv(sensitivity, paths["sensitivity"])
    return paths
