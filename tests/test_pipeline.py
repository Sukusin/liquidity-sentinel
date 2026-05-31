from __future__ import annotations

from pathlib import Path

import pandas as pd

from ru_liquidity_sentinel.backtest import run_backtest
from ru_liquidity_sentinel.pipeline import run_pipeline


def test_full_pipeline_and_backtest(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    paths = run_pipeline(data_dir, output_dir)
    run_backtest(data_dir, output_dir)

    history = pd.read_csv(paths["lsi_history"])
    assert not history.empty
    assert history["lsi"].between(0, 100).all()
    contrib_cols = [c for c in history.columns if c.startswith("contrib_") and c != "contrib_sum"]
    assert (history[contrib_cols].sum(axis=1) - history["lsi"]).abs().max() < 0.03
    assert {"GREEN", "YELLOW", "RED"}.intersection(set(history["status"]))

    backtest = pd.read_csv(output_dir / "backtest_report.csv")
    assert {"dec_2014", "feb_mar_2022", "aug_2023"}.issubset(set(backtest["episode"]))
    assert (backtest["max_lsi"] >= 40).all()
