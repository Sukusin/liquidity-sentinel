from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ru_liquidity_sentinel.backtest import run_backtest
from ru_liquidity_sentinel.dashboard import build_dashboard_html
from ru_liquidity_sentinel.pipeline import run_pipeline


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"OK: {message}")


def main() -> None:
    output = ROOT / "output"
    run_pipeline(ROOT / "data" / "raw", output)
    run_backtest(ROOT / "data" / "raw", output)
    build_dashboard_html(output, output / "dashboard.html")

    history = pd.read_csv(output / "lsi_history.csv")
    backtest = pd.read_csv(output / "backtest_report.csv")
    sensitivity = pd.read_csv(output / "sensitivity_analysis.csv")
    coeffs = pd.read_csv(output / "model_coefficients.csv")

    required_module_cols = [
        "m1_mad_spread",
        "m1_mad_ruonia",
        "m2_mad_cover",
        "m2_mad_rate_spread",
        "m3_mad_cover",
        "m3_mad_yield_spread",
        "m4_tax_week_flag",
        "m4_seasonal_factor",
        "m5_mad_budget_drain",
        "m5_mad_deposit_drop",
        "lsi",
        "status",
    ]
    check(all(col in history.columns for col in required_module_cols), "все обязательные сигналы М1-М5 и LSI присутствуют")
    check(history["lsi"].between(0, 100).all(), "LSI всегда находится в диапазоне 0-100")
    check(set(history["status"]).issubset({"GREEN", "YELLOW", "RED"}), "статусы используют только GREEN/YELLOW/RED")
    contrib_cols = [c for c in history.columns if c.startswith("contrib_") and c != "contrib_sum"]
    max_gap = (history[contrib_cols].sum(axis=1) - history["lsi"]).abs().max()
    check(max_gap < 0.03, f"сумма вкладов модулей совпадает с LSI, max_gap={max_gap:.4f}")
    check((history.loc[history["m4_tax_week_flag"] == 1, "tax_dampening_applied"] == 1).all(), "в налоговые недели применяется защита от двойного счета")
    check(len(coeffs) >= 10 and {"module", "feature", "coef"}.issubset(coeffs.columns), "коэффициенты ML-агрегатора сохранены и интерпретируемы")
    check((coeffs["coef"] >= 0).all(), "коэффициенты стресс-признаков неотрицательны и не противоречат экономическому смыслу")
    check((backtest["found"] == True).all(), "все обязательные стресс-эпизоды найдены в истории")
    check((backtest["max_lsi"] >= 40).all(), "на каждом стресс-эпизоде LSI показывает минимум напряжение")
    check(len(sensitivity) >= 8, "sensitivity analysis по весам ±20% выполнен")
    check((output / "dashboard.html").exists(), "HTML-дашборд создан")
    print("\nSELF-CHECK PASSED")


if __name__ == "__main__":
    main()
