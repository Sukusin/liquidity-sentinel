from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import DATE_COL, as_date, assert_required_columns, positive_part, robust_mad_score, safe_divide, to_daily


MAD_WINDOW_DAYS = 756  # примерно 3 года торговых/календарных наблюдений в daily-представлении


def compute_m1_reserves(reserves: pd.DataFrame) -> pd.DataFrame:
    """М1: усреднение обязательных резервов + RUONIA."""
    required = [DATE_COL, "actual_balances_bln", "required_reserves_bln", "accounting_reserves_bln", "ruonia"]
    assert_required_columns(reserves, required, "M1 reserves")
    df = as_date(reserves)
    df["m1_reserve_spread_bln"] = df["actual_balances_bln"] - df["required_reserves_bln"]
    df["m1_mad_spread"] = positive_part(robust_mad_score(df["m1_reserve_spread_bln"], window=MAD_WINDOW_DAYS))
    df["m1_mad_ruonia"] = positive_part(robust_mad_score(df["ruonia"], window=MAD_WINDOW_DAYS))
    days_to_month_end = (df[DATE_COL] + pd.offsets.MonthEnd(0) - df[DATE_COL]).dt.days
    df["m1_flag_end_period"] = (days_to_month_end <= 5).astype(int)
    df["m1_signal"] = 0.55 * df["m1_mad_spread"] + 0.35 * df["m1_mad_ruonia"] + 0.10 * df["m1_flag_end_period"]
    return df[
        [
            DATE_COL,
            "m1_reserve_spread_bln",
            "ruonia",
            "m1_mad_spread",
            "m1_mad_ruonia",
            "m1_flag_end_period",
            "m1_signal",
        ]
    ]


def compute_m2_repo(repo: pd.DataFrame) -> pd.DataFrame:
    """М2: аукционы репо ЦБ, фокус на 7-дневных аукционах."""
    required = [DATE_COL, "term_days", "demand_bln", "allotment_bln", "cut_rate", "weighted_avg_rate", "key_rate"]
    assert_required_columns(repo, required, "M2 repo")
    df = as_date(repo)
    if (df["term_days"] == 7).any():
        df = df[df["term_days"] == 7].copy()
    df["m2_cover_ratio"] = safe_divide(df["demand_bln"], df["allotment_bln"])
    df["m2_rate_spread"] = df["cut_rate"] - df["key_rate"]
    df["m2_mad_cover"] = positive_part(robust_mad_score(df["m2_cover_ratio"], window=MAD_WINDOW_DAYS))
    df["m2_mad_rate_spread"] = positive_part(robust_mad_score(df["m2_rate_spread"], window=MAD_WINDOW_DAYS))
    df["m2_flag_demand"] = (df["m2_cover_ratio"] > 2.0).astype(int)
    df["m2_signal"] = 0.55 * df["m2_mad_cover"] + 0.35 * df["m2_mad_rate_spread"] + 0.10 * df["m2_flag_demand"]
    return df[
        [
            DATE_COL,
            "m2_cover_ratio",
            "m2_rate_spread",
            "m2_mad_cover",
            "m2_mad_rate_spread",
            "m2_flag_demand",
            "m2_signal",
        ]
    ]


def compute_m3_ofz(ofz: pd.DataFrame, full_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """М3: размещение ОФЗ. Низкий cover ratio - стресс."""
    required = [DATE_COL, "issue", "offer_bln", "demand_bln", "placement_bln", "weighted_avg_yield", "curve_yield"]
    assert_required_columns(ofz, required, "M3 OFZ")
    df = as_date(ofz)
    df["m3_cover_ratio"] = safe_divide(df["demand_bln"], df["offer_bln"])
    df["m3_yield_spread"] = df["weighted_avg_yield"] - df["curve_yield"]
    df["m3_mad_cover"] = positive_part(robust_mad_score(df["m3_cover_ratio"], window=260, min_periods=20, lower_is_stress=True))
    df["m3_mad_yield_spread"] = positive_part(robust_mad_score(df["m3_yield_spread"], window=260, min_periods=20))
    df["m3_flag_nedospros"] = (df["m3_cover_ratio"] < 1.2).astype(int)
    df["m3_flag_perespros"] = (df["m3_cover_ratio"] > 2.0).astype(int)
    # Переспрос - скорее признак избытка свободных денег, поэтому уменьшает стресс-сигнал.
    df["m3_signal"] = (
        0.55 * df["m3_mad_cover"]
        + 0.35 * df["m3_mad_yield_spread"]
        + 0.20 * df["m3_flag_nedospros"]
        - 0.10 * df["m3_flag_perespros"]
    ).clip(lower=0.0)
    out = df[
        [
            DATE_COL,
            "m3_cover_ratio",
            "m3_yield_spread",
            "m3_mad_cover",
            "m3_mad_yield_spread",
            "m3_flag_nedospros",
            "m3_flag_perespros",
            "m3_signal",
        ]
    ]
    if full_index is not None:
        # После аукциона сигнал имеет смысл несколько дней, до следующего размещения.
        out = to_daily(out, full_index=full_index, fill="ffill", limit=4).fillna(0.0)
    return out


def compute_m4_tax(tax_calendar: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """М4: налоговый календарь и сезонность.

    Этот модуль не складывается с остальными сигналами напрямую. Он формирует
    контекстный множитель и флаги, которые используются агрегатором.
    """
    required = [DATE_COL, "tax_name"]
    assert_required_columns(tax_calendar, required, "M4 tax calendar")
    tax = as_date(tax_calendar)
    tax_dates = pd.to_datetime(tax[DATE_COL]).dt.normalize().unique()
    df = pd.DataFrame({DATE_COL: pd.DatetimeIndex(full_index)})
    df["m4_tax_week_flag"] = [int(any(abs((d - t).days) <= 7 for t in tax_dates)) for d in df[DATE_COL]]
    days_to_month_end = (df[DATE_COL] + pd.offsets.MonthEnd(0) - df[DATE_COL]).dt.days
    df["m4_end_of_month_flag"] = (days_to_month_end <= 3).astype(int)
    df["m4_end_of_quarter_flag"] = ((df[DATE_COL].dt.month.isin([3, 6, 9, 12])) & (days_to_month_end <= 5)).astype(int)
    df["m4_seasonal_factor"] = (
        1.0
        + 0.20 * df["m4_tax_week_flag"]
        + 0.08 * df["m4_end_of_month_flag"]
        + 0.12 * df["m4_end_of_quarter_flag"]
    ).clip(1.0, 1.4)
    return df


def compute_m5_treasury(treasury: pd.DataFrame) -> pd.DataFrame:
    """М5: средства федерального казначейства.

    Внутренняя интерпретация: падение средств/депозитов в банках означает
    отток ликвидности из банковской системы, поэтому стресс считается по
    отрицательной недельной дельте.
    """
    required = [DATE_COL, "budget_funds_in_banks_bln", "eks_deposits_bln", "participants"]
    assert_required_columns(treasury, required, "M5 treasury")
    df = as_date(treasury)
    df["m5_budget_delta_7d_bln"] = df["budget_funds_in_banks_bln"].diff(7)
    df["m5_deposits_delta_7d_bln"] = df["eks_deposits_bln"].diff(7)
    # Стресс для М5 - не только резкий недельный отток, но и аномально
    # низкий уровень средств/депозитов в банках относительно истории. Такой
    # level-MAD устойчивее на длительных стресс-эпизодах, чем один diff(7).
    df["m5_mad_budget_drain"] = positive_part(
        robust_mad_score(df["budget_funds_in_banks_bln"], window=MAD_WINDOW_DAYS, lower_is_stress=True)
    )
    df["m5_mad_deposit_drop"] = positive_part(
        robust_mad_score(df["eks_deposits_bln"], window=MAD_WINDOW_DAYS, lower_is_stress=True)
    )
    df["m5_flag_budget_drain"] = ((df["m5_budget_delta_7d_bln"] < -300) | (df["m5_deposits_delta_7d_bln"] < -180)).astype(int)
    df["m5_signal"] = 0.55 * df["m5_mad_budget_drain"] + 0.35 * df["m5_mad_deposit_drop"] + 0.10 * df["m5_flag_budget_drain"]
    return df[
        [
            DATE_COL,
            "m5_budget_delta_7d_bln",
            "m5_deposits_delta_7d_bln",
            "m5_mad_budget_drain",
            "m5_mad_deposit_drop",
            "m5_flag_budget_drain",
            "m5_signal",
        ]
    ]
