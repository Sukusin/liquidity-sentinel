from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATE_COL, write_csv


STRESS_EPISODES = {
    "dec_2014": ("2014-12-01", "2014-12-31"),
    "feb_mar_2022": ("2022-02-24", "2022-03-31"),
    "aug_2023": ("2023-08-01", "2023-08-31"),
}


def _gaussian_pulse(dates: pd.DatetimeIndex, center: str, width_days: float, amplitude: float) -> np.ndarray:
    center_ts = pd.Timestamp(center)
    delta = np.asarray((dates - center_ts).days, dtype=float)
    return amplitude * np.exp(-(delta**2) / (2 * width_days**2))


def _stress_intensity(dates: pd.DatetimeIndex) -> np.ndarray:
    return (
        _gaussian_pulse(dates, "2014-12-16", 16, 1.0)
        + _gaussian_pulse(dates, "2022-03-03", 24, 1.35)
        + _gaussian_pulse(dates, "2023-08-15", 18, 0.9)
    )


def generate_tax_calendar(start: str = "2014-01-01", end: str = "2026-03-31") -> pd.DataFrame:
    months = pd.period_range(start=start, end=end, freq="M")
    rows: list[dict[str, object]] = []
    for period in months:
        y, m = period.year, period.month
        # 25-е и 28-е число - основные точки налогового давления в упрощенной модели.
        for day, tax_name in [(25, "НДС / акцизы"), (28, "ЕНП / налог на прибыль / страховые взносы")]:
            date = pd.Timestamp(year=y, month=m, day=min(day, period.days_in_month))
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                rows.append({DATE_COL: date, "tax_name": tax_name, "importance": 1.0})
        # Квартальный эффект: конец марта, июня, сентября, декабря.
        if m in (3, 6, 9, 12):
            date = pd.Timestamp(year=y, month=m, day=period.days_in_month)
            rows.append({DATE_COL: date, "tax_name": "Конец квартала", "importance": 0.8})
    return pd.DataFrame(rows).sort_values(DATE_COL).reset_index(drop=True)


def generate_sample_raw_data(data_dir: Path, start: str = "2014-01-01", end: str = "2026-03-31") -> None:
    """Create deterministic synthetic-but-realistic data for offline demo and tests.

    The project is written so that real CSV/Excel data can replace these files.
    This generator exists because the execution environment used for checking may
    not have internet access to CBR/Minfin/Ros казна sources.
    """
    rng = np.random.default_rng(42)
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    stress = _stress_intensity(dates)
    year_trend = np.linspace(0, 1, n)
    weekly = np.sin(np.arange(n) / 7 * 2 * np.pi)
    monthly = np.sin(np.arange(n) / 30.4 * 2 * np.pi)

    tax_calendar = generate_tax_calendar(start, end)
    tax_dates = set(pd.to_datetime(tax_calendar[DATE_COL]).dt.normalize())
    tax_week = np.array([any(abs((d - t).days) <= 7 for t in tax_dates) for d in dates]).astype(float)

    # Key rate and RUONIA.
    key_rate = 7.0 + 0.7 * np.sin(np.arange(n) / 365.25 * 2 * np.pi) + 4.8 * stress
    key_rate += np.where((dates >= "2022-02-28") & (dates <= "2022-04-10"), 6.0, 0.0)
    ruonia = key_rate + 0.1 * weekly + 1.4 * stress + 0.25 * tax_week + rng.normal(0, 0.12, n)

    required_reserves = 980 + 280 * year_trend + 20 * np.sin(np.arange(n) / 180 * 2 * np.pi)
    actual_balances = required_reserves + 90 + 35 * monthly + 360 * stress + 70 * tax_week + rng.normal(0, 22, n)
    reserves = pd.DataFrame(
        {
            DATE_COL: dates,
            "actual_balances_bln": actual_balances.round(2),
            "required_reserves_bln": required_reserves.round(2),
            "accounting_reserves_bln": (required_reserves * 0.18).round(2),
            "ruonia": ruonia.round(3),
        }
    )

    allotment = 900 + 80 * np.sin(np.arange(n) / 90 * 2 * np.pi) + rng.normal(0, 35, n)
    demand = allotment * (1.05 + 1.6 * stress + 0.22 * tax_week + rng.normal(0, 0.08, n))
    cut_rate = key_rate + 0.15 + 1.0 * stress + 0.20 * tax_week + rng.normal(0, 0.08, n)
    repo = pd.DataFrame(
        {
            DATE_COL: dates,
            "term_days": 7,
            "demand_bln": demand.clip(min=50).round(2),
            "allotment_bln": allotment.clip(min=50).round(2),
            "cut_rate": cut_rate.round(3),
            "weighted_avg_rate": (cut_rate - 0.05 + rng.normal(0, 0.05, n)).round(3),
            "key_rate": key_rate.round(3),
        }
    )

    # OFZ auctions usually happen 1-2 times a week. Use Wednesdays and selected Fridays.
    auction_mask = (dates.weekday == 2) | ((dates.weekday == 4) & (np.arange(n) % 2 == 0))
    ofz_dates = dates[auction_mask]
    idx = np.searchsorted(dates.values, ofz_dates.values)
    offer = 120 + 25 * np.sin(idx / 65 * 2 * np.pi) + rng.normal(0, 8, len(ofz_dates))
    cover = 2.0 - 1.15 * stress[idx] - 0.18 * tax_week[idx] + rng.normal(0, 0.18, len(ofz_dates))
    demand_ofz = offer * cover.clip(min=0.35)
    placement = np.minimum(offer, demand_ofz * (0.86 + rng.normal(0, 0.03, len(ofz_dates))))
    curve_yield = 8.0 + 0.7 * np.sin(idx / 280 * 2 * np.pi) + 2.8 * stress[idx]
    wa_yield = curve_yield + 0.08 + 0.8 * stress[idx] + rng.normal(0, 0.08, len(ofz_dates))
    ofz = pd.DataFrame(
        {
            DATE_COL: ofz_dates,
            "issue": [f"ОФЗ-{26200 + i % 90}" for i in range(len(ofz_dates))],
            "offer_bln": offer.clip(min=20).round(2),
            "demand_bln": demand_ofz.clip(min=5).round(2),
            "placement_bln": placement.clip(min=0).round(2),
            "weighted_avg_yield": wa_yield.round(3),
            "curve_yield": curve_yield.round(3),
        }
    )

    budget_funds = 1400 + 230 * np.sin(np.arange(n) / 120 * 2 * np.pi) + 210 * monthly
    budget_funds += 900 * year_trend - 560 * stress - 190 * tax_week + rng.normal(0, 35, n)
    eks_deposits = 760 + 110 * np.sin(np.arange(n) / 75 * 2 * np.pi) - 300 * stress - 90 * tax_week + rng.normal(0, 22, n)
    treasury = pd.DataFrame(
        {
            DATE_COL: dates,
            "budget_funds_in_banks_bln": budget_funds.round(2),
            "eks_deposits_bln": eks_deposits.round(2),
            "participants": (15 + 5 * np.sin(np.arange(n) / 50) - 3 * stress + rng.normal(0, 1, n)).clip(min=3).round().astype(int),
        }
    )

    structural_liquidity = 2600 - 950 * stress - 180 * tax_week + 180 * np.sin(np.arange(n) / 180) + rng.normal(0, 65, n)
    stress_label = (stress > 0.45).astype(int)
    ground_truth = pd.DataFrame(
        {
            DATE_COL: dates,
            "structural_liquidity_bln": structural_liquidity.round(2),
            "stress_label": stress_label,
            "stress_intensity": stress.round(4),
        }
    )

    write_csv(reserves, data_dir / "reserves.csv")
    write_csv(repo, data_dir / "repo.csv")
    write_csv(ofz, data_dir / "ofz.csv")
    write_csv(tax_calendar, data_dir / "tax_calendar.csv")
    write_csv(treasury, data_dir / "treasury.csv")
    write_csv(ground_truth, data_dir / "ground_truth.csv")


def ensure_sample_data(data_dir: Path) -> None:
    required = ["reserves.csv", "repo.csv", "ofz.csv", "tax_calendar.csv", "treasury.csv", "ground_truth.csv"]
    if not all((data_dir / name).exists() for name in required):
        generate_sample_raw_data(data_dir)
