from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .utils import DATE_COL, status_from_lsi


FEATURE_GROUPS: dict[str, list[str]] = {
    "M1_reserves": ["m1_mad_spread", "m1_mad_ruonia", "m1_flag_end_period"],
    "M2_repo": ["m2_mad_cover", "m2_mad_rate_spread", "m2_flag_demand"],
    "M3_ofz": ["m3_mad_cover", "m3_mad_yield_spread", "m3_flag_nedospros"],
    "M5_treasury": ["m5_mad_budget_drain", "m5_mad_deposit_drop", "m5_flag_budget_drain"],
}

ALL_FEATURES: list[str] = [feature for cols in FEATURE_GROUPS.values() for feature in cols]


@dataclass
class LogisticLSIAggregator:
    """Interpretable ML aggregator for Liquidity Stress Index.

    The model is deliberately simple: logistic regression over robust module
    signals. Interpretability is obtained by decomposing the linear logit into
    module-level positive contributions. M4 is treated separately as a seasonal
    context factor to avoid double counting tax-week effects already visible in
    M1/M2/M5.
    """

    tax_dampening: float = 0.75
    probability_temperature: float = 6.0
    random_state: int = 42
    model: LogisticRegression = field(init=False)
    coefficients_: pd.Series | None = field(default=None, init=False)
    intercept_: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=self.random_state, C=0.12)

    def _prepare_features(self, df: pd.DataFrame, weight_multipliers: Mapping[str, float] | None = None) -> pd.DataFrame:
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features for LSI aggregation: {missing}")
        x = df[ALL_FEATURES].copy().apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        x = x.clip(lower=0.0, upper=8.0)

        if "m4_tax_week_flag" in df.columns:
            tax_mask = pd.to_numeric(df["m4_tax_week_flag"], errors="coerce").fillna(0).astype(bool)
            overlap_modules = ["M1_reserves", "M2_repo", "M5_treasury"]
            for module in overlap_modules:
                for col in FEATURE_GROUPS[module]:
                    x.loc[tax_mask, col] = x.loc[tax_mask, col] * self.tax_dampening

        if weight_multipliers:
            for module, multiplier in weight_multipliers.items():
                for col in FEATURE_GROUPS.get(module, []):
                    x[col] = x[col] * float(multiplier)
        return x

    def fit(self, df: pd.DataFrame, labels: pd.Series | np.ndarray) -> "LogisticLSIAggregator":
        x = self._prepare_features(df)
        y = pd.Series(labels).astype(int).values
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two classes to fit LSI aggregator")
        self.model.fit(x, y)
        raw_coef = pd.Series(self.model.coef_[0], index=ALL_FEATURES)
        # Экономический смысл всех входных признаков одинаковый: большее значение
        # означает больший стресс ликвидности. Поэтому отрицательный коэффициент
        # после обучения обнуляем, а не позволяем стресс-сигналу уменьшать LSI.
        # Это сохраняет ML-калибровку, но делает объяснение устойчивым и
        # экономически интерпретируемым на небольших/шумных выборках.
        self.coefficients_ = raw_coef.clip(lower=0.0)
        self.intercept_ = float(self.model.intercept_[0])
        return self

    def predict(self, df: pd.DataFrame, weight_multipliers: Mapping[str, float] | None = None) -> pd.DataFrame:
        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("Aggregator is not fitted")
        x = self._prepare_features(df, weight_multipliers=weight_multipliers)
        logit = self.intercept_ + np.dot(x.values, self.coefficients_.values)
        base_prob = 1.0 / (1.0 + np.exp(-(logit / self.probability_temperature)))
        base_lsi = np.clip(base_prob * 100.0, 0.0, 100.0)
        seasonal_factor = pd.to_numeric(df.get("m4_seasonal_factor", 1.0), errors="coerce").fillna(1.0).clip(1.0, 1.4).values
        lsi = np.clip(base_lsi * seasonal_factor, 0.0, 100.0)

        linear_parts = x.mul(self.coefficients_, axis=1)
        positive_parts = linear_parts.clip(lower=0.0)
        raw_module = pd.DataFrame(index=df.index)
        for module, features in FEATURE_GROUPS.items():
            raw_module[module] = positive_parts[features].sum(axis=1)
        raw_sum = raw_module.sum(axis=1).replace(0, np.nan)

        m4_extra = np.maximum(lsi - base_lsi, 0.0)
        structural_total = np.maximum(lsi - m4_extra, 0.0)
        weights = raw_module.div(raw_sum, axis=0)
        no_positive_signal = raw_sum.isna()
        if no_positive_signal.any():
            # Low LSI can still be non-zero because of model intercept/calibration.
            # Distribute this small baseline equally across structural modules so
            # the displayed decomposition always sums to LSI.
            weights.loc[no_positive_signal, list(FEATURE_GROUPS.keys())] = 1.0 / len(FEATURE_GROUPS)
        contributions = weights.fillna(0.0).mul(structural_total, axis=0)
        contributions["M4_tax_seasonality"] = m4_extra

        row_sum = contributions.sum(axis=1).replace(0, np.nan)
        scale = pd.Series(lsi, index=df.index) / row_sum
        contributions = contributions.mul(scale.fillna(0.0), axis=0)

        out = pd.DataFrame({DATE_COL: df[DATE_COL].values if DATE_COL in df.columns else df.index})
        out["base_lsi_without_m4"] = base_lsi.round(3)
        out["lsi"] = lsi.round(3)
        out["status"] = [status_from_lsi(v) for v in lsi]
        out["tax_dampening_applied"] = pd.to_numeric(df.get("m4_tax_week_flag", 0), errors="coerce").fillna(0).astype(int).values
        for col in contributions.columns:
            out[f"contrib_{col}"] = contributions[col].round(3).values
        # Interpretability diagnostics: sum should be equal to LSI up to rounding.
        contrib_cols = [c for c in out.columns if c.startswith("contrib_")]
        out["contrib_sum"] = out[contrib_cols].sum(axis=1).round(3)
        return out

    def coefficients_table(self) -> pd.DataFrame:
        if self.coefficients_ is None:
            raise RuntimeError("Aggregator is not fitted")
        rows = []
        for module, features in FEATURE_GROUPS.items():
            for feature in features:
                rows.append({"module": module, "feature": feature, "coef": float(self.coefficients_.loc[feature])})
        return pd.DataFrame(rows)


def build_labels(frame: pd.DataFrame) -> pd.Series:
    """Prefer official/proxy ground-truth label if available; otherwise derive one."""
    if "stress_label" in frame.columns:
        return pd.to_numeric(frame["stress_label"], errors="coerce").fillna(0).astype(int)
    if "structural_liquidity_bln" in frame.columns:
        liq = pd.to_numeric(frame["structural_liquidity_bln"], errors="coerce")
        threshold = liq.quantile(0.15)
        return (liq <= threshold).astype(int)
    structural_signal = frame[[c for c in frame.columns if c.endswith("_signal")]].mean(axis=1)
    return (structural_signal >= structural_signal.quantile(0.85)).astype(int)
