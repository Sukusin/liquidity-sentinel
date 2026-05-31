from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .utils import DATE_COL, status_ru


MODULE_NAMES = {
    "contrib_M1_reserves": "усреднение резервов / RUONIA",
    "contrib_M2_repo": "аукционы РЕПО ЦБ",
    "contrib_M3_ofz": "аукционы ОФЗ",
    "contrib_M5_treasury": "казначейство",
    "contrib_M4_tax_seasonality": "налоговая сезонность",
}


@dataclass
class CommentConfig:
    horizon_days: int = 7
    top_n: int = 3


def active_flags(row: pd.Series) -> list[str]:
    flags = []
    mapping = {
        "m1_flag_end_period": "конец периода усреднения резервов",
        "m2_flag_demand": "переспрос на РЕПО ЦБ",
        "m3_flag_nedospros": "недоспрос на ОФЗ",
        "m5_flag_budget_drain": "отток средств казначейства",
        "m4_tax_week_flag": "налоговая неделя",
        "m4_end_of_quarter_flag": "конец квартала",
    }
    for col, label in mapping.items():
        if int(row.get(col, 0) or 0) == 1:
            flags.append(label)
    return flags


def top_contributors(row: pd.Series, top_n: int = 3) -> list[tuple[str, float]]:
    items = []
    for col, name in MODULE_NAMES.items():
        value = float(row.get(col, 0.0) or 0.0)
        items.append((name, value))
    return sorted(items, key=lambda x: x[1], reverse=True)[:top_n]


def build_llm_prompt(row: pd.Series) -> str:
    flags = ", ".join(active_flags(row)) or "нет активных флагов"
    contributors = ", ".join([f"{name}: {value:.1f}" for name, value in top_contributors(row, top_n=5)])
    return (
        f"Текущий LSI: {row['lsi']:.1f} ({status_ru(row['status'])}).\n"
        f"Вклад модулей: {contributors}.\n"
        f"Активные флаги: {flags}.\n"
        "Задача: напиши аналитический комментарий на русском языке в 3-5 предложений: "
        "что происходило на рынке, какие факторы дали основной вклад, чего ожидать в ближайшие дни."
    )


def generate_comment(row: pd.Series, config: CommentConfig | None = None) -> str:
    config = config or CommentConfig()
    status = status_ru(str(row["status"]))
    leaders = top_contributors(row, top_n=config.top_n)
    leader_text = ", ".join([f"{name} ({value:.1f} п.)" for name, value in leaders if value > 0]) or "значимых факторов нет"
    flags = active_flags(row)
    flags_text = ", ".join(flags) if flags else "активных стресс-флагов нет"

    if row["status"] == "RED":
        tone = "Система фиксирует стресс ликвидности"
    elif row["status"] == "YELLOW":
        tone = "Система фиксирует повышенное напряжение ликвидности"
    else:
        tone = "Система оценивает состояние ликвидности как нормальное"

    comment = (
        f"{tone}: LSI = {float(row['lsi']):.1f}, статус - {status}. "
        f"Основной вклад дают: {leader_text}. "
        f"Активные флаги: {flags_text}. "
        "При интерпретации налоговой недели модель снижает вес пересекающихся сигналов М1/М2/М5, чтобы не завысить индекс из-за двойного счета. "
        "В ближайшие дни стоит отслеживать динамику RUONIA, cover ratio РЕПО и ОФЗ, а также недельную дельту средств казначейства."
    )
    return comment


def simple_rag_answer(question: str, lsi_history: pd.DataFrame) -> str:
    """Small data-grounded assistant for the dashboard Analyst tab.

    It does not pretend to be a full LLM. It answers by retrieving rows from the
    computed LSI history and summarising them. A real deployment can replace this
    function with an API/open-source LLM using build_llm_prompt + retrieved rows.
    """
    q = question.lower().strip()
    dates = re.findall(r"(20\d{2}|2014)", q)
    df = lsi_history.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    if "максим" in q or "топ" in q or "пик" in q:
        sample = df.nlargest(5, "lsi")
        lines = ["Периоды максимального стресса по рассчитанному LSI:"]
        for _, row in sample.iterrows():
            leaders = ", ".join([f"{name}: {value:.1f}" for name, value in top_contributors(row, 2)])
            lines.append(f"- {row[DATE_COL].date()}: LSI {row['lsi']:.1f}, {status_ru(row['status'])}; лидеры: {leaders}")
        return "\n".join(lines)

    if dates:
        year = int(dates[0])
        sample = df[df[DATE_COL].dt.year == year]
        # crude month detection for common Russian names
        months = {
            "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
            "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
        }
        for token, month in months.items():
            if token in q:
                sample = sample[sample[DATE_COL].dt.month == month]
                break
        if sample.empty:
            return "По указанному периоду нет данных в рассчитанной истории LSI."
        peak = sample.loc[sample["lsi"].idxmax()]
        mean_lsi = sample["lsi"].mean()
        leaders = ", ".join([f"{name}: {value:.1f}" for name, value in top_contributors(peak, 3)])
        return (
            f"За выбранный период средний LSI = {mean_lsi:.1f}. "
            f"Пиковое значение было {peak['lsi']:.1f} ({status_ru(peak['status'])}) "
            f"на дату {peak[DATE_COL].date()}. Главные факторы в пик: {leaders}."
        )

    latest = df.iloc[-1]
    return generate_comment(latest)
