from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .llm import generate_comment
from .utils import DATE_COL, project_root, status_ru


CONTRIB_COLS = [
    "contrib_M1_reserves",
    "contrib_M2_repo",
    "contrib_M3_ofz",
    "contrib_M5_treasury",
    "contrib_M4_tax_seasonality",
]

CONTRIB_LABELS = {
    "contrib_M1_reserves": "М1 резервы",
    "contrib_M2_repo": "М2 РЕПО",
    "contrib_M3_ofz": "М3 ОФЗ",
    "contrib_M5_treasury": "М5 казначейство",
    "contrib_M4_tax_seasonality": "М4 сезонность",
}


def _load_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    history = pd.read_csv(output_dir / "lsi_history.csv", parse_dates=[DATE_COL])
    backtest = pd.read_csv(output_dir / "backtest_report.csv") if (output_dir / "backtest_report.csv").exists() else None
    coeffs = pd.read_csv(output_dir / "model_coefficients.csv") if (output_dir / "model_coefficients.csv").exists() else None
    return history, backtest, coeffs


def build_dashboard_html(output_dir: Path | None = None, html_path: Path | None = None) -> Path:
    root = project_root()
    output_dir = output_dir or root / "output"
    html_path = html_path or output_dir / "dashboard.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    history, backtest, coeffs = _load_outputs(output_dir)
    latest = history.iloc[-1]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Liquidity Stress Index",
            "Вклад модулей в LSI",
            "Ключевые рыночные сигналы",
            "Казначейство и налоговые флаги",
        ),
    )
    fig.add_trace(go.Scatter(x=history[DATE_COL], y=history["lsi"], mode="lines", name="LSI"), row=1, col=1)
    fig.add_hrect(y0=0, y1=40, line_width=0, opacity=0.08, row=1, col=1)
    fig.add_hrect(y0=40, y1=70, line_width=0, opacity=0.08, row=1, col=1)
    fig.add_hrect(y0=70, y1=100, line_width=0, opacity=0.08, row=1, col=1)

    for col in CONTRIB_COLS:
        if col in history.columns:
            fig.add_trace(
                go.Scatter(x=history[DATE_COL], y=history[col], stackgroup="contrib", mode="lines", name=CONTRIB_LABELS[col]),
                row=2,
                col=1,
            )

    for col, name in [
        ("m1_mad_ruonia", "М1 RUONIA MAD"),
        ("m2_cover_ratio", "М2 cover РЕПО"),
        ("m3_cover_ratio", "М3 cover ОФЗ"),
    ]:
        if col in history.columns:
            fig.add_trace(go.Scatter(x=history[DATE_COL], y=history[col], mode="lines", name=name), row=3, col=1)

    for col, name in [
        ("m5_budget_delta_7d_bln", "М5 дельта бюджета 7д"),
        ("m4_seasonal_factor", "М4 seasonal factor"),
    ]:
        if col in history.columns:
            fig.add_trace(go.Scatter(x=history[DATE_COL], y=history[col], mode="lines", name=name), row=4, col=1)

    fig.update_layout(height=1050, title=" liquidity sentinel - дашборд раннего предупреждения", hovermode="x unified")
    fig.update_yaxes(range=[0, 100], row=1, col=1)

    comment = generate_comment(latest)
    status = status_ru(latest["status"])
    top_contrib = "".join(
        f"<li>{CONTRIB_LABELS.get(col, col)}: <b>{latest.get(col, 0):.1f}</b> п.</li>" for col in CONTRIB_COLS
    )

    backtest_html = ""
    if backtest is not None:
        backtest_html = "<h2>Backtest на стресс-эпизодах</h2>" + backtest.to_html(index=False, classes="table")

    coeffs_html = ""
    if coeffs is not None:
        coeffs_html = "<h2>Коэффициенты интерпретируемой ML-модели</h2>" + coeffs.to_html(index=False, classes="table")

    html = f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title> liquidity sentinel</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17212b; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,.04); }}
    .metric {{ font-size: 32px; font-weight: 700; }}
    .muted {{ color: #667085; }}
    .comment {{ border-left: 5px solid #334155; padding: 12px 16px; background: #f8fafc; border-radius: 8px; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
    .table th, .table td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    .table th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  <h1> liquidity sentinel</h1>
  <p class="muted">Система раннего выявления напряжения ликвидности по пяти модулям и интерпретируемой агрегации LSI.</p>
  <div class="cards">
    <div class="card"><div class="muted">Текущий LSI</div><div class="metric">{latest['lsi']:.1f}</div></div>
    <div class="card"><div class="muted">Статус</div><div class="metric">{status}</div></div>
    <div class="card"><div class="muted">Дата расчёта</div><div class="metric">{pd.to_datetime(latest[DATE_COL]).date()}</div></div>
  </div>
  <h2>Автоматический аналитический комментарий</h2>
  <div class="comment">{comment}</div>
  <h2>Вклад модулей в текущий LSI</h2>
  <ul>{top_contrib}</ul>
  {fig.to_html(full_html=False, include_plotlyjs='cdn')}
  {backtest_html}
  {coeffs_html}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path
