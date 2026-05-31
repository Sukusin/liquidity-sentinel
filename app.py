from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from liquidity_sentinel.llm import generate_comment, simple_rag_answer
from liquidity_sentinel.pipeline import run_pipeline
from liquidity_sentinel.backtest import run_backtest
from liquidity_sentinel.utils import DATE_COL, project_root, status_ru

ROOT = project_root()
OUTPUT = ROOT / "output"
DATA = ROOT / "data" / "raw"

st.set_page_config(page_title=" liquidity sentinel", layout="wide")
st.title(" liquidity sentinel")
st.caption("Система раннего выявления стресса ликвидности рублёвого денежного рынка")

if st.sidebar.button("Пересчитать LSI") or not (OUTPUT / "lsi_history.csv").exists():
    run_pipeline(DATA, OUTPUT)
    run_backtest(DATA, OUTPUT)

history = pd.read_csv(OUTPUT / "lsi_history.csv", parse_dates=[DATE_COL])
latest = history.iloc[-1]

page = st.sidebar.radio("Раздел", ["Дашборд", "Аналитик"])

if page == "Дашборд":
    c1, c2, c3 = st.columns(3)
    c1.metric("LSI", f"{latest['lsi']:.1f}")
    c2.metric("Статус", status_ru(latest["status"]))
    c3.metric("Дата", str(latest[DATE_COL].date()))

    st.subheader("Автоматический комментарий")
    st.info(generate_comment(latest))

    fig_lsi = px.line(history, x=DATE_COL, y="lsi", title="Динамика Liquidity Stress Index")
    fig_lsi.add_hline(y=40, line_dash="dash")
    fig_lsi.add_hline(y=70, line_dash="dash")
    st.plotly_chart(fig_lsi, use_container_width=True)

    contrib_cols = [c for c in history.columns if c.startswith("contrib_") and c != "contrib_sum"]
    st.plotly_chart(px.area(history, x=DATE_COL, y=contrib_cols, title="Разложение LSI по модулям"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(history, x=DATE_COL, y=["m2_cover_ratio", "m3_cover_ratio"], title="Cover ratio РЕПО и ОФЗ"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(history, x=DATE_COL, y=["m5_budget_delta_7d_bln", "m4_seasonal_factor"], title="Казначейство и сезонность"), use_container_width=True)

    if (OUTPUT / "backtest_report.csv").exists():
        st.subheader("Backtest")
        st.dataframe(pd.read_csv(OUTPUT / "backtest_report.csv"), use_container_width=True)
else:
    st.subheader("Аналитик: вопросы по истории LSI")
    st.caption("Базовый RAG-режим: ответы строятся только по рассчитанным данным системы.")
    question = st.text_input("Вопрос", value="Почему в августе 2023 вырос LSI?")
    if question:
        st.write(simple_rag_answer(question, history))
