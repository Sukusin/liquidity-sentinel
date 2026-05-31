# RU Liquidity Sentinel

Система раннего выявления стресса ликвидности рублёвого денежного рынка.
Проект реализует задание из ТЗ ПСБ: пять независимых модулей сигналов, MAD-нормализацию, интерпретируемый Liquidity Stress Index 0-100, backtest на стресс-эпизодах, защиту от двойного счёта налогового эффекта, HTML/Streamlit-дашборд и текстовый аналитический комментарий.

## Что реализовано

| Блок | Реализация |
|---|---|
| М1. Усреднение обязательных резервов | Спред фактических остатков к обязательным резервам, RUONIA, флаг конца периода, MAD-score |
| М2. Аукционы РЕПО ЦБ | 7-дневные аукционы, cover ratio, спред ставки отсечения к ключевой, флаг переспроса > 2.0, MAD-score |
| М3. Размещение ОФЗ | cover ratio спрос/предложение, спред доходности к кривой, флаги недоспроса < 1.2 и переспроса > 2.0, MAD-score |
| М4. Налоговый период | налоговая неделя, конец месяца, конец квартала, Seasonal_Factor 1.0-1.4 |
| М5. Казначейство | недельная дельта средств/депозитов казначейства в банках, флаг резкого оттока, MAD-score |
| LSI | Logistic Regression как простой интерпретируемый ML-агрегатор |
| Интерпретация | вклад каждого модуля в итоговый LSI, сумма вкладов равна LSI |
| Двойной счёт | в налоговые недели веса пересекающихся сигналов М1/М2/М5 уменьшаются, а М4 применяется как контекстный множитель |
| Backtest | декабрь 2014, февраль-март 2022, август 2023 |
| LLM-бонус | rule-based аналитический комментарий + промпт для подключения настоящей LLM + простой RAG-ответ по истории LSI |

## Структура проекта

```text
ru_liquidity_sentinel/
├── app.py                          # Streamlit-дашборд
├── check_project.py                # самопроверка по критериям
├── data/raw/                       # входные CSV; если их нет, создаются демо-данные
├── output/                         # результаты расчёта
├── src/ru_liquidity_sentinel/
│   ├── aggregation.py              # ML-агрегатор LSI + вклады модулей
│   ├── backtest.py                 # backtest и sensitivity analysis
│   ├── dashboard.py                # статический HTML-дашборд
│   ├── data_sources.py             # загрузка CSV и заготовки для официальных источников
│   ├── llm.py                      # автокомментарий и простой RAG-аналитик
│   ├── modules.py                  # М1-М5
│   ├── pipeline.py                 # полный расчётный пайплайн
│   ├── sample_data.py              # офлайн-демо-данные для проверки
│   └── utils.py                    # MAD, статусы, утилиты
└── tests/                          # pytest-проверки
```

## Быстрый запуск

```bash
cd ru_liquidity_sentinel
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# полный расчёт: сгенерирует демо-данные, рассчитает LSI, backtest и dashboard.html
PYTHONPATH=src python check_project.py
```

После успешного запуска откройте файл:

```text
output/dashboard.html
```

## Запуск по шагам

```bash
# 1. Создать демо-данные, если нет реальных CSV
PYTHONPATH=src python -m ru_liquidity_sentinel.cli generate-sample

# 2. Рассчитать сигналы М1-М5 и LSI
PYTHONPATH=src python -m ru_liquidity_sentinel.cli run

# 3. Провести backtest и sensitivity analysis
PYTHONPATH=src python -m ru_liquidity_sentinel.cli backtest

# 4. Собрать HTML-дашборд
PYTHONPATH=src python -m ru_liquidity_sentinel.cli dashboard
```

Streamlit-версия:

```bash
PYTHONPATH=src streamlit run app.py
```

## Формат входных данных

Проект принимает CSV в `data/raw/`. Если файлов нет, создаются детерминированные демо-данные, чтобы можно было проверить код без доступа к интернету.

### `reserves.csv`

```text
date,actual_balances_bln,required_reserves_bln,accounting_reserves_bln,ruonia
```

### `repo.csv`

```text
date,term_days,demand_bln,allotment_bln,cut_rate,weighted_avg_rate,key_rate
```

### `ofz.csv`

```text
date,issue,offer_bln,demand_bln,placement_bln,weighted_avg_yield,curve_yield
```

### `tax_calendar.csv`

```text
date,tax_name,importance
```

### `treasury.csv`

```text
date,budget_funds_in_banks_bln,eks_deposits_bln,participants
```

### `ground_truth.csv`

```text
date,structural_liquidity_bln,stress_label,stress_intensity
```

`stress_label` нужен для калибровки Logistic Regression. Если его нет, код может построить прокси-метку по нижнему квантилю структурной ликвидности или по высоким средним сигналам.

## Метод агрегации LSI

Используется `LogisticRegression(class_weight="balanced")` поверх MAD-нормализованных сигналов М1, М2, М3, М5. После обучения отрицательные коэффициенты стресс-признаков обнуляются, потому что все входные признаки сконструированы так, что большее значение означает больший стресс; это защищает интерпретацию от шумной калибровки на короткой истории. Модель выбрана потому, что:

1. это настоящий ML-метод, обучаемый на ground truth/proxy-метках;
2. линейная форма даёт понятную декомпозицию: вклад признака = `coef * value`;
3. вклады признаков агрегируются до уровня модулей;
4. результат переводится в шкалу LSI 0-100 как вероятность стрессового режима.

М4 не добавляется как обычное слагаемое. Налоговый период уже отражается в RUONIA, спросе на РЕПО и движениях казначейства, поэтому прямое сложение создало бы двойной счёт. В проекте М4 работает как контекст:

```text
в налоговую неделю: признаки М1/М2/М5 *= 0.75
LSI_final = min(100, LSI_structural * Seasonal_Factor)
```

Вклад М4 показывается как разница между `LSI_final` и структурным LSI после защиты от двойного счёта.

## Выходные файлы

| Файл | Назначение |
|---|---|
| `output/signals.csv` | рассчитанные сигналы М1-М5 |
| `output/lsi_history.csv` | сигналы + LSI + вклад каждого модуля |
| `output/model_coefficients.csv` | коэффициенты Logistic Regression для объяснения агрегации |
| `output/backtest_report.csv` | результаты на декабре 2014, феврале-марте 2022, августе 2023 |
| `output/sensitivity_analysis.csv` | чувствительность LSI к изменению весов модулей на ±20% |
| `output/dashboard.html` | интерактивный HTML-дашборд |

## Самопроверка

Команда:

```bash
PYTHONPATH=src python check_project.py
```

Проверяет:

- все обязательные сигналы М1-М5 присутствуют;
- LSI всегда находится в диапазоне 0-100;
- статусы соответствуют шкале 0-40 / 40-70 / 70-100;
- сумма вкладов модулей совпадает с LSI;
- в налоговые недели включается защита от двойного счёта;
- коэффициенты ML-модели сохранены;
- коэффициенты стресс-признаков неотрицательны;
- backtest покрывает обязательные стресс-эпизоды;
- sensitivity analysis выполнен;
- HTML-дашборд создан.

## Как заменить демо-данные на реальные

1. Скачайте данные из официальных источников ЦБ, Минфина, ФНС и Росказны.
2. Приведите их к CSV-схемам из раздела «Формат входных данных».
3. Положите файлы в `data/raw/` с теми же именами.
4. Запустите `PYTHONPATH=src python check_project.py`.

В `data_sources.py` оставлены URL и downloader-заготовка. Основная логика проекта отделена от парсеров, потому что официальные страницы могут менять HTML/Excel-структуру. Это уменьшает риск поломки расчётов из-за изменения вёрстки источника. Проверка публичных источников и текущие ограничения вынесены в `SOURCE_VALIDATION.md`.
