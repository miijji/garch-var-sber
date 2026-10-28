"""Build the readable notebook; computed outputs are added by execute_notebook.py."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(r"""
# GARCH(1,1): прогноз волатильности, VaR и бэктестинг

Исследуем дневные цены Сбербанка (SBER). Сравним GARCH с нормальными ошибками,
GARCH со стандартизированными ошибками Стьюдента и исторический VaR.
Уровни доверия: **95% и 99%**, горизонт: **следующее торговое наблюдение**.

**Порядок:** данные → EDA на обучающем периоде → модель → последовательные прогнозы → бэктестинг → ES → выводы.
Доходности и риск выражены в процентных пунктах. Отрицательная граница доходности отличается знаком от VaR убытка.
Выберите окружение `.venv` в правом верхнем углу VS Code и выполните **Run All**.
""")
code("""
from pathlib import Path
import sys
import json
import platform
import importlib.metadata
from datetime import datetime, timezone

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists() and (p / 'src/garch_var').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from garch_var.data import download_moex, read_prices, log_returns, quality_report, sha256
from garch_var.analysis import stationarity, residual_diagnostics, eda_figure, correlation_figure, backtest_figure, conclusions
from garch_var.model import fit_garch, forecast_one, walk_forward, historical_forecasts
from garch_var.backtest import summarize

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.18})
pd.set_option('display.max_columns', 24)
print('Python:', sys.version.split()[0], '| interpreter:', sys.executable)
""")
md("""
## 1. Настройки и источник

Базовый период — 2016–2025, последние 500 доходностей оставляем для бэктеста.
Период фиксирован для воспроизводимости; это историческое исследование, а не актуальная торговая рекомендация.
Котировки скачиваются с MOEX ISS только при отсутствии локального снимка. При повторном запуске проверяется контрольная сумма.

Для Kaggle скачайте CSV и задайте `CSV_PATH`, имена столбцов и `SOURCE_URL`. Данные должны относиться к одному активу и одному наблюдению в день.
Пример найденного набора: [RUSSIA 249 Stocks Prices MOEX](https://www.kaggle.com/datasets/olegshpagin/russia-stocks-prices-ohlcv).
Основной источник здесь — Мосбиржа: доступен прямой запрос одного актива вместо загрузки большого набора разных частот.
""")
code("""
TICKER = 'SBER'
START, END = '2016-01-01', '2025-12-31'
TEST_SIZE = 500
LEVELS = (0.95, 0.99)
DISTRIBUTIONS = ('normal', 't')
CSV_PATH = None  # например ROOT / 'data/raw/SBER_D1.csv'
DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR = 'datetime', 'close', ','
SOURCE_URL = None  # ссылка на набор Kaggle при использовании CSV

if CSV_PATH is None:
    data_path, manifest = download_moex(ROOT, TICKER, START, END)
    prices = read_prices(data_path)
else:
    if not SOURCE_URL:
        raise ValueError('Укажите SOURCE_URL для импортируемого CSV.')
    data_path = Path(CSV_PATH)
    prices = read_prices(data_path, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR).loc[START:END]
    manifest = {'source': SOURCE_URL, 'input_file': data_path.name, 'sha256': sha256(data_path),
                'date_column': DATE_COLUMN, 'price_column': PRICE_COLUMN,
                'adjustment': 'Проверить корректировки по описанию исходного CSV.'}

returns = log_returns(prices)
if TEST_SIZE <= 0 or len(returns) - TEST_SIZE < 252:
    raise ValueError('Оставьте минимум 252 наблюдения для обучения.')
train, test = returns.iloc[:-TEST_SIZE], returns.iloc[-TEST_SIZE:]
run_id = 'notebook_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT = ROOT / 'outputs/runs' / run_id
OUT.mkdir(parents=True, exist_ok=False)
quality = quality_report(prices)
display(pd.Series(quality, name='Контроль качества'))
display(prices.head())
display(pd.DataFrame({'выборка': ['обучение', 'бэктест'], 'начало': [train.index[0], test.index[0]],
                      'конец': [train.index[-1], test.index[-1]], 'наблюдений': [len(train), len(test)]}))
""")
md(r"""
### Доходности и качество

\(r_t=100\ln(P_t/P_{t-1})\). Например, значение `-2` означает логарифмическую доходность −2 процентных пункта.
Неторговые дни не заполняются. Дубликаты, пропуски цен и неположительные цены вызывают ошибку.
Аномальные доходности сохраняются. Большие перерывы показаны в отчёте качества: доходность после остановки торгов охватывает весь перерыв.
MOEX close не корректируется на дивиденды; дивидендные разрывы влияют на оценку риска цены и отличаются от полной доходности инвестора.

## 2. EDA: только обучающий период

Графики и тесты этого раздела не используют отложенные наблюдения для выбора модели.
ADF: нулевая гипотеза — единичный корень. KPSS: нулевая гипотеза — стационарность вокруг константы.
ARCH-LM: нулевая гипотеза — отсутствие ARCH-эффекта до выбранного лага.
""")
code("""
eda_figure(prices.loc[:train.index[-1]], train).savefig(OUT / 'eda.png', dpi=140)
plt.show()
display(train.describe().to_frame('доходности, п.п.'))
tests = stationarity(train)
display(tests)
for _, row in tests.iterrows():
    decision = 'отвергается' if row.p_value < 0.05 else 'не отвергается'
    print(f'{row.test}: H0 «{row.null_hypothesis}» {decision} при 5%, p={row.p_value:.4g}')
tests.to_csv(OUT / 'stationarity.csv', index=False)
""")
code("""
correlation_figure(train).savefig(OUT / 'correlations.png', dpi=140)
plt.show()
""")
md(r"""
Зависимость квадратов доходностей при слабой зависимости самих доходностей согласуется с кластеризацией волатильности.
ACF/PACF и ARCH-LM рассматриваются совместно; график сам по себе не доказывает пригодность GARCH.

## 3. Обучение GARCH(1,1)

\(r_t=\mu+\epsilon_t,\quad \epsilon_t=\sigma_t z_t\),
\(\sigma_t^2=\omega+a\epsilon_{t-1}^2+b\sigma_{t-1}^2\).
Для ошибок Стьюдента используется распределение с единичной дисперсией.
Сумма \(a+b\), близкая к единице, показывает высокую устойчивость шоков; при \(a+b\ge1\) конечная безусловная дисперсия стандартной модели не существует.
Проверяем сходимость и зависимости стандартизированных остатков.
""")
code("""
initial_fits = {}
for distribution in DISTRIBUTIONS:
    fit = fit_garch(train, distribution)
    initial_fits[distribution] = fit
    print(f'\\nРаспределение: {distribution}')
    print(fit.summary())
    persistence = fit.params['alpha[1]'] + fit.params['beta[1]']
    print(f'alpha + beta = {persistence:.6f}')
    if persistence >= 1:
        print('Условие конечной безусловной дисперсии не выполнено.')
    diagnostics = residual_diagnostics(fit)
    display(diagnostics)
    diagnostics.to_csv(OUT / f'residual_diagnostics_{distribution}.csv', index=False)
    (OUT / f'initial_fit_{distribution}.txt').write_text(str(fit.summary()), encoding='utf-8')

fig, ax = plt.subplots(figsize=(12, 4))
for name, fit in initial_fits.items():
    ax.plot(fit.conditional_volatility, label=name, linewidth=0.8)
ax.set(title='Условная волатильность на обучении', ylabel='процентные пункты')
ax.legend()
fig.savefig(OUT / 'conditional_volatility.png', dpi=140)
plt.show()
""")
md(r"""
## 4. VaR и прогноз на один шаг

Пусть \(c\) — уровень доверия и \(\alpha=1-c\). Нижняя граница доходности:
\(q_{t+1,\alpha}=\hat\mu_{t+1}+\hat\sigma_{t+1}F^{-1}(\alpha)\).
При нормальных ошибках это \(\mu-Z_c\sigma\), где \(Z_c>0\).
**Нарушение:** \(r_{t+1}<q_{t+1,\alpha}\). **VaR убытка:** \(-q_{t+1,\alpha}\).

Для каждого следующего наблюдения модель заново обучается на расширяющейся истории.
Доходность проверяемого дня не участвует в его прогнозе. Дата `origin` — последнее известное наблюдение, индекс — дата проверки.
Исторический VaR использует ту же доступную историю. Ошибки сходимости останавливают расчёт, а не исчезают из статистики.
""")
code("""
parts = []
for distribution in DISTRIBUTIONS:
    parts.append(walk_forward(returns, TEST_SIZE, distribution, LEVELS, progress=print))
parts.append(historical_forecasts(returns, TEST_SIZE, LEVELS))
predictions = pd.concat(parts)
assert (pd.to_datetime(predictions.origin).to_numpy() < predictions.index.to_numpy()).all()
predictions.to_csv(OUT / 'predictions.csv')
display(predictions.head())
""")
md("""
## 5. Бэктестинг: доходности и нарушения

На графиках показаны отрицательные границы доходности. При 95% доверии ожидаемая частота нарушений — 5%, при 99% — 1%.
Это ожидаемые частоты, а не требование точного числа нарушений в конечной выборке.
""")
code("""
for model in predictions.model.unique():
    backtest_figure(predictions, model, LEVELS).savefig(OUT / f'backtest_{model}.png', dpi=140)
    plt.show()
summary = summarize(predictions, LEVELS)
display(summary[['model', 'confidence', 'n', 'exceptions', 'expected', 'exception_rate',
                 'kupiec_p', 'binomial_exact_p', 'independence_p', 'conditional_coverage_p', 'quantile_loss']])
summary.to_csv(OUT / 'backtest_summary.csv', index=False)
""")
md("""
Тест Купика оценивает соответствие частоты нарушений заданной вероятности. Тест Кристофферсена — зависимость соседних нарушений;
также сохранён совместный тест условного покрытия. Малый p-value (ниже 0.05) — свидетельство против соответствующей нулевой гипотезы.
Большой p-value не доказывает качество. Если один из типов предыдущего состояния не наблюдался, тест независимости не определён (`NaN`).
Для редких нарушений полезен точный биномиальный тест, а асимптотические результаты требуют осторожности.
`quantile_loss` — средняя квантильная потеря (меньше лучше), для сравнения при одинаковом уровне и наборе дат.

## 6. Expected Shortfall

ES оценивает средний условный убыток в нижнем хвосте модели и превышает VaR убытка.
Для Стьюдента учтена стандартизация до единичной дисперсии. Отдельно показываем фактический средний убыток в дни нарушения.
Сравнение этих средних — описание хвоста, а не полноценный бэктест ES; нарушений на уровне 99% может быть мало.
""")
code("""
display(summary[['model', 'confidence', 'exceptions', 'realized_tail_loss_pct', 'mean_predicted_es_on_hits_pct']])
next_forecasts = pd.DataFrame([forecast_one(returns, d, LEVELS) for d in DISTRIBUTIONS])
display(Markdown('### Прогноз после последнего наблюдения'))
display(next_forecasts[['model', 'origin', 'mu_pct', 'sigma_pct', 'variance_pct2',
                        'q_95_pct', 'var_95_loss_pct', 'es_95_loss_pct',
                        'q_99_pct', 'var_99_loss_pct', 'es_99_loss_pct']])
next_forecasts.to_csv(OUT / 'next_observation_forecast.csv', index=False)
print('Это прогноз на следующее наблюдение после конца выбранной истории, без проверки на будущей фактической доходности.')
""")
md("""
## 7. Выводы и сохранение результатов

Выводы ниже формируются из фактически рассчитанной статистики. Смена распределения — сравнение заранее заданных спецификаций;
если по тестовому периоду выбирать лучшую модель, для независимой финальной оценки потребуется новая выборка.
Прогноз после последней даты исторической выборки не является актуальным прогнозом на сегодня.
""")
code("""
report = conclusions(summary)
display(Markdown(report))
(OUT / 'REPORT.md').write_text(report, encoding='utf-8')
manifest['quality'] = quality
manifest['python'] = platform.python_version()
manifest['versions'] = {name: importlib.metadata.version(name) for name in ['arch', 'numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib']}
(OUT / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
settings = {'ticker': TICKER, 'start': START, 'end': END, 'test_size': TEST_SIZE, 'levels': LEVELS,
            'distributions': DISTRIBUTIONS, 'window': 'expanding', 'refit': 'every observation',
            'train_end': str(train.index[-1].date()), 'return_unit': 'percentage points',
            'source': 'MOEX' if CSV_PATH is None else str(CSV_PATH)}
(OUT / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
returns.to_csv(OUT / 'returns.csv')
print('Результаты сохранены:', OUT)
""")
md("""
## Источники и методические ссылки

- [MOEX ISS: справочник запросов](https://iss.moex.com/iss/reference/) — источник котировок.
- [arch: последовательные прогнозы волатильности](https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_forecasting.html).
- [arch: Value-at-Risk Forecasting](https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_forecasting.html#value-at-risk-forecasting).
- [statsmodels: ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) и [KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html).

Снимки котировок локальны и исключены из Git. Перед распространением данных следует проверить условия источника.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (GARCH VaR .venv)', 'language': 'python', 'name': 'garch-var'},
    'language_info': {'name': 'python', 'version': '3.13'},
})
(root / 'outputs').mkdir(exist_ok=True)
nbf.write(nb, root / 'outputs/garch_var_research.ipynb')
print('Notebook created:', len(cells), 'cells')
