"""Training-only diagnostics and readable research figures."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for name, null, function in [
        ("ADF", "Единичный корень (нестационарность)", lambda: adfuller(returns, autolag="AIC")),
        ("KPSS", "Стационарность вокруг константы", lambda: kpss(returns, regression="c", nlags="auto")),
    ]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = function()
        rows.append({"test": name, "null_hypothesis": null, "statistic": result[0], "p_value": result[1],
                     "lags": result[2], "note": "; ".join(str(w.message) for w in caught)})
    lm, lm_p, _, _ = het_arch(returns - returns.mean(), nlags=10, ddof=1)
    rows.append({"test": "ARCH-LM", "null_hypothesis": "Нет ARCH-эффекта до лага 10", "statistic": lm,
                 "p_value": lm_p, "lags": 10, "note": ""})
    return pd.DataFrame(rows)


def residual_diagnostics(fit) -> pd.DataFrame:
    standardized = fit.std_resid.dropna()
    rows = []
    for name, values in [("residuals", standardized), ("squared_residuals", standardized**2)]:
        lb = acorr_ljungbox(values, lags=[10, 20], return_df=True)
        for lag, row in lb.iterrows():
            rows.append({"series": name, "test": "Ljung-Box", "lag": lag,
                         "statistic": row.lb_stat, "p_value": row.lb_pvalue})
    lm, p, _, _ = het_arch(standardized, nlags=10, ddof=0)
    rows.append({"series": "residuals", "test": "ARCH-LM", "lag": 10, "statistic": lm, "p_value": p})
    return pd.DataFrame(rows)


def eda_figure(prices, returns):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(prices.index, prices.close, color="#235789")
    axes[0].set(title="Цены закрытия: обучающий период", ylabel="руб.")
    axes[1].plot(returns.index, returns, color="#248277", linewidth=.7)
    axes[1].set(title="Логарифмические доходности", ylabel="процентные пункты")
    axes[2].hist(returns, bins=70, color="#235789", alpha=.8, density=True)
    axes[2].set(title="Распределение доходностей", xlabel="процентные пункты", ylabel="плотность")
    return fig


def correlation_figure(returns, lags=30):
    lags = min(lags, len(returns) // 2 - 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_acf(returns, lags=lags, ax=axes[0, 0], zero=False, title="ACF доходностей")
    plot_pacf(returns, lags=lags, ax=axes[0, 1], zero=False, method="ywm", title="PACF доходностей")
    plot_acf(returns**2, lags=lags, ax=axes[1, 0], zero=False, title="ACF квадратов доходностей")
    plot_pacf(returns**2, lags=lags, ax=axes[1, 1], zero=False, method="ywm", title="PACF квадратов доходностей")
    return fig


def backtest_figure(predictions, model="GARCH-normal", levels=(.95, .99)):
    group = predictions[predictions.model == model]
    if group.empty:
        raise ValueError(f"Нет прогнозов {model}.")
    fig, axes = plt.subplots(len(levels), 1, figsize=(13, 4 * len(levels)), sharex=True, squeeze=False, constrained_layout=True)
    for ax, level in zip(axes[:, 0], levels):
        tag = str(int(round(level * 100)))
        hits = group.actual_pct < group[f"q_{tag}_pct"]
        ax.plot(group.index, group.actual_pct, color="#56616d", linewidth=.6, label="Фактическая доходность")
        ax.plot(group.index, group[f"q_{tag}_pct"], color="#235789", label=f"Нижняя граница VaR {tag}%")
        ax.scatter(group.index[hits], group.actual_pct[hits], color="#d1495b", s=22, zorder=3, label=f"Нарушения: {hits.sum()}")
        ax.set(title=f"{model}: VaR {tag}% вне обучающей выборки", ylabel="процентные пункты")
        ax.legend(loc="lower left")
    return fig


def conclusions(summary: pd.DataFrame) -> str:
    lines = ["# Результаты бэктестинга", "", "VaR показан как нижняя граница доходности; нарушение означает доходность ниже этой границы.", ""]
    for _, row in summary.iterrows():
        verdict = "отвергает" if row.kupiec_p < .05 else "не отвергает"
        independence = ("не определён из-за отсутствия одного из состояний переходов" if not np.isfinite(row.independence_p)
                        else f"p-value = {row.independence_p:.4f}")
        lines.append(f"- {row.model}, {row.confidence:.0%}: {int(row.exceptions)} нарушений из {int(row.n)} "
                     f"({row.exception_rate:.2%}), ожидается {row.expected:.1f}. Тест Купика {verdict} "
                     f"соответствие частоты на уровне 5% (p = {row.kupiec_p:.4f}). Тест независимости: {independence}.")
    lines += ["", "Неотвержение гипотезы не доказывает качество модели. Асимптотические тесты могут иметь малую мощность при редких нарушениях; дополнительно сохранён точный биномиальный p-value.",
              "", "ES — модельное условное среднее хвостового убытка. Средний фактический убыток в дни нарушения — описательная оценка, а не полноценный тест ES. При отсутствии нарушений он не определён.",
              "", "Ограничения: цены не скорректированы на дивиденды; при остановках торгов доходность между соседними наблюдениями охватывает несколько календарных дней. Результаты относятся к выбранной исторической выборке."]
    return "\n".join(lines)
