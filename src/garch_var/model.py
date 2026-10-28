"""Expanding-window forecasts. Inputs/outputs are percentage points, not fractions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t


def validate_returns(returns: pd.Series):
    if not isinstance(returns.index, pd.DatetimeIndex) or not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("Доходности должны иметь уникальный отсортированный DatetimeIndex.")
    if not np.isfinite(returns.to_numpy()).all() or len(returns) < 100 or returns.std() <= 0:
        raise ValueError("Нужно не менее 100 конечных, непостоянных доходностей.")


def fit_garch(history: pd.Series, distribution="normal"):
    validate_returns(history)
    if distribution not in ("normal", "t"):
        raise ValueError("Поддерживаются normal и t.")
    model = arch_model(history, mean="Constant", vol="GARCH", p=1, q=1, dist=distribution, rescale=False)
    result = model.fit(disp="off", show_warning=False, options={"maxiter": 2000})
    if result.convergence_flag != 0 or not np.isfinite(result.params).all():
        raise RuntimeError(f"GARCH не сошёлся на истории до {history.index[-1]}: {result.optimization_result.message}")
    return result


def tail_parameters(alpha: float, distribution="normal", nu=None) -> tuple[float, float]:
    if not 0 < alpha < 0.5:
        raise ValueError("Хвостовая вероятность должна быть между 0 и 0.5.")
    if distribution == "normal":
        q = norm.ppf(alpha)
        return float(q), float(-norm.pdf(q) / alpha)
    if distribution != "t" or nu is None or not np.isfinite(nu) or nu <= 2:
        raise ValueError("Для стандартизированного Стьюдента нужны nu > 2.")
    scale = np.sqrt((nu - 2) / nu)
    q = t.ppf(alpha, nu)
    es = -scale * (nu + q * q) / (nu - 1) * t.pdf(q, nu) / alpha
    return float(scale * q), float(es)


def forecast_one(history: pd.Series, distribution="normal", levels=(0.95, 0.99)) -> dict:
    fit = fit_garch(history, distribution)
    forecast = fit.forecast(horizon=1, reindex=False)
    mu = float(forecast.mean.iloc[-1, 0])
    variance = float(forecast.variance.iloc[-1, 0])
    if not np.isfinite([mu, variance]).all() or variance <= 0:
        raise RuntimeError("Некорректный прогноз дисперсии.")
    sigma = np.sqrt(variance)
    row = {"model": f"GARCH-{distribution}", "origin": history.index[-1], "train_n": len(history),
           "mu_pct": mu, "variance_pct2": variance, "sigma_pct": sigma,
           "omega": float(fit.params["omega"]), "alpha1": float(fit.params["alpha[1]"]),
           "beta1": float(fit.params["beta[1]"]), "convergence_flag": int(fit.convergence_flag),
           "nu": float(fit.params.get("nu", np.nan))}
    for level in levels:
        tag = str(int(round(level * 100)))
        q, es = tail_parameters(1 - level, distribution, row["nu"])
        row[f"q_{tag}_pct"] = mu + sigma * q
        row[f"var_{tag}_loss_pct"] = -row[f"q_{tag}_pct"]
        row[f"es_{tag}_loss_pct"] = -(mu + sigma * es)
    return row


def walk_forward(returns: pd.Series, test_size=500, distribution="normal", levels=(0.95, 0.99), progress=None) -> pd.DataFrame:
    validate_returns(returns)
    if not isinstance(test_size, int) or test_size <= 0 or len(returns) - test_size < 252:
        raise ValueError("Оставьте не менее 252 наблюдений для обучения и положительный тестовый период.")
    rows = []
    first = len(returns) - test_size
    for i in range(first, len(returns)):
        # Actual return is read only after forecast construction; no lookahead.
        row = forecast_one(returns.iloc[:i], distribution, levels)
        row.update(date=returns.index[i], actual_pct=float(returns.iloc[i]))
        rows.append(row)
        if progress and ((i - first + 1) % 50 == 0 or i == len(returns) - 1):
            progress(f"{distribution}: {i - first + 1}/{test_size}")
    return pd.DataFrame(rows).set_index("date")


def historical_forecasts(returns: pd.Series, test_size=500, levels=(0.95, 0.99)) -> pd.DataFrame:
    validate_returns(returns)
    if not isinstance(test_size, int) or test_size <= 0 or len(returns) - test_size < 252:
        raise ValueError("Недостаточно обучающих наблюдений или некорректный test_size.")
    rows = []
    for i in range(len(returns) - test_size, len(returns)):
        history = returns.iloc[:i]
        row = {"date": returns.index[i], "model": "Historical", "origin": history.index[-1],
               "train_n": len(history), "actual_pct": float(returns.iloc[i])}
        for level in levels:
            tag = str(int(round(level * 100)))
            q = float(history.quantile(1 - level))
            row[f"q_{tag}_pct"] = q
            row[f"var_{tag}_loss_pct"] = -q
            row[f"es_{tag}_loss_pct"] = float(-history[history <= q].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")
