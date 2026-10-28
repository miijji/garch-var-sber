import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t

from garch_var.backtest import coverage_tests, summarize
from garch_var.data import log_returns, quality_report, validate_prices
from garch_var.model import forecast_one, historical_forecasts, tail_parameters, walk_forward


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    shocks = rng.standard_normal(330)
    values = np.empty(330)
    variance = 1.
    for i, shock in enumerate(shocks):
        values[i] = np.sqrt(variance) * shock
        variance = .05 + .1 * values[i] ** 2 + .85 * variance
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), name="return_pct")


def test_log_return_units_and_no_calendar_fill():
    frame = pd.DataFrame({"date": ["2022-02-25", "2022-03-24"], "close": [100., 110.]})
    prices = validate_prices(frame)
    actual = log_returns(prices)
    assert len(actual) == 1
    assert actual.iloc[0] == pytest.approx(100 * np.log(1.1))
    assert quality_report(prices)["gaps_over_7_calendar_days"][0]["gap_days"] == 27


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_reject_invalid_prices(bad):
    with pytest.raises(ValueError):
        validate_prices(pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "close": [100., bad]}))


def test_reject_duplicate_dates():
    with pytest.raises(ValueError, match="повторные"):
        validate_prices(pd.DataFrame({"date": ["2020-01-01"] * 2, "close": [100., 101.]}))


@pytest.mark.parametrize("alpha", [.01, .05])
def test_normal_es_matches_tail_integral(alpha):
    q, es = tail_parameters(alpha)
    integral = quad(lambda x: x * norm.pdf(x), -np.inf, q)[0] / alpha
    assert es == pytest.approx(integral)
    assert es < q < 0


@pytest.mark.parametrize("nu", [3., 8., 30.])
def test_student_es_and_standardization(nu):
    alpha = .05
    q, es = tail_parameters(alpha, "t", nu)
    scale = np.sqrt((nu - 2) / nu)
    assert t.cdf(q / scale, nu) == pytest.approx(alpha)
    integral = quad(lambda x: x * t.pdf(x / scale, nu) / scale, -np.inf, q)[0] / alpha
    assert es == pytest.approx(integral, rel=1e-6)
    assert es < q < 0


def test_kupiec_calibrated_frequency():
    result = coverage_tests([1] * 5 + [0] * 95, .05)
    assert result["kupiec_lr"] == pytest.approx(0)
    assert result["kupiec_p"] == pytest.approx(1)
    assert result["n11"] == 4
    assert result["n10"] == 1
    assert result["n00"] == 94


@pytest.mark.parametrize("hits", [[0] * 100, [1] * 100, [0]])
def test_degenerate_coverage(hits):
    result = coverage_tests(hits, .05)
    assert np.isfinite(result["kupiec_p"])
    assert np.isnan(result["independence_p"])


def test_independence_detects_clusters():
    hits = [0] * 90 + [1] * 10 + [0] * 90 + [1] * 10
    assert coverage_tests(hits, .1)["independence_p"] < .001


def test_garch_forecast_dates_units_and_no_lookahead(returns):
    base = walk_forward(returns, 3)
    changed = returns.copy()
    changed.iloc[-1] = -50.
    altered = walk_forward(changed, 3)
    pd.testing.assert_frame_equal(base.drop(columns="actual_pct"), altered.drop(columns="actual_pct"))
    assert (base.origin.to_numpy() < base.index.to_numpy()).all()
    assert base.index.equals(returns.index[-3:])
    assert (base.q_99_pct < base.q_95_pct).all()
    assert (base.es_99_loss_pct > base.var_99_loss_pct).all()
    np.testing.assert_allclose(base.sigma_pct**2, base.variance_pct2)
    np.testing.assert_allclose(base.q_95_pct, base.mu_pct + norm.ppf(.05) * base.sigma_pct)
    np.testing.assert_allclose(base.var_95_loss_pct, -base.q_95_pct)


def test_historical_forecast_excludes_target(returns):
    frame = historical_forecasts(returns, 3)
    assert frame.iloc[0].q_95_pct == pytest.approx(returns.iloc[:-3].quantile(.05))
    changed = returns.copy()
    changed.iloc[-1] = -1000
    pd.testing.assert_frame_equal(frame.drop(columns="actual_pct"), historical_forecasts(changed, 3).drop(columns="actual_pct"))


def test_student_forecast_and_multiple_model_summary(returns):
    row = forecast_one(returns.iloc[:-3], "t")
    assert row["nu"] > 2
    assert row["es_99_loss_pct"] > row["var_99_loss_pct"]
    base = walk_forward(returns, 3)
    historical = historical_forecasts(returns, 3)
    summary = summarize(pd.concat([base, historical]))
    assert len(summary) == 4
    assert (summary.n == 3).all()


@pytest.mark.parametrize("size", [0, -1, 1000])
def test_reject_invalid_split(returns, size):
    with pytest.raises(ValueError):
        walk_forward(returns, size)


def test_reject_missing_prediction(returns):
    frame = historical_forecasts(returns, 3)
    frame.iloc[0, frame.columns.get_loc("q_95_pct")] = np.nan
    with pytest.raises(ValueError, match="молча"):
        summarize(frame)
