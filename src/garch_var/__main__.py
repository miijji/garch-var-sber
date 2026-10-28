"""Run with python -m garch_var --help."""
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import backtest_figure, conclusions, correlation_figure, eda_figure, residual_diagnostics, stationarity
from .backtest import summarize
from .data import download_moex, log_returns, quality_report, read_prices, sha256
from .model import fit_garch, forecast_one, historical_forecasts, walk_forward


def main():
    parser = argparse.ArgumentParser(description="GARCH(1,1), VaR и бэктестинг. Доходности в процентных пунктах.")
    parser.add_argument("--csv", type=Path, help="Локальный дневной CSV, например с Kaggle; без этого параметра — MOEX.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--price-column", default="close")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--source-url", help="Источник импортируемого CSV (для воспроизводимости).")
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--distribution", choices=["normal", "t", "both"], default="both")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Каталог результатов и кэша данных.")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.csv:
        path = args.csv.resolve()
        if not args.source_url:
            parser.error("Для CSV укажите --source-url: ссылку на датасет или описание происхождения.")
        prices = read_prices(path, args.date_column, args.price_column, args.sep)
        prices = prices.loc[args.start:args.end]
        manifest = {"source": args.source_url, "input_file": path.name, "sha256": sha256(path),
                    "date_column": args.date_column, "price_column": args.price_column,
                    "separator": args.sep, "adjustment": "Определяется источником CSV; проверить перед интерпретацией."}
    else:
        path, manifest = download_moex(root, args.ticker, args.start, args.end)
        prices = read_prices(path)
    returns = log_returns(prices)
    if args.test_size <= 0 or len(returns) - args.test_size < 252:
        parser.error("Нужно оставить не менее 252 доходностей для обучения.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "outputs" / "runs" / stamp
    out.mkdir(parents=True, exist_ok=False)
    train = returns.iloc[:-args.test_size]
    manifest["quality"] = quality_report(prices)
    manifest["python"] = platform.python_version()
    manifest["versions"] = {name: importlib.metadata.version(name) for name in ["arch", "numpy", "pandas", "scipy", "statsmodels", "matplotlib"]}
    settings = vars(args).copy()
    settings.update(created_utc=stamp, return_unit="percentage points", window="expanding", refit="every observation", train_end=str(train.index[-1].date()))
    (out / "settings.json").write_text(json.dumps(settings, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    returns.to_csv(out / "returns.csv")
    stationarity(train).to_csv(out / "stationarity.csv", index=False)
    eda_figure(prices.loc[:train.index[-1]], train).savefig(out / "eda.png", dpi=140)
    correlation_figure(train).savefig(out / "correlations.png", dpi=140)
    models = ["normal", "t"] if args.distribution == "both" else [args.distribution]
    forecasts = []
    for distribution in models:
        fit = fit_garch(train, distribution)
        (out / f"initial_fit_{distribution}.txt").write_text(str(fit.summary()), encoding="utf-8")
        residual_diagnostics(fit).to_csv(out / f"residual_diagnostics_{distribution}.csv", index=False)
        forecasts.append(walk_forward(returns, args.test_size, distribution, progress=lambda message: print(message, flush=True)))
    forecasts.append(historical_forecasts(returns, args.test_size))
    predictions = pd.concat(forecasts)
    predictions.to_csv(out / "predictions.csv")
    summary = summarize(predictions)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    for model in predictions.model.unique():
        backtest_figure(predictions, model).savefig(out / f"backtest_{model}.png", dpi=140)
    next_rows = [forecast_one(returns, distribution) for distribution in models]
    pd.DataFrame(next_rows).to_csv(out / "next_observation_forecast.csv", index=False)
    report = conclusions(summary)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    plt.close("all")
    print(summary[["model", "confidence", "n", "exceptions", "expected", "kupiec_p"]].to_string(index=False))
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()
