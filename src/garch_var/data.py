"""Explicit price validation and immutable, attributed MOEX snapshots."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def validate_prices(frame: pd.DataFrame, date_column="date", price_column="close") -> pd.DataFrame:
    if date_column not in frame or price_column not in frame:
        raise ValueError(f"Нужны столбцы {date_column!r} и {price_column!r}.")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        raise ValueError("Даты должны быть заполнены и не содержать часового пояса.")
    dates = dates.dt.normalize()
    prices = pd.to_numeric(frame[price_column], errors="raise")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Цены должны быть положительными и конечными; пропуски не заполняются.")
    if dates.duplicated().any():
        raise ValueError("Найдены повторные даты. Выберите один актив и дневную частоту.")
    result = pd.DataFrame({"date": dates, "close": prices.astype(float)}).sort_values("date")
    if len(result) < 2:
        raise ValueError("Нужно не менее двух цен.")
    return result.set_index("date")


def read_prices(path: str | Path, date_column="date", price_column="close", sep=",") -> pd.DataFrame:
    return validate_prices(pd.read_csv(path, sep=sep), date_column, price_column)


def log_returns(prices: pd.DataFrame) -> pd.Series:
    # Validate even callers that bypass read_prices; one unit = one percentage point.
    clean = validate_prices(prices.reset_index())
    return (100 * np.log(clean.close).diff().dropna()).rename("return_pct")


def quality_report(prices: pd.DataFrame) -> dict:
    gaps = prices.index.to_series().diff().dt.days
    returns = log_returns(prices)
    return {
        "observations": len(prices), "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()), "missing_close": int(prices.close.isna().sum()),
        "duplicates": int(prices.index.duplicated().sum()),
        "gaps_over_7_calendar_days": [
            {"date": str(d.date()), "gap_days": int(gaps.loc[d])} for d in gaps[gaps > 7].index
        ],
        "absolute_log_returns_over_15_pct": {str(k.date()): float(v) for k, v in returns[returns.abs() > 15].items()},
        "policy": "Не заполняем неторговые дни; не удаляем экстремальные доходности. Большие интервалы требуют интерпретации.",
    }


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_moex(root: Path, ticker="SBER", start="2016-01-01", end="2025-12-31") -> tuple[Path, dict]:
    if not re.fullmatch(r"[A-Z0-9_-]+", ticker):
        raise ValueError("Недопустимый тикер.")
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("Начало периода позже конца.")
    start, end = str(start_date.date()), str(end_date.date())
    # TQBR is a shares board; other markets should be imported explicitly via CSV.
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
    folder = Path(root) / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker}_{start}_{end}"
    path, manifest_path = folder / f"{stem}.csv", folder / f"{stem}.metadata.json"
    if path.exists() or manifest_path.exists():
        if not (path.exists() and manifest_path.exists()):
            raise ValueError("Неполный снимок данных. Используйте другой период или проверьте файлы.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provider_path = folder / f"{stem}.provider.json"
        if (sha256(path) != manifest["sha256"] or not provider_path.exists()
                or sha256(provider_path) != manifest["provider_sha256"]):
            raise ValueError("Контрольная сумма снимка не совпадает.")
        read_prices(path)
        return path, manifest
    rows, offset, pages, urls = [], 0, [], []
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        while True:
            response = session.get(url, params={"from": start, "till": end, "interval": 24, "start": offset, "iss.meta": "off"}, timeout=40)
            response.raise_for_status()
            payload = response.json()
            if "candles" not in payload:
                raise ValueError("ISS не вернул таблицу candles.")
            table = payload["candles"]
            if not table["data"]:
                break
            pages.append(payload)
            urls.append(response.url)
            rows.extend(dict(zip(table["columns"], row)) for row in table["data"])
            offset += len(table["data"])
            if len(pages) > 200:
                raise ValueError("Превышен предел страниц ISS.")
    if not rows:
        raise ValueError("Нет котировок для выбранного тикера и периода.")
    raw = pd.DataFrame(rows)
    prices = validate_prices(raw, "begin", "close")
    if prices.index.min() < start_date or prices.index.max() > end_date:
        raise ValueError("Источник вернул даты вне запрошенного периода.")
    prices.to_csv(path, date_format="%Y-%m-%d")
    provider_path = folder / f"{stem}.provider.json"
    provider_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "source": "MOEX ISS", "url": url, "requests": urls, "ticker": ticker, "board": "TQBR",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "requested_start": start, "requested_end": end,
        "price_field": "close", "date_field": "begin", "interval": 24, "currency": "RUB",
        "adjustment": "Не скорректировано на дивиденды и корпоративные события.",
        "transformation": "begin -> дата без времени; close без изменений; сортировка по дате.",
        "sha256": sha256(path), "provider_sha256": sha256(provider_path),
        "usage": "Проверить условия MOEX перед распространением котировок; снимки исключены из Git.",
        "quality": quality_report(prices),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, manifest
