import json

import pytest

from garch_var.data import download_moex, read_prices


def test_moex_pagination_cache_and_checksum(tmp_path, monkeypatch):
    calls = []
    columns = ["begin", "close"]
    pages = [
        {"candles": {"columns": columns, "data": [["2020-01-03 00:00:00", 101.], ["2020-01-06 00:00:00", 103.]]}},
        {"candles": {"columns": columns, "data": [["2020-01-07 00:00:00", 102.]]}},
        {"candles": {"columns": columns, "data": []}},
    ]

    class Response:
        url = "https://iss.moex.com/example"

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def mount(self, *args):
            pass

        def get(self, url, params, timeout):
            calls.append(params["start"])
            return Response(pages[len(calls) - 1])

    monkeypatch.setattr("garch_var.data.requests.Session", Session)
    path, manifest = download_moex(tmp_path, "SBER", "2020-01-01", "2020-01-10")
    assert calls == [0, 2, 3]
    assert len(read_prices(path)) == 3
    assert manifest["source"] == "MOEX ISS"
    assert manifest["price_field"] == "close"
    again, cached = download_moex(tmp_path, "SBER", "2020-01-01", "2020-01-10")
    assert again == path and cached == manifest
    assert calls == [0, 2, 3]  # offline cache: no extra request
    provider = path.with_name(path.stem + ".provider.json")
    original = provider.read_bytes()
    provider.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="сумма"):
        download_moex(tmp_path, "SBER", "2020-01-01", "2020-01-10")
    provider.write_bytes(original)
    path.write_text(path.read_text().replace("101.0", "999.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="сумма"):
        download_moex(tmp_path, "SBER", "2020-01-01", "2020-01-10")


def test_reject_invalid_ticker_without_network(tmp_path):
    with pytest.raises(ValueError):
        download_moex(tmp_path, "../SBER")


def test_reject_incomplete_snapshot(tmp_path):
    folder = tmp_path / "data/raw"
    folder.mkdir(parents=True)
    (folder / "SBER_2020-01-01_2020-01-10.metadata.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="Неполный"):
        download_moex(tmp_path, "SBER", "2020-01-01", "2020-01-10")
