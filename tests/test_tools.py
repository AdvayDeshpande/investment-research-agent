"""Unit tests for all tools. External APIs are mocked - no network access needed."""
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

import config
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from tools.calculator import calculate
from tools.document_search import search_documents
from tools.financial_data import get_stock_data
from tools.news_search import search_news


# ---------------------------------------------------------------- registry
def test_registry_schemas_match_functions():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == set(TOOL_FUNCTIONS)


# ---------------------------------------------------------------- calculator
@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", 14),
    ("(230 - 200) / 200 * 100", 15.0),
    ("2 ^ 10", 1024),
    ("sqrt(16) + round(2.567, 1)", 6.6),
    ("-5 + abs(-3)", -2),
])
def test_calculate_valid(expr, expected):
    result = calculate(expr)
    assert result["success"] is True
    assert result["result"] == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')", "open('x')", "a + 1", "2 ** 99999", "", "1 +",
])
def test_calculate_rejects_unsafe_or_invalid(expr):
    assert calculate(expr)["success"] is False


def test_calculate_division_by_zero():
    result = calculate("1 / 0")
    assert result["success"] is False
    assert "zero" in result["error"].lower()


# ---------------------------------------------------------------- financial data
def _fake_yfinance(info, closes):
    ticker = MagicMock()
    ticker.info = info
    ticker.history.return_value = pd.DataFrame({"Close": closes})
    module = types.ModuleType("yfinance")
    module.Ticker = MagicMock(return_value=ticker)
    return module


def test_get_stock_data_success():
    fake = _fake_yfinance(
        {"longName": "Apple Inc.", "currency": "USD", "currentPrice": 110.0, "trailingPE": 30.5,
         "marketCap": 3_000_000_000_000, "fiftyTwoWeekLow": 90, "fiftyTwoWeekHigh": 120},
        [100.0, 105.0, 110.0],
    )
    with patch.dict(sys.modules, {"yfinance": fake}):
        result = get_stock_data("aapl", "1mo")
    assert result["success"] is True
    assert result["ticker"] == "AAPL"
    assert result["pe_ratio_trailing"] == 30.5
    assert result["period_return_pct"] == 10.0
    fake.Ticker.assert_called_once_with("AAPL")


def test_get_stock_data_unknown_ticker():
    fake = _fake_yfinance({}, [])
    with patch.dict(sys.modules, {"yfinance": fake}):
        result = get_stock_data("NOTREAL")
    assert result["success"] is False
    assert "No price data" in result["error"]


def test_get_stock_data_network_error():
    fake = types.ModuleType("yfinance")
    fake.Ticker = MagicMock(side_effect=ConnectionError("boom"))
    with patch.dict(sys.modules, {"yfinance": fake}):
        result = get_stock_data("AAPL")
    assert result["success"] is False
    assert "boom" in result["error"]


def test_get_stock_data_timeout(monkeypatch):
    import time

    def slow(*_args, **_kwargs):
        time.sleep(2)

    monkeypatch.setattr(config, "STOCK_DATA_TIMEOUT_SECONDS", 0.2)
    with patch("tools.financial_data._fetch", side_effect=slow):
        result = get_stock_data("AAPL")
    assert result["success"] is False
    assert "did not respond" in result["error"]


def test_get_stock_data_requires_ticker():
    assert get_stock_data("")["success"] is False


# ---------------------------------------------------------------- news search
@pytest.fixture
def news_key(monkeypatch):
    monkeypatch.setattr(config, "NEWS_API_KEY", "test-key")


def _response(status_code, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_search_news_success(news_key):
    payload = {"status": "ok", "totalResults": 1, "articles": [{
        "title": "Apple beats estimates", "source": {"name": "Reuters"},
        "publishedAt": "2026-11-19T10:00:00Z", "description": "Strong iPhone sales", "url": "https://x"}]}
    with patch("tools.news_search.requests.get", return_value=_response(200, payload)) as mock_get:
        result = search_news("Apple earnings", days_back=3)
    assert result["success"] is True
    assert result["articles"][0]["source"] == "Reuters"
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["X-Api-Key"] == "test-key"
    assert kwargs["timeout"] == config.HTTP_TIMEOUT_SECONDS
    assert kwargs["params"]["q"] == "Apple earnings"


def test_search_news_missing_key(monkeypatch):
    monkeypatch.setattr(config, "NEWS_API_KEY", "")
    result = search_news("Apple")
    assert result["success"] is False
    assert "NEWS_API_KEY" in result["error"]


def test_search_news_api_error(news_key):
    payload = {"status": "error", "message": "Your API key is invalid."}
    with patch("tools.news_search.requests.get", return_value=_response(401, payload)):
        result = search_news("Apple")
    assert result["success"] is False
    assert "invalid" in result["error"]


def test_search_news_timeout(news_key):
    with patch("tools.news_search.requests.get", side_effect=requests.Timeout()):
        result = search_news("Apple")
    assert result["success"] is False
    assert "did not respond" in result["error"]


# ---------------------------------------------------------------- document search
def test_search_documents_success():
    collection = MagicMock()
    collection.count.return_value = 10
    collection.query.return_value = {
        "documents": [["Revenue grew 8%."]],
        "metadatas": [[{"source": "report.pdf", "page": 12}]],
        "distances": [[0.25]],
    }
    model = MagicMock()
    model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
    with patch("tools.document_search.get_collection", return_value=collection), \
         patch("tools.document_search.get_embedding_model", return_value=model):
        result = search_documents("revenue growth", n_results=1)
    assert result["success"] is True
    assert result["results"][0] == {"text": "Revenue grew 8%.", "source": "report.pdf", "page": 12, "relevance": 0.75}


def test_search_documents_store_missing():
    with patch("tools.document_search.get_collection", side_effect=ValueError("no collection")):
        result = search_documents("anything")
    assert result["success"] is False
    assert "ingest" in result["error"]


def test_search_documents_empty_store():
    collection = MagicMock()
    collection.count.return_value = 0
    with patch("tools.document_search.get_collection", return_value=collection):
        result = search_documents("anything")
    assert result["success"] is False
    assert "empty" in result["error"]


# ---------------------------------------------------------------- ingestion chunking
def test_chunk_text_overlap():
    from ingestion.ingest import chunk_text

    class WordTokenizer:  # 1 word == 1 token, keeps the test fast and offline
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    text = " ".join(f"w{i}" for i in range(120))
    chunks = chunk_text(text, WordTokenizer(), chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # 10-token overlap
    assert chunks[-1].split()[-1] == "w119"
