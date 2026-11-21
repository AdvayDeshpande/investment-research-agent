"""Stock market data tool backed by Yahoo Finance (yfinance, no API key needed)."""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict

import config

logger = logging.getLogger(__name__)

VALID_PERIODS = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"}


def _fetch(ticker: str, period: str) -> Dict[str, Any]:
    import yfinance as yf  # imported lazily so tests / other tools don't pay the import cost

    stock = yf.Ticker(ticker)
    info = stock.info or {}
    history = stock.history(period=period)
    if history is None or history.empty:
        raise LookupError(f"No price data found for ticker '{ticker}'. Is the symbol correct?")

    first_close = float(history["Close"].iloc[0])
    last_close = float(history["Close"].iloc[-1])
    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName"),
        "currency": info.get("currency"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or round(last_close, 2),
        "pe_ratio_trailing": info.get("trailingPE"),
        "pe_ratio_forward": info.get("forwardPE"),
        "market_cap": info.get("marketCap"),
        "dividend_yield": info.get("dividendYield"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "sector": info.get("sector"),
        "period": period,
        "period_start_close": round(first_close, 2),
        "period_end_close": round(last_close, 2),
        "period_return_pct": round((last_close - first_close) / first_close * 100, 2),
    }


def get_stock_data(ticker: str, period: str = "1mo") -> Dict[str, Any]:
    """Return price, valuation and performance data for a stock ticker."""
    logger.info("get_stock_data(ticker=%r, period=%r)", ticker, period)
    if not ticker or not isinstance(ticker, str):
        return {"success": False, "error": "A ticker symbol (e.g. 'AAPL') is required."}
    ticker = ticker.strip().upper()
    if period not in VALID_PERIODS:
        logger.warning("Invalid period %r, falling back to '1mo'", period)
        period = "1mo"

    # yfinance has no global timeout, so we enforce one with a worker thread.
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch, ticker, period)
    try:
        data = future.result(timeout=config.STOCK_DATA_TIMEOUT_SECONDS)
        return {"success": True, **data}
    except FutureTimeout:
        logger.error("Yahoo Finance timed out for %s", ticker)
        return {"success": False, "error": f"Yahoo Finance did not respond within {config.STOCK_DATA_TIMEOUT_SECONDS}s."}
    except LookupError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # network errors, rate limits, parsing issues
        logger.exception("Failed to fetch stock data for %s", ticker)
        return {"success": False, "error": f"Could not fetch stock data: {exc}"}
    finally:
        executor.shutdown(wait=False)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_stock_data",
        "description": "Get live market data for a stock: current price, P/E ratio, market cap, "
                       "52-week range, dividend yield and price return over a period.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'SAP.DE'"},
                "period": {
                    "type": "string",
                    "enum": sorted(VALID_PERIODS),
                    "description": "Lookback window for the price return. Default '1mo'.",
                },
            },
            "required": ["ticker"],
        },
    },
}
