"""Financial news tool backed by the NewsAPI REST API (https://newsapi.org)."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import requests

import config

logger = logging.getLogger(__name__)

MAX_ARTICLES = 5


def search_news(query: str, days_back: int = 7) -> Dict[str, Any]:
    """Search recent news articles via NewsAPI /v2/everything."""
    logger.info("search_news(query=%r, days_back=%r)", query, days_back)
    if not query or not isinstance(query, str):
        return {"success": False, "error": "A search query is required."}
    if not config.NEWS_API_KEY:
        return {"success": False, "error": "NEWS_API_KEY is not set. Add it to your .env file (free at newsapi.org)."}

    try:
        days_back = max(1, min(int(days_back), 30))  # free tier only covers ~1 month
    except (TypeError, ValueError):
        days_back = 7
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": MAX_ARTICLES,
    }
    headers = {"X-Api-Key": config.NEWS_API_KEY}

    try:
        response = requests.get(config.NEWS_API_URL, params=params, headers=headers,
                                timeout=config.HTTP_TIMEOUT_SECONDS)
    except requests.Timeout:
        logger.error("NewsAPI request timed out")
        return {"success": False, "error": f"NewsAPI did not respond within {config.HTTP_TIMEOUT_SECONDS}s."}
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed: %s", exc)
        return {"success": False, "error": f"Network error calling NewsAPI: {exc}"}

    try:
        payload = response.json()
    except ValueError:
        return {"success": False, "error": f"NewsAPI returned non-JSON response (HTTP {response.status_code})."}

    if response.status_code != 200 or payload.get("status") != "ok":
        message = payload.get("message", "unknown error")
        logger.error("NewsAPI error %s: %s", response.status_code, message)
        return {"success": False, "error": f"NewsAPI error (HTTP {response.status_code}): {message}"}

    articles = [
        {
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "published_at": a.get("publishedAt"),
            "description": a.get("description"),
            "url": a.get("url"),
        }
        for a in payload.get("articles", [])[:MAX_ARTICLES]
    ]
    return {
        "success": True,
        "query": query,
        "from_date": from_date,
        "total_results": payload.get("totalResults", 0),
        "articles": articles,
    }


SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_news",
        "description": "Search recent English-language news articles (headlines + summaries) about a company, "
                       "sector or market topic. Use for current events and sentiment.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, e.g. 'Nvidia earnings' or 'ECB interest rates'"},
                "days_back": {"type": "integer", "description": "How many days back to search (1-30). Default 7."},
            },
            "required": ["query"],
        },
    },
}
