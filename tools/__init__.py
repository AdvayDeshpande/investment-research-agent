"""Tool registry: maps tool names to Python functions and their JSON schemas for the LLM."""
from tools import calculator, document_search, financial_data, news_search

TOOL_FUNCTIONS = {
    "get_stock_data": financial_data.get_stock_data,
    "search_news": news_search.search_news,
    "search_documents": document_search.search_documents,
    "calculate": calculator.calculate,
}

TOOL_SCHEMAS = [
    financial_data.SCHEMA,
    news_search.SCHEMA,
    document_search.SCHEMA,
    calculator.SCHEMA,
]

__all__ = ["TOOL_FUNCTIONS", "TOOL_SCHEMAS"]
