# Investment Research Agent

An agentic AI system that answers investment research questions with a **ReAct loop (Reason + Act)**.
The LLM is reached through the **Groq API** using native function calling. It decides on its own which external
system to query: live market data, news, or the internal document library. It can make several calls in a row
before it writes a cited answer.

> Built to show one skill in particular: **integrating AI models, AI services, data sources and external systems via APIs and other interfaces.**

## Architecture

```
                         +-----------------------------+
   User question  -----> |  main.py (CLI)              |
                         +--------------+--------------+
                                        v
                         +-----------------------------+      HTTPS / JSON (OpenAI-compatible)
                         |  ResearchAgent (ReAct loop) | <--------------------------------------> Groq API
                         |  agent/agent.py             |      tools=[JSON schemas], tool_choice=auto  (Llama 3.3 70B)
                         +--------------+--------------+
                    Thought -> Action -> Observation -> ... -> Final answer
                                        |
        +------------------+------------+-------------+--------------------+
        v                  v                          v                    v
 get_stock_data       search_news               search_documents        calculate
 (yfinance ->         (REST: NewsAPI            (local RAG:             (safe AST
  Yahoo Finance)       /v2/everything)           ChromaDB + MiniLM)      evaluator)
                                                      ^
                                    ingestion/ingest.py: PDF -> pages -> 500-token chunks
                                    (50 overlap) -> sentence-transformers -> ChromaDB
```

| Integration | Interface | Auth | What it shows |
|---|---|---|---|
| Groq LLM | REST via the official SDK, function calling | API key (`.env`) | Using a hosted AI service, timeouts, retries, error mapping |
| Yahoo Finance | `yfinance` Python library | none | Wrapping a third-party data source, thread-based timeout |
| NewsAPI | Raw HTTP REST (`requests`) | `X-Api-Key` header | Building HTTP requests, status codes, JSON parsing, timeouts |
| ChromaDB + sentence-transformers | Local embedded DB plus a local ML model | none | RAG over internal documents |

## Tools

| Tool | Purpose | Returns |
|---|---|---|
| `get_stock_data(ticker, period)` | Price, P/E (trailing/forward), market cap, 52-week range, dividend yield, period return | dict |
| `search_news(query, days_back)` | Top 5 relevant English news articles (from the last 1–30 days) | dict |
| `search_documents(query, n_results)` | Semantic search over ingested PDFs, with source file and page | dict |
| `calculate(expression)` | Safe arithmetic (no `eval`) so the LLM doesn't have to do math in its head | dict |

Every tool returns a dict like `{"success": bool, ...}` or `{"success": false, "error": "..."}`. Tools never raise
exceptions into the agent. The error goes back to the LLM as an observation, so the model can adapt by retrying
with a different ticker or explaining what's missing.

## Setup (Windows, PowerShell)

```powershell
git clone https://github.com/<your-user>/investment-research-agent.git
cd investment-research-agent
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env               # then edit .env and add your keys
```

1. **Groq key (free, no credit card):** https://console.groq.com/keys → put it in `GROQ_API_KEY`
2. **NewsAPI key (free, 100 req/day):** https://newsapi.org/register → put it in `NEWS_API_KEY`
3. **Build the document index** (the first run downloads the ~90 MB embedding model):
   ```powershell
   python ingestion/ingest.py data/documents/
   ```
   A sample public PDF is already included: the Berkshire Hathaway 2023 shareholder letter. You can drop more PDFs
   into `data/documents/` and re-run the script. Re-running is idempotent and won't create duplicates.

## Usage

```powershell
python main.py "What is the current P/E ratio of Apple and how does it compare to Microsoft?"
python main.py                 # interactive chat
python main.py -v "..."        # DEBUG: also prints the raw JSON returned by every tool
```

The console log shows each ReAct step (`ITERATION`, `THOUGHT`, `ACTION`, `OBSERVATION`, `FINAL ANSWER`).

### Example queries
1. `What is the P/E ratio of Apple?` (1 tool call)
2. `Compare the 6-month performance of NVDA and AMD and calculate the difference in percentage points.` (stock data ×2, then calculate)
3. `What are the latest news on SAP and how has SAP.DE traded over the last month?` (news + stock data)
4. `What does Warren Buffett say about share buybacks in the Berkshire shareholder letter?` (document RAG)
5. `Summarise Berkshire's view on its Japanese investments and give me the current price of BRK-B.` (RAG + live data)

## Tests

```powershell
pytest -q
```
The tests cover all four tools and the agent loop: multi-step reasoning, tool errors, unknown tools, malformed JSON
arguments, the max-steps limit, and LLM failures. Every external API is mocked, so the tests make no network calls
and need no keys.

## Why this architecture? (ReAct vs. chain-of-thought)

- **Chain-of-thought** = the model reasons inside one call, using only what it already knows. It can't check
  today's share price or read your internal documents, so it ends up hallucinating numbers.
- **ReAct** interleaves *reasoning* with *actions*. The model writes a thought ("I need Apple's P/E"), picks a tool,
  reads the real observation, and then reasons again. Answers are grounded in live data and cited sources.
- **Native function calling** (JSON schemas) replaces fragile text parsing of `Action: ...` lines. The API returns
  structured `tool_calls` with arguments that have already been validated against the schema.
- **Robustness by design:** a step budget (`AGENT_MAX_STEPS`) stops endless loops. Every network call has a timeout
  (Groq 30 s + 2 retries, NewsAPI 10 s, Yahoo 15 s). Tool errors become observations instead of crashes.
- **Extensible:** to add a new data source, you write one function plus one JSON schema and register it in
  `tools/__init__.py`. The agent loop doesn't change. That's the same pattern as MCP servers or OpenAI Assistants tools.

## Project structure
```
agent/        ReAct loop (agent.py) + Groq API wrapper (llm_client.py)
tools/        financial_data, news_search, document_search, calculator + registry
ingestion/    PDF -> ChromaDB pipeline
tests/        pytest suite (mocked APIs)
docs/         architecture.md (German interview notes)
config.py     all settings, overridable via .env
```

*Disclaimer: research support only, not investment advice.*
