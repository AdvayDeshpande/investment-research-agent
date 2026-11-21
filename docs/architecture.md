# Architektur – Investment Research Agent (Interview-Vorbereitung)

## 1. Worum geht es?
Ein KI-Agent beantwortet Fragen zur Investmentrecherche. Er beantwortet sie nicht aus dem „Gedächtnis“ des Modells,
sondern holt sich die Fakten selbst über **APIs und Schnittstellen**:

| Komponente | Schnittstelle | Zweck |
|---|---|---|
| Groq (Llama 3.3 70B) | REST-API über das offizielle SDK, **Function Calling** | Das „Gehirn“: plant, wählt Tools aus und formuliert die Antwort |
| Yahoo Finance | Python-Bibliothek `yfinance` | Live-Kurse, KGV, Marktkapitalisierung, 52-Wochen-Spanne |
| NewsAPI | HTTP-REST (`requests`), API-Key im Header `X-Api-Key` | Aktuelle Finanznachrichten |
| ChromaDB + sentence-transformers | Lokale Vektordatenbank und lokales Embedding-Modell | RAG über interne PDFs (z. B. Geschäftsberichte) |
| Rechner | Sicherer AST-Parser | Präzise Berechnungen ohne `eval()` |

**Bezug zur Stelle:** *„Du integrierst KI-Modelle, KI-Dienste, Datenquellen und externe Systeme über APIs und andere Schnittstellen“*.
Genau das macht das Projekt: Es verbindet einen KI-Dienst (Groq), zwei externe Datenquellen (Yahoo, NewsAPI) und
eine interne Wissensquelle (Vektordatenbank) zu einem einzigen Agenten.

## 2. Das ReAct-Muster (Reason + Act)
```
Frage -> [LLM: Gedanke] -> Aktion (Tool-Aufruf) -> Beobachtung (Tool-Ergebnis)
      -> [LLM: neuer Gedanke] -> ... -> finale Antwort mit Quellen
```
1. Die Frage geht zusammen mit den **JSON-Schemas** aller Tools an die Groq-API (`tools=[...]`, `tool_choice="auto"`).
2. Das Modell antwortet entweder mit Text (dann ist das die **finale Antwort**) oder mit strukturierten `tool_calls`.
3. Der Agent führt die Tools in Python aus und hängt die Ergebnisse als `role="tool"`-Nachrichten an
   (verknüpft über die `tool_call_id`).
4. Dann folgt der nächste LLM-Aufruf. Das Modell kann weitere Tools aufrufen, zum Beispiel erst Kurse abfragen und dann rechnen.
5. Ein **Schritt-Limit** (`AGENT_MAX_STEPS`) verhindert Endlosschleifen. Ist es erreicht, wird eine Antwort ohne Tools erzwungen.

**ReAct vs. Chain-of-Thought:** Bei Chain-of-Thought denkt das Modell nur intern nach und kennt keine aktuellen Daten.
Dadurch drohen Halluzinationen. ReAct verankert jeden Denkschritt in echten Beobachtungen. Das ist nachvollziehbar
(jeder Schritt wird geloggt) und lässt sich prüfen (Quellenangaben).

## 3. Robustheit (wichtig in der Finanzbranche)
- **Timeouts überall:** Groq 30 s mit 2 Retries, NewsAPI 10 s, Yahoo 15 s (über einen Thread, weil `yfinance` kein eigenes Timeout hat).
- **Fehler als Beobachtung:** Tools werfen keine Exceptions, sondern geben `{"success": false, "error": ...}` zurück.
  Das LLM liest den Fehler und reagiert darauf, etwa mit einem anderen Ticker oder einem Hinweis auf die Einschränkung.
- **Abgesichert gegen LLM-Fehler:** unbekannte Tools, kaputtes JSON in den Argumenten, falsche Parameter, Rate-Limits.
- **Secrets** liegen nur in `.env`, die nicht ins Git-Repository kommt. Für jede API gibt es eine Fehlermeldung, die man versteht.
- **Tests:** pytest mit Mocks. Es werden keine echten API-Calls gemacht, deshalb laufen die Tests deterministisch und CI-tauglich.

## 4. RAG-Pipeline (`ingestion/ingest.py`)
PDF → Text pro Seite (pypdf) → Chunks à 500 Tokens mit 50 Tokens Überlappung (Tokenizer des Embedding-Modells) →
Embeddings (`all-MiniLM-L6-v2`, normalisiert) → ChromaDB (Kosinus-Ähnlichkeit). Metadaten: Datei und Seite, damit
die Quelle zitiert werden kann. **Idempotent:** Beim erneuten Einlesen werden alle Chunks einer Datei vorher gelöscht.

## 5. Mögliche Erweiterungen (gute Interview-Antworten)
- Weitere Datenquellen als Tool: interne SQL-Datenbank, Bloomberg/Refinitiv, EZB-Zinsdaten (SDW-API)
- Bereitstellung als REST-Service mit **FastAPI** oder als **MCP-Server**, damit andere Systeme den Agenten nutzen können
- Lokales LLM (Ollama) statt Cloud, wenn sensible Daten im Spiel sind (DSGVO, Vertraulichkeit)
- Evaluierung (Antwortqualität, Halluzinationsrate), Caching, Monitoring der Token-Kosten
