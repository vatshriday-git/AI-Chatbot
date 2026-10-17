# AI-Powered Support Chatbot

FastAPI + Transformers (MiniLM embeddings) + NLTK + SQLite.

## Run
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export ADMIN_TOKEN=my-secret
uvicorn app:app --reload
```
Open http://localhost:8000 (first start downloads models + NLTK data).

## How it works
1. NLTK tokenizes/lemmatizes input; small talk is handled by regex.
2. A Transformer encodes the query; cosine similarity vs. FAQ questions/variants,
   blended with keyword overlap, picks the best answer.
3. **Context**: short/pronoun follow-ups ("what about international?") are merged with the
   previous user message and used if they score better.
4. Confidence tiers: answer / "did you mean…" suggestions / fallback + human escalation.
5. Optional sentiment model flags upset users for escalation (`ENABLE_SENTIMENT=0` to disable).
6. Every message is logged to SQLite (`chatbot.db`) with intent, confidence, latency, feedback.

## API
| Endpoint | Description |
|---|---|
| `POST /api/chat` `{session_id?, message}` | Get a reply |
| `GET /api/history/{session_id}` | Conversation history |
| `POST /api/feedback` `{message_id, helpful}` | Thumbs up/down |
| `GET /api/logs`, `GET /api/stats` | Admin only: header `X-Admin-Token` |

```bash
curl -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
     -d '{"message":"where is my package?"}'
curl localhost:8000/api/stats -H 'X-Admin-Token: my-secret'
```

## Customize
Edit `faqs.json` (add `variants` for better matching), tune `HIGH_CONF` / `LOW_CONF` env vars,
or swap `EMBED_MODEL`. Docker: `docker build -t chatbot . && docker run -p 8000:8000 chatbot`.
