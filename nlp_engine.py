"""NLP engine: NLTK preprocessing + Transformer sentence embeddings for FAQ retrieval,
conversation-context handling, small talk, and optional sentiment-based escalation."""
import json
import os
import re

import nltk
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer, pipeline

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "1") == "1"
HIGH_CONF = float(os.getenv("HIGH_CONF", "0.60"))   # answer directly
LOW_CONF = float(os.getenv("LOW_CONF", "0.42"))     # below this -> fallback

SMALL_TALK = [
    (re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|howdy)\b", re.I), "greeting",
     "Hello! 👋 I'm your support assistant. Ask me about orders, shipping, returns, payments or your account."),
    (re.compile(r"\b(thanks|thank you|thx|cheers)\b", re.I), "thanks",
     "You're welcome! Is there anything else I can help with?"),
    (re.compile(r"^\s*(bye|goodbye|see you|that'?s all)\b", re.I), "goodbye",
     "Goodbye! Have a great day. 👋"),
    (re.compile(r"\b(who are you|what can you do|help me)\b", re.I), "about",
     "I'm an AI support assistant. I can answer questions about orders, shipping, returns, payments and accounts."),
]
FOLLOWUP_RE = re.compile(r"^\s*(and|but|also|then|what about|how about)\b|\b(it|that|this|those|them|they)\b", re.I)


def _ensure_nltk():
    for res, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


class ChatEngine:
    def __init__(self, faq_path: str = "faqs.json"):
        _ensure_nltk()
        self.stop = set(stopwords.words("english")) - {"not", "no"}
        self.lemmatizer = WordNetLemmatizer()

        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).eval()

        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.faq_by_id = {f["id"]: f for f in self.faqs}

        self.rows, self.row_faq_idx, self.row_tokens = [], [], []
        for i, faq in enumerate(self.faqs):
            for q in [faq["question"], *faq.get("variants", [])]:
                self.rows.append(q)
                self.row_faq_idx.append(i)
                self.row_tokens.append(set(self.content_tokens(q)))
        self.matrix = self.embed(self.rows)

        self._sentiment = None  # lazy-loaded

    # ---------- NLP helpers ----------
    def content_tokens(self, text: str):
        toks = word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(t) for t in toks if t.isalnum() and t not in self.stop]

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 32):
            batch = self.tokenizer(texts[i:i + 32], padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(pooled, dim=1))
        return torch.cat(out)

    def is_followup(self, msg: str) -> bool:
        n_content = len(self.content_tokens(msg))
        return n_content <= 2 or (bool(FOLLOWUP_RE.search(msg)) and len(msg.split()) <= 8)

    def sentiment(self, text: str):
        if not ENABLE_SENTIMENT:
            return None, 0.0
        try:
            if self._sentiment is None:
                self._sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
            r = self._sentiment(text[:512])[0]
            return r["label"].lower(), float(r["score"])
        except Exception:
            return None, 0.0

    # ---------- retrieval ----------
    def rank(self, query: str):
        """Return FAQs ranked by hybrid score (semantic cosine + keyword overlap)."""
        q_vec = self.embed([query])
        sims = (self.matrix @ q_vec.T).squeeze(1).tolist()
        q_tokens = set(self.content_tokens(query))
        best = {}
        for row, sim in enumerate(sims):
            tokens = self.row_tokens[row]
            union = q_tokens | tokens
            overlap = len(q_tokens & tokens) / len(union) if union else 0.0
            score = 0.85 * sim + 0.15 * overlap
            idx = self.row_faq_idx[row]
            if idx not in best or score > best[idx]:
                best[idx] = score
        return sorted(((self.faqs[i], s) for i, s in best.items()), key=lambda x: -x[1])

    # ---------- main entry ----------
    def respond(self, message: str, history: list | None = None) -> dict:
        history = history or []
        message = message.strip()

        for pattern, intent, reply in SMALL_TALK:
            if pattern.search(message) and len(message.split()) <= 6:
                return {"reply": reply, "intent": intent, "matched_faq": None,
                        "confidence": 1.0, "suggestions": [], "escalate": False, "sentiment": None}

        ranked = self.rank(message)
        used_context = False
        if self.is_followup(message):
            prev_users = [m["text"] for m in history if m["role"] == "user"]
            if prev_users:
                ctx_ranked = self.rank(prev_users[-1] + " " + message)
                if ctx_ranked[0][1] > ranked[0][1]:
                    ranked, used_context = ctx_ranked, True

        top_faq, top_score = ranked[0]
        sent_label, sent_score = self.sentiment(message)
        upset = sent_label == "negative" and sent_score > 0.95

        if top_score >= HIGH_CONF:
            reply = top_faq["answer"]
            if upset:
                reply = "I'm sorry you're having a frustrating experience. " + reply
            return {"reply": reply, "intent": top_faq["category"], "matched_faq": top_faq["id"],
                    "confidence": round(top_score, 3), "suggestions": [], "escalate": upset,
                    "sentiment": sent_label, "used_context": used_context}

        if top_score >= LOW_CONF:
            sugg = [f["question"] for f, s in ranked[:3] if s >= LOW_CONF - 0.1]
            return {"reply": "I'm not fully sure I understood. Did you mean one of these?",
                    "intent": "clarify", "matched_faq": None, "confidence": round(top_score, 3),
                    "suggestions": sugg, "escalate": upset, "sentiment": sent_label}

        return {"reply": "Sorry, I couldn't find an answer to that. I can connect you with a human agent: "
                         "email support@example.com or call 1-800-555-0199 (Mon-Fri 9am-6pm).",
                "intent": "fallback", "matched_faq": None, "confidence": round(top_score, 3),
                "suggestions": [f["question"] for f, _ in ranked[:3]], "escalate": True,
                "sentiment": sent_label}
