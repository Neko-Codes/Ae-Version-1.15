"""Meaning search over long-term memory. Stdlib TF-IDF cosine. No extra deps."""
import math
import re

_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text):
    return _WORD.findall((text or "").lower())


def search(entries, query, limit=5):
    """entries: [{text, metadata}]. Returns top matches with score + method."""
    docs = [(e.get("text", "") or "") for e in entries]
    if not docs or not (query or "").strip():
        return []
    # Document frequency
    df = {}
    doc_tokens = []
    for d in docs:
        toks = set(_tokens(d))
        doc_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    qtoks = _tokens(query)
    if not qtoks:
        return []
    qtf = {}
    for t in qtoks:
        qtf[t] = qtf.get(t, 0) + 1
    qnorm = math.sqrt(sum((v * idf.get(t, math.log(n + 1) + 1.0)) ** 2 for t, v in qtf.items())) or 1.0
    scored = []
    for i, entry in enumerate(entries):
        toks = _tokens(docs[i])
        if not toks:
            continue
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dot = sum(qtf.get(t, 0) * tf.get(t, 0) * (idf.get(t, 0.0) ** 2) for t in qtf)
        dnorm = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in tf.items())) or 1.0
        score = dot / (qnorm * dnorm)
        # Small substring bonus for exact phrase recall
        if query.lower() in docs[i].lower():
            score += 0.05
        if score > 0:
            scored.append({"text": docs[i], "metadata": entry.get("metadata", {}),
                           "score": round(score, 4), "method": "tfidf"})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[: max(1, limit)]
