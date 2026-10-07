"""Real reasoning display: prefer provider reasoning_content, fall back to content."""
import re


def extract_reasoning(message):
    """Return (reasoning_text, answer_text). Works with OpenAI-style
    reasoning models (reasoning_content field) and plain models."""
    if not isinstance(message, dict):
        return "", str(message or "")
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )
    content = message.get("content") or ""
    # Some providers embed <think>...</think> blocks in content
    if not reasoning and "<think>" in content:
        m = re.search(r"<think>(.*?)</think>", content, re.S)
        if m:
            reasoning = m.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return (reasoning or "").strip(), content


def render_reasoning(reasoning, limit=2000):
    text = (reasoning or "").strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[:limit] + "\n[…reasoning truncated]"
    return text
