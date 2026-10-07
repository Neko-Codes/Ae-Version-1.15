"""Budget tracker: token + cost accounting with soft limits. No extra deps."""
import os

# USD per 1M tokens (approx, conservative). Unknown models fall back to default.
PRICING = {
    "default": (1.0, 3.0),
    "codestral-latest": (0.3, 0.9),
    "mistral": (0.3, 0.9),
    "gpt-oss-120b": (0.15, 0.6),
    "llama-3.3-70b": (0.35, 0.4),
    "glm-5.3": (0.5, 1.0),
    "gemini-3.8-flash": (0.1, 0.4),
}


def _price_for(model):
    m = (model or "").lower()
    for key, price in PRICING.items():
        if key != "default" and key in m:
            return price
    return PRICING["default"]


class BudgetTracker:
    def __init__(self):
        try:
            self.max_tokens = int(os.getenv("AE_MAX_TOKENS", "0") or 0)
        except Exception:
            self.max_tokens = 0
        try:
            self.max_usd = float(os.getenv("AE_MAX_USD", "0") or 0)
        except Exception:
            self.max_usd = 0.0
        self.used_tokens = 0
        self.used_usd = 0.0

    def record(self, model, prompt_tokens=0, completion_tokens=0):
        total = int(prompt_tokens or 0) + int(completion_tokens or 0)
        pin, pout = _price_for(model)
        cost = (int(prompt_tokens or 0) / 1e6) * pin + (int(completion_tokens or 0) / 1e6) * pout
        self.used_tokens += total
        self.used_usd += cost
        return {"total": total, "cost_usd": round(cost, 6)}

    def check(self):
        """Return (ok, message). Soft gate checked before each LLM call."""
        if self.max_tokens and self.used_tokens >= self.max_tokens:
            return False, f"Token budget exceeded: {self.used_tokens}/{self.max_tokens}"
        if self.max_usd and self.used_usd >= self.max_usd:
            return False, f"Cost budget exceeded: ${self.used_usd:.4f}/${self.max_usd:.4f}"
        return True, ""

    def status(self):
        return {"used_tokens": self.used_tokens, "used_usd": round(self.used_usd, 6),
                "max_tokens": self.max_tokens, "max_usd": self.max_usd}
