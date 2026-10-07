"""Diary: append-only JSONL trace of agent activity. Cheap, queryable, learn-from-mistakes."""
import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, base_path, enabled=True):
        self.enabled = enabled
        try:
            self.path = Path(base_path) / "trace.jsonl"
        except Exception:
            self.path = None

    def log(self, kind, **fields):
        if not self.enabled or self.path is None:
            return
        try:
            rec = {"ts": time.time(), "kind": kind, **fields}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def recent(self, limit=20, kind=None):
        if self.path is None or not self.path.is_file():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-max(1, limit * 3):]
        except Exception:
            return []
        out = []
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if kind and rec.get("kind") != kind:
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def last_errors(self, limit=5):
        recs = self.recent(limit=50)
        errs = [r for r in recs if r.get("is_error") or r.get("kind") in {"error", "tool_error"}]
        return errs[-limit:]
