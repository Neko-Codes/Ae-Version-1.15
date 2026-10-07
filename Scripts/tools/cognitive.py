"""Cognitive primitives: observable reasoning, self-critique, meaning search, capability discovery."""
import json


def register_cognitive_tools(registry):
    registry.Register(
        "reasoning_step",
        "Record one observable reasoning step (thought, alternatives, confidence). Shown in terminal; stored in trace diary.",
        {"type": "object", "properties": {
            "thought": {"type": "string"},
            "alternatives": {"type": "string"},
            "confidence": {"type": "number"},
            "next_action": {"type": "string"},
        }, "required": ["thought"], "additionalProperties": False},
        lambda thought, alternatives="", confidence=0.5, next_action="": _reasoning_step(registry, thought, alternatives, confidence, next_action),
    )
    registry.Register(
        "self_reflect",
        "Critique your last output against criteria before returning it. Returns verdict + issues.",
        {"type": "object", "properties": {
            "output": {"type": "string"},
            "criteria": {"type": "string"},
        }, "required": ["output", "criteria"], "additionalProperties": False},
        lambda output, criteria: _self_reflect(output, criteria),
    )
    registry.Register(
        "knowledge_query",
        "Search long-term memory by meaning (TF-IDF), not just keywords. Consolidated knowledge layer.",
        {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        }, "required": ["query"], "additionalProperties": False},
        lambda query, limit=5: registry._KnowledgeQuery(query, limit),
    )
    registry.Register(
        "capability_search",
        "Find tools/skills/executables by capability description at runtime.",
        {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        }, "required": ["query"], "additionalProperties": False},
        lambda query, limit=8: registry._CapabilitySearch(query, limit),
    )


def _reasoning_step(registry, thought, alternatives="", confidence=0.5, next_action=""):
    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except Exception:
        conf = 0.5
    rec = {"thought": thought, "alternatives": alternatives,
           "confidence": conf, "next_action": next_action,
           "agent": getattr(registry, "CurrentAgentName", "*")}
    try:
        registry.Tracer.log("reasoning_step", **rec)
    except Exception:
        pass
    return {"recorded": True, **rec}


def _self_reflect(output, criteria):
    crits = [c.strip() for c in (criteria or "").splitlines() if c.strip()]
    if not crits:
        crits = [criteria] if criteria else ["correct", "complete", "grounded"]
    issues = []
    text = output or ""
    for c in crits:
        cl = c.lower()
        if "citat" in cl or "source" in cl or "ground" in cl:
            if "http" not in text and "source" not in text.lower():
                issues.append(f"Criterion '{c}': no sources/citations visible.")
        elif "concis" in cl or "brief" in cl:
            if len(text.split()) > 400:
                issues.append(f"Criterion '{c}': output is long ({len(text.split())} words).")
        elif "complet" in cl:
            if text.strip().endswith(("...", "TODO", "TBD")):
                issues.append(f"Criterion '{c}': output looks unfinished.")
    verdict = "pass" if not issues else "revise"
    return {"verdict": verdict, "issues": issues, "criteria": crits}
