"""Safe hands: path validation + dynamic command risk classification. No extra deps."""
import re
from pathlib import Path

# Critical patterns: always blocked, in every approval mode (no bypass except
# deleting the harness code itself).
CRITICAL_RE = re.compile(
    r"(rm\s+-[^\s]*r|rmdir\s+/s|del\s+/[fs]|format\s+[a-z]:|mkfs|dd\s+of=|"
    r"rd\s+/s|rm\s+--no-preserve-root|:\(\)\s*\{|"
    r"shutdown\s+/[sp]|halt\b|reboot\b|init\s+0|poweroff)",
    re.IGNORECASE,
)

# Weighted risk signals: (pattern, points, reason). Scored dynamically per command.
RISK_SIGNALS = [
    (re.compile(r"\bsudo\b|\brunas\b", re.I), 3, "elevation"),
    (re.compile(r"curl\s+.*\|\s*(sh|bash)|wget\s+.*\|\s*(sh|bash)", re.I), 5, "remote-code-pipe"),
    (re.compile(r"\bchmod\s+-[^\s]*r\b|\bchown\s+-[^\s]*r\b|\battrib\s+-[rs]", re.I), 2, "recursive-perms"),
    (re.compile(r">\s*\S|>>\s*\S", re.I), 1, "redirection"),
    (re.compile(r"\brm\b|\bdel\b|\brd\b|\brmdir\b|\bmv\b", re.I), 2, "delete/move"),
    (re.compile(r"\bapt(-get)?\b|\bbrew\b|\bchoco\b|\bwinget\b|\bpip\s+install\b", re.I), 1, "installer"),
    (re.compile(r"\breg\s+(add|delete)\b|\bregedit\b", re.I), 3, "registry"),
    (re.compile(r"\bnet\s+(user|localgroup)\b|\bnetsh\b", re.I), 3, "system-config"),
    (re.compile(r"\bpython\s+-c\b|\bpowershell\s+-[eE]nc?\b", re.I), 1, "inline-code"),
    (re.compile(r"&&|\|\||;", re.I), 1, "chained"),
]

LEVELS = [(8, "critical"), (5, "high"), (3, "moderate"), (1, "low"), (0, "minimal")]


def classify(command):
    """Score a shell command -> (level, reasons). Pure function, no side effects."""
    if not isinstance(command, str) or not command.strip():
        return "minimal", []
    if CRITICAL_RE.search(command):
        return "critical", ["blocklisted-destructive"]
    Score, Reasons = 0, []
    for Pattern, Points, Reason in RISK_SIGNALS:
        if Pattern.search(command):
            Score += Points
            Reasons.append(Reason)
    for Threshold, Level in LEVELS:
        if Score >= Threshold:
            return Level, Reasons
    return "minimal", Reasons


class Sandbox:
    def __init__(self, project_root):
        try:
            self.root = Path(project_root).resolve()
        except Exception:
            self.root = Path(".").resolve()

    def resolve(self, rel):
        """Resolve a workspace-relative path; raise if it escapes the root."""
        target = (self.root / (rel or ".")).resolve()
        try:
            target.relative_to(self.root)
        except Exception:
            raise ValueError(f"Path escapes workspace: {rel}")
        return target

    def check_command(self, command):
        """Raise on critical commands in ALL modes. Returns (level, reasons)."""
        level, reasons = classify(command)
        if level == "critical":
            raise ValueError(
                "Blocked destructive command (risk: critical"
                + (f" [{', '.join(reasons)}]" if reasons else "")
                + "). Use directory_manage/delete_entity for scoped deletes."
            )
        return level, reasons
