"""AE ToolRegistry: thin Python syscall layer.

Design rule (see AutomatableExecutables/AE_SYNTAX.md): Python holds ONLY true
syscalls — workspace files, terminal, HTTP, memory store, inbox, event polling,
LLM dispatch helpers. Everything composable (create_tool, update_entity, ...)
ships as editable .ae files under AutomatableExecutables/ and is composed on
the ae_read / ae_write / ae_delete / ae_list syscalls below.
"""
import ast
import json
import datetime
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from urllib import error, parse, request
from pathlib import Path

import yaml
try:
    import Taxonomy
except ImportError:  # Scripts/ may be imported as a package
    from . import Taxonomy
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False


if __package__:
    from .AEX import AEXError, AEXScript
else:
    from AEX import AEXError, AEXScript

# Stable per-process cache key for subagent requests (Mistral prompt caching).
import uuid as _uuid
_SUBAGENT_CACHE_KEY = f"ae-sub-{_uuid.uuid4().hex[:12]}"


TOOLSET_MAP = {
    "web": ["web_search", "web_fetch", "fetch_many"],
    "file": ["list_files", "glob", "read_file", "write_file", "patch_file", "search_files", "search_project", "workspace_overview", "inspect_changes", "wait_for_file"],
    "terminal": ["terminal", "bash", "run_executable"],
    "memory": ["memory_retain", "memory_recall", "memory_forget", "checkpoint", "remind", "session_usage", "agent_note"],
    "git": ["git_status", "git_diff", "git_log"],
    "skill": ["list_skills", "load_skill"],
    "event": ["list_events", "load_event", "create_event"],
    "executable": ["list_executables", "load_executable", "run_executable"],
    "delegation": ["check_subagent"],
}

# Tools a background worker may run without asking (stdin belongs to the user):
# reads (by effect) plus free, casual operations. Mirrors smart-mode auto-approve.
# NOTE: self notes run via run_skill(self_prompt|self_message), but run_skill as
# a whole stays approval-gated; background .ae code can still use self_prompt()
# directly inside execute blocks (no tool call needed).
BACKGROUND_AUTO_NAMES = {
    "web_search", "web_fetch", "fetch_many", "memory_retain", "memory_recall",
    "send_message", "todo_list", "list_skills", "load_skill", "list_executables",
    "load_executable", "list_events", "load_event", "list_files", "glob",
    "read_file", "search_files", "search_project", "git_status", "git_diff",
    "git_log", "get_syntax_guide", "ae_read", "ae_list", "skill_stats",
    "list_all_entities", "read_entity", "check_subagent", "session_usage",
    "wait_for_file", "remind", "checkpoint", "agent_note", "turn_stats", "handoff",
    "executions", "stop_execution", "clock",
}


def LooksLikeHardError(Result):
    """True when a tool returned an EXPLICIT hard-failure payload.

    IsError used to mean "an exception was raised", so a tool that caught the
    failure and returned {"status": "error", ...} or {"success": false,
    "error": ...} rendered as green "ok" and — worse — did not arm the runtime
    circuit breaker that refuses to repeat an identical failing call. A rejected
    dependency_install therefore looked successful and got retried verbatim.

    Deliberately conservative: it fires only on an explicit error marker, never
    on an absent one. Informational results (e.g. a search with zero hits) still
    count as success.
    """
    if not isinstance(Result, str):
        return False
    if not Result.lstrip().startswith("{"):
        return False
    try:
        Data = json.loads(Result)
    except Exception:
        return False
    if not isinstance(Data, dict):
        return False
    Status = Data.get("status")
    if isinstance(Status, str) and Status.strip().lower() == "error":
        return True
    if Data.get("success") is False and Data.get("error"):
        return True
    return False


class VolatileMemory:
    """In-memory only store for Temp Agents. No file persistence."""
    def __init__(self):
        self.ShortTerm = []
        self.ToolEvents = []
        self.LongTerm = []
        self.Todos = []

    def AddMessage(self, Role, Content):
        self.ShortTerm.append({"role": Role, "content": Content})

    def AddToolEvent(self, ToolName, Arguments, Result, IsError=False):
        self.ToolEvents.append({"name": ToolName, "arguments": Arguments, "result": str(Result)[:1200], "is_error": IsError})

    # Compatibility with Memory API
    def AddLongTerm(self, Text, Metadata=None, ID=None):
        if self.LongTerm is None:
            self.LongTerm = []
        try:
            entry = {"id": ID or f"vtm-{len(self.LongTerm)+1}", "text": Text, "metadata": Metadata or {}}
            self.LongTerm.append(entry)
            return entry["id"]
        except Exception:
            return None

    def ManageTodos(self, Action, *a, **kw):
        try:
            return {"todos": self.Todos, "count": len(self.Todos)}
        except Exception:
            return {"todos": [], "count": 0}

    def Save(self):
        pass

    def GetContext(self, AnchorTakeover=0):
        """Mirror of Memory.GetContext — trailing messages the caller re-sends
        itself (the recency anchor) are skipped so nothing is duplicated."""
        Recent = list(self.ShortTerm)
        if AnchorTakeover > 0:
            Recent = Recent[:-AnchorTakeover] if len(Recent) > AnchorTakeover else []
        return "\n\n".join(["Recent chat:\n" + "\n".join(f"{Item['role']}: {Item['content']}" for Item in Recent[-6:])] if Recent else [])

    @property
    def ShortTermCount(self):
        return len(self.ShortTerm)

    @property
    def LongTermCount(self):
        return len(self.LongTerm)


class TempAgent:
    """Temporary delegated worker. Executes until completion, then terminates.
    Contract: receives task + context, runs isolated, returns result to delegator.
    No nesting: delegate_task / steer / stop / revive inside a TempAgent are blocked.
    Background workers cannot prompt: reads + BACKGROUND_AUTO_NAMES run, everything
    else is skipped with a note the worker sees and routes around.
    """
    BLOCKED_IN_SUBAGENT = {"delegate_task", "steer_subagent", "stop_subagent", "revive_subagent"}

    def __init__(self, Goal, Context, ModelStr, ParentRegistry, Toolsets=None, MaxTokens=None, TimeoutSeconds=120, ReasoningMode="none", DelegatorName="*", SteerBox=None, StopFlag=None, Background=False, Stance="neutral"):
        self.Goal = Goal
        self.ContextText = Context or ""
        self.ModelStr = ModelStr
        self.Parent = ParentRegistry
        self.MaxTokens = MaxTokens or 2048
        self.Timeout = TimeoutSeconds
        self.ReasoningMode = ReasoningMode
        self.Delegator = DelegatorName
        self.SteerBox = SteerBox if SteerBox is not None else []
        self.StopFlag = StopFlag if StopFlag is not None else threading.Event()
        self.Background = Background
        self.Stance = Stance if Stance in {"neutral", "adversarial", "reviewer"} else "neutral"
        # Subagent memory: try to allocate a node under the delegator's tree.
        self.Memory = VolatileMemory()
        self.MemoryNode = None
        self.MemoryLabel = None
        try:
            ParentTree = getattr(self.Parent, "_MemoryTree", None)
            ParentRoot = getattr(self.Parent, "_MemoryTreeNode", None) or getattr(self.Parent, "_MemoryTreeRoot", None)
            if ParentTree is not None and ParentRoot:
                DelegatorName = DelegatorName or (getattr(self.Parent, "CurrentAgentName", "*"))
                SubName = f"subagent-{self.Goal[:32]}" if self.Goal else "subagent"
                Label, Mem = ParentTree.Create(ParentRoot, SubName)
                self.MemoryNode = Mem
                self.MemoryLabel = Label
                self.Memory = Mem
        except Exception:
            self.Memory = VolatileMemory()
            self.MemoryNode = None
            self.MemoryLabel = None
        self.LastUsage = {}
        Allowed = []
        if Toolsets:
            for Ts in Toolsets:
                Allowed.extend(TOOLSET_MAP.get(Ts, []))
        self.AllowedTools = set(Allowed) if Allowed else None  # None = all except blocked

    def _SiblingSnapshot(self):
        """Other live workers on the same team (read-only awareness)."""
        try:
            Sibs = [d for d in (self.Parent.Delegations or {}).values() if d.get("status") == "running"][:6]
        except Exception:
            return ""
        if not Sibs:
            return ""
        import time as _time
        Lines = []
        for _d in Sibs:
            try:
                _el = round(_time.time() - float(_d.get("started") or _time.time()), 0)
            except Exception:
                _el = 0
            Lines.append(f"- {_d.get('id', '?')}: {str(_d.get('goal', ''))[:100]} (running {_el}s)")
        return ("Sibling workers on your team (read-only — you cannot steer, stop, or message them; "
                "coordinate through your debrief to the delegator):\n" + "\n".join(Lines))

    def _filtered_tools(self):
        AllTools = self.Parent.GetOpenAITools()
        Out = []
        for T in AllTools:
            Name = T["function"]["name"]
            if Name in self.BLOCKED_IN_SUBAGENT:
                continue
            if self.AllowedTools is not None and Name not in self.AllowedTools:
                continue
            Out.append(T)
        return Out

    def _BackgroundAllowed(self, Name):
        try:
            if self.Parent._ToolEffect(Name) == "read":
                return True
        except Exception:
            pass
        return Name in BACKGROUND_AUTO_NAMES

    def run(self):
        if self.StopFlag.is_set():
            return {"status": "killed", "result": "Subagent stopped by user."}
        # Label .ae self-notes with who invoked this worker (best-effort;
        # restored afterwards since the registry is shared across threads).
        _PrevAgent = getattr(self.Parent, "CurrentAgentName", "*")
        try:
            self.Parent.CurrentAgentName = self.Delegator or _PrevAgent
        except Exception:
            pass
        try:
            # Delegation is pre-authorized: the delegator's goal IS the approval.
            # Tools run without prompts (sandbox jails still apply). Cleared below.
            try:
                self.Parent._local.auto_approve = True
            except Exception:
                pass
        except Exception:
            pass
        try:
            return self._run_inner()
        finally:
            try:
                self.Parent._local.auto_approve = False
            except Exception:
                pass
            try:
                self.Parent.CurrentAgentName = _PrevAgent
            except Exception:
                pass

    def _run_inner(self):
        sys.path.insert(0, str(self.Parent.ProjectRoot / "Scripts"))
        from ProviderCatalog import CHAT_PROVIDERS, ChatEndpointWithFallback
        try:
            endpoint, FailoverNote = ChatEndpointWithFallback(self.ModelStr)
        except Exception as Ex:
            HaveKeys = [P for P, C in CHAT_PROVIDERS.items() if any(os.getenv(K) for K in C["key_env"])]
            Hint = f" Providers with keys configured: {', '.join(sorted(HaveKeys)) or 'none'}. Use target_agent to inherit a profile's working provider, or the providers command to check."
            return {"status": "error", "result": f"Error resolving subagent provider: {Ex}{Hint}"}
        if FailoverNote:
            self.Memory.AddMessage("user", f"[System notice]{FailoverNote}")
        Prompt = f"Goal: {self.Goal}\n\n"
        if self.ContextText:
            Prompt += f"Context: {self.ContextText}\n\n"
        Prompt += ("You are a temporary subagent. Your tools run WITHOUT approval prompts — use them freely, get results back, and keep going until the goal is done. NEVER call delegate_task (no nested delegation). "
                   "When done, report BACK TO YOUR DELEGATOR with a detailed debrief, never a one-liner: restate the goal, list what you did per step with key results, give every file path you touched or created, and state clearly what remains or failed. The delegator relays to the user — your report is what they will see. "
                   "ARTIFACTS section (mandatory, last): one line per created/modified file as `path=<workspace-relative path> bytes=<size>`. "
                   "CONSTRAINT CHECK (mandatory): repeat the goal's numeric/format constraints back (lengths, counts, 'real', formats) with measured compliance per item (e.g. `chars=612/500 PASS`). No constraint restated = constraint ignored.")
        if self.Stance == "adversarial":
            Prompt += (" STANCE: RED-TEAM. The context contains work to attack — find holes, false claims, missing verification, breakage. Be merciless and specific; your debrief lists every flaw with evidence, severity-first.")
        elif self.Stance == "reviewer":
            Prompt += (" STANCE: REVIEWER. Verify claims against evidence: check files exist, re-run reported results where possible. Your debrief is verdict-first (PASS/FAIL) with findings.")
        _Snap = ""
        try:
            _Snap = self._SiblingSnapshot()
        except Exception:
            pass
        if _Snap:
            Prompt += "\n\n" + _Snap
        self._ElabAsked = False
        if self.ReasoningMode == "explicit":
            Prompt += "\n\n## Reasoning\nThink step by step before acting."
        Messages = [{"role": "user", "content": Prompt}]
        self.Memory.AddMessage("user", Prompt)
        Tools = self._filtered_tools()
        MaxRounds = 12
        for _ in range(MaxRounds + 1):
            if self.StopFlag.is_set():
                return {"status": "killed", "result": "Subagent stopped by user."}
            if self.SteerBox:
                Notes = list(self.SteerBox)
                del self.SteerBox[:]
                SteerText = "\n".join(f"- {N}" for N in Notes)
                Messages.append({"role": "user", "content": f"[Steering from {self.Delegator} — adjust course, then continue]\n{SteerText}"})
                self.Memory.AddMessage("user", f"[Steering consumed]\n{SteerText}")
            Payload = {"model": endpoint["model"], "messages": Messages, "temperature": 0.7, "max_tokens": self.MaxTokens, "tools": Tools, "tool_choice": "auto", "parallel_tool_calls": True}
            try:
                if endpoint.get("provider") == "mistral":
                    Payload["prompt_cache_key"] = _SUBAGENT_CACHE_KEY
            except Exception:
                pass
            Req = request.Request(endpoint["chat_url"], data=json.dumps(Payload).encode("utf-8"), headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {endpoint['api_key']}"}, method="POST")
            Data = None
            for _Attempt in range(2):
                try:
                    with request.urlopen(Req, timeout=self.Timeout) as Response:
                        Data = json.loads(Response.read().decode("utf-8"))
                    break
                except error.HTTPError as Ex:
                    if Ex.code == 429 and _Attempt == 0:
                        time.sleep(2.5)
                        continue
                    Body = Ex.read().decode("utf-8", errors="replace")
                    return {"status": "error", "result": f"Subagent API error: {Body}"}
                except Exception as Ex:
                    return {"status": "error", "result": f"Subagent error: {Ex}"}
            if Data is None:
                return {"status": "error", "result": "Subagent error: empty response"}
            Usage = Data.get("usage") or {}
            if Usage:
                self.LastUsage = Usage
            Message = Data["choices"][0]["message"]
            ToolCalls = Message.get("tool_calls", [])
            if not ToolCalls:
                Content = (Message.get("content") or "").strip() or "Subagent completed but returned no content."
                # Thin-report guard: a debrief after real tool use must carry
                # weight for the delegator. One elaboration nudge, then accept.
                if len(Content) < 300 and list(self.Memory.ToolEvents) and not self._ElabAsked:
                    self._ElabAsked = True
                    Messages.append(Message)
                    Messages.append({"role": "user", "content": "[Delegator] That debrief is too thin to relay. Expand it now: per-step actions with key results, every file path touched or created, and what remains or failed. No tools needed — just the full report."})
                    continue
                self.Memory.AddMessage("assistant", Content)
                return {"status": "done", "result": Content}  # terminate + callback payload
            if len(Messages) > 40:
                return {"status": "error", "result": "Subagent tool-call limit reached."}
            Messages.append(Message)
            for Tc in ToolCalls:
                Fn = Tc.get("function", {})
                Name = Fn.get("name", "")
                if Name in self.BLOCKED_IN_SUBAGENT:
                    Messages.append({"role": "tool", "tool_call_id": Tc.get("id", Name), "content": "Blocked: nested delegation is not allowed."})
                    continue
                if self.AllowedTools is not None and Name not in self.AllowedTools:
                    Messages.append({"role": "tool", "tool_call_id": Tc.get("id", Name), "content": f"Blocked: tool '{Name}' not in allowed toolsets."})
                    continue
                # No background skip: delegation is pre-authorized (auto_approve),
                # so workers run the same tools as sync ones. Tools that truly
                # need the main thread (ask_user, update_being) raise catchable
                # errors inside their handlers.
                try:
                    Args = json.loads(Fn.get("arguments") or "{}")
                    with self.Parent._lock:
                        Result = self.Parent.Execute(Name, Args)
                    self.Memory.AddToolEvent(Name, Args, Result, False)
                except Exception as Ex:
                    Result = f"Tool error: {Ex}"
                    self.Memory.AddToolEvent(Name, {}, Result, True)
                Messages.append({"role": "tool", "tool_call_id": Tc.get("id", Name), "content": Result})
        return {"status": "error", "result": "Subagent did not complete in time."}

class ToolRegistry:
    # The bootstrap set: with only these, an agent can still BUILD every other
    # capability. This is the default profile -- everything else is opt-in via
    # `permissions` in the agent's Being.ae.
    MetaCapabilities = frozenset({
        "create_tool", "create_skill", "create_workflow", "create_executable",
        "create_event", "create_knowledge", "create_note",
        "read_entity", "update_entity", "delete_entity",
        "list_capabilities", "list_all_entities", "get_syntax_guide",
        "share_capability",
        "query_knowledge", "task_complete", "clock", "ask_user",
        "memory_retain", "memory_recall", "memory_forget", "todo_list",
        "ae_read", "ae_write", "ae_delete", "ae_list",
    })

    def __init__(self, ProjectRoot, ApprovalCallback=None, MemorySystem=None, AutomationRoot=None):
        self.ProjectRoot = Path(ProjectRoot).resolve()
        self.AutomationRoot = Path(AutomationRoot).resolve() if AutomationRoot else self.ProjectRoot / "AutomatableExecutables"
        # Agents live beside AutomatableExecutables at the repo root, NOT inside the workspace.
        # Main.py passes WorkspaceRoot as ProjectRoot, so resolve repo root from AutomationRoot.
        self.RepoRoot = self.AutomationRoot.parent if self.AutomationRoot.name == "AutomatableExecutables" else self.ProjectRoot
        self.ToolsPath = self.AutomationRoot / "Tools"
        self.CoreToolsPath = self.ToolsPath / "core"
        self.SkillsPath = self.AutomationRoot / "Skills"
        self.ExecutablesPath = self.AutomationRoot / "Executables"
        self.EventsPath = self.AutomationRoot / "Events"
        self.EventStatePath = self.EventsPath / ".runtime-state.json"
        self.ApprovalCallback = ApprovalCallback
        self.Memory = MemorySystem
        self.Tools = {}
        self._skills_cache = None
        self.DeniedCache = set()  # (tool, canonical-args) denied this session: auto-deny, don't re-prompt
        self.AllowedCache = set()  # tool names always allowed this session via `allow` command
        self.LastDelegation = {}  # metadata of the most recent TempAgent run (model, elapsed, status, scope)
        self.CurrentAgentName = "*"  # set per-turn by Main; exposed to .ae as agent_name
        self.CallLog = deque(maxlen=500)  # (timestamp, tool, skill-or-empty, is_error) for usage stats + loop alerts
        self.RunningExecutions = {}  # exec_id -> {name, kind, started, progress, stop} for live .ae (loops can be stopped)
        self._ExecSeq = 0
        self.TurnCounts = {}  # live per-turn tool counts; Main rebinds this dict each turn (turn_stats reads it)
        # Per-turn evidence/failure ledgers. Main resets these each turn; they are
        # initialized here so every entry point (delegation, tests) can append
        # safely instead of the append being swallowed by a missing attribute.
        self.TurnToolEvidence = []
        self.TurnToolFailures = []
        self._local = threading.local()  # thread-local auto-approve flag for subagent execution
        self._lock = threading.RLock()  # serializes engine access across main thread + delegation threads
        # Hierarchical memory tree: every agent and subagent gets its own node.
        self._MemoryTree = None
        self._MemoryTreeRoot = None
        self._MemoryTreeNode = None
        try:
            AgentName = getattr(self, "CurrentAgentName", "*") or "*"
            if __package__:
                from .MemoryTree import MemoryTree
            else:
                from MemoryTree import MemoryTree
            self._MemoryTree = MemoryTree(BasePath=self.RepoRoot, RootLabel=AgentName if AgentName != "*" else "AE")
            if AgentName and AgentName != "*":
                RootLabel, _ = self._MemoryTree.CreateRoot(AgentName, NodeNumber=1)
                self._MemoryTreeRoot = RootLabel
                self._MemoryTreeNode = RootLabel
        except Exception:
            pass
        self.Delegations = {}  # id -> background delegation record
        self._DelegationSeq = 0
        self._Snapshots = []  # stack of (workspace-rel-path, previous-text-or-None) for undo
        self._SchedulerStop = threading.Event()
        self._SchedulerThread = None
        self._DashboardThread = None  # ticker: one-line runner status alongside input
        self.SchedulerEnabled = True  # global kill-switch, toggled by `scheduler` CLI command
        self.EventState = self._LoadEventState()
        # Capability permissions: which capability NAMES this agent may use.
        # Unrelated to approval (that is about side effects, decided by effect
        # + requires_approval). Defaults to the meta/bootstrap set.
        self.Permissions = None       # None = not resolved yet; [] = explicit none
        self.PermissionsSource = "default"
        self.PermissionCache = set()
        self.PermissionMode = "meta"
        self._Resolving = False
        # Deep-reason gate: set when a loop is detected; the agent must run the
        # deep_reason skill before any other tool (enforced in Main.Prompt).
        self.ReasoningRequired = False
        # Organized runtime services (Scripts/runtime/): diary, budget, safe hands.
        try:
            if __package__:
                from .runtime.tracing import Tracer
            else:
                from runtime.tracing import Tracer
            self.Tracer = Tracer(self.RepoRoot)
        except Exception:
            self.Tracer = None
        try:
            if __package__:
                from .runtime.budget import BudgetTracker
            else:
                from runtime.budget import BudgetTracker
            self.Budget = BudgetTracker()
        except Exception:
            self.Budget = None
        try:
            if __package__:
                from .runtime.sandbox import Sandbox
            else:
                from runtime.sandbox import Sandbox
            self.Sandbox = Sandbox(self.ProjectRoot)
        except Exception:
            self.Sandbox = None

        self.Register(
            "list_files",
            "List files under a workspace-relative directory.",
            {"type": "object", "properties": {"path": {"type": "string", "description": "Workspace-relative directory, default '.'"}}, "additionalProperties": False},
            self._ListFiles,
        )
        self.Register(
            "glob",
            "Find workspace files using a glob pattern, such as '**/*.py'.",
            {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["pattern"], "additionalProperties": False},
            self._Glob,
        )
        self.Register(
            "read_file",
            "Read a UTF-8 text file inside the workspace.",
            {"type": "object", "properties": {"path": {"type": "string", "description": "Workspace-relative file path"}}, "required": ["path"], "additionalProperties": False},
            self._ReadFile,
        )
        self.Register(
            "git_status",
            "Show the workspace Git status, including changed and untracked files.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._GitStatus,
        )
        self.Register(
            "git_diff",
            "Show workspace Git changes. Set staged=true to show the staged diff.",
            {"type": "object", "properties": {"staged": {"type": "boolean"}}, "additionalProperties": False},
            self._GitDiff,
        )
        self.Register(
            "git_log",
            "Show recent workspace commits.",
            {"type": "object", "properties": {"limit": {"type": "integer"}}, "additionalProperties": False},
            self._GitLog,
        )
        self.Register(
            "write_file",
            "Create or replace a UTF-8 text file inside the workspace. Requires user approval.",
            {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
            self._WriteFile,
            RequiresApproval=True,
        )
        self.Register(
            "search_files",
            "Search text inside workspace files, similar to a coding-agent grep tool.",
            {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "case_sensitive": {"type": "boolean"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._SearchFiles,
        )
        self.Register(
            "search_project",
            "Search text across the entire workspace (alias for search_files).",
            {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "case_sensitive": {"type": "boolean"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._SearchProject,
        )
        self.Register(
            "web_search",
            "Public web search (works with NO key — leave provider unset). fetch_full=true also pulls the top result's page. Auto backend: SearXNG multi-engine when SEARXNG_URL is set, else duckduckgo.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}, "fetch_full": {"type": "boolean", "description": "Also fetch the top result's full page"}, "provider": {"type": "string", "description": "Optional; unset = auto (SearXNG when configured, else duckduckgo). Keyed providers without keys are auto-skipped"}, "providers": {"type": "array", "items": {"type": "string"}, "description": "Optional fan-out; dead providers auto-skipped, merged deduped"}}, "required": ["query"], "additionalProperties": False},
            self._WebSearch,
            Effect="external",
        )
        self.Register(
            "weather",
            "Current weather + 3-day outlook for any place (free Open-Meteo backend, no key). Use for ALL weather questions instead of web_search.",
            {"type": "object", "properties": {"location": {"type": "string", "description": "Place name, e.g. 'Paris' or 'Clovis, California'"}}, "required": ["location"], "additionalProperties": False},
            self._Weather,
            Effect="external",
        )
        self.Register(
            "wikipedia",
            "Wikipedia search + article summaries (free, no key). Use for background, definitions, stable facts — not for current events.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._Wikipedia,
            Effect="external",
        )
        self.Register(
            "stack_search",
            "Stack Overflow Q&A search (free, no key). Use for programming errors, APIs, how-tos before generic web_search.",
            {"type": "object", "properties": {"query": {"type": "string", "description": "Error text or how-to question"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._StackSearch,
            Effect="external",
        )
        self.Register(
            "news_search",
            "World news search across outlets via GDELT (free, no key). Use for current events instead of web_search.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._NewsSearch,
            Effect="external",
        )
        self.Register(
            "image_search",
            "Image search (free DuckDuckGo library, no key). Returns direct image URLs plus source pages.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._ImageSearch,
            Effect="external",
        )
        self.Register(
            "web_fetch",
            "Fetch a public HTTP or HTTPS page as readable text using Jina Reader.",
            {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False},
            self._WebFetch,
            Effect="external",
        )
        self.Register(
            "text_to_speech",
            "Generate speech audio with ElevenLabs. This may incur provider charges and requires user approval.",
            {"type": "object", "properties": {"text": {"type": "string"}, "voice_id": {"type": "string"}, "model_id": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
            self._TextToSpeech,
            RequiresApproval=True,
            Effect="external",
        )
        self.Register(
            "patch_file",
            "Replace one exact text block in a workspace file. The old text must match exactly. Requires user approval.",
            {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False},
            self._PatchFile,
            RequiresApproval=True,
        )
        self.Register(
            "terminal",
            "Run a command in the workspace directory. Requires user approval.",
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False},
            self._RunTerminal,
            RequiresApproval=True,
        )
        self.Register(
            "bash",
            "Run a command with Bash in the workspace. Requires Bash and user approval.",
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False},
            self._RunBash,
            RequiresApproval=True,
            Effect="execute",
        )
        self.Register(
            "memory_retain",
            "Store durable facts only. Do not save temporary task outputs, educational examples, topic lists, outlines, report content, or todo data. Project facts must describe stable repository/workspace configuration unless the user explicitly asks you to remember them. A non-durable request returns stored=false without an error.",
            {"type": "object", "properties": {"text": {"type": "string"}, "category": {"type": "string", "enum": ["user_fact", "preference", "instruction", "project_fact", "learned_lesson"]}}, "required": ["text"], "additionalProperties": False},
            self._RetainMemory,
            Effect="write",
        )
        self.Register(
            "memory_recall",
            "Search long-term memory for facts relevant to a query.",
            {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"], "additionalProperties": False},
            self._RecallMemory,
        )
        self.Register(
            "memory_forget",
            "Forget a memory by its exact ID or exact text.",
            {"type": "object", "properties": {"memory_id_or_text": {"type": "string"}}, "required": ["memory_id_or_text"], "additionalProperties": False},
            self._ForgetMemory,
            Effect="write",
        )
        self.Register(
            "todo_list",
            "Manage the shared durable task list — flat or nested. add (with parent for subtasks), plan (whole tree in ONE call: [{title, status?, priority?, assignee?, subtasks:[...]}]), update, complete, delete (cascades), list. Statuses: todo/in_progress/blocked/done/cancelled. Phases auto-complete when all children are done/cancelled.",
            {"type": "object", "properties": {"action": {"type": "string", "enum": ["list", "add", "plan", "update", "complete", "delete"]}, "task_id": {"type": "string"}, "title": {"type": "string"}, "description": {"type": "string"}, "status": {"type": "string", "enum": ["todo", "in_progress", "blocked", "done", "cancelled"]}, "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]}, "assignee": {"type": "string", "description": "Agent profile name, or '*' for anyone"}, "parent": {"type": "string", "description": "Parent task id (add subtasks; plan attaches the whole tree under it)"}, "plan": {"type": "array", "description": "Nested tree for one-call planning", "items": {"type": "object"}}, "tasks": {"type": "array", "description": "Alias for plan (accepted for model leniency)", "items": {"type": "object"}}}, "required": ["action"], "additionalProperties": False},
            self._ManageTodos,
            Effect="write",
        )
        self.Register(
            "list_skills",
            "List available harness skills from AutomatableExecutables/Skills.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._ListSkills,
        )
        # NOTE: create_tool/create_skill/create_executable/create_event/read_entity/
        # update_entity/delete_entity/list_all_entities/get_syntax_guide/update_syntax_guide
        # are NOT registered here on purpose: they ship as editable .ae files in
        # Tools/*.ae composed on the ae_read/ae_write/ae_delete/ae_list syscalls below.
        # GetOpenAITools() auto-discovers them. Agents can read and improve them.
        self.Register(
            "load_skill",
            "Load the full SKILL.md or skill.ae instructions for a discovered skill.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False},
            self._LoadSkill,
        )
        self.Register(
            "run_skill",
            "Execute a skill with parameters. Skills are executable .ae scripts with defined parameters.",
            {"type": "object", "properties": {"name": {"type": "string"}, "parameters": {"type": "object"}}, "required": ["name"], "additionalProperties": False},
            self._RunSkill,
            RequiresApproval=True,
        )
        self.Register(
            "list_executables",
            "List AE automation scripts available in AutomatableExecutables/Executables.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._ListExecutables,
        )
        self.Register(
            "load_executable",
            "Read the full definition of an AE automation script before running it.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False},
            self._LoadExecutable,
        )
        self.Register(
            "run_executable",
            "Run a custom .ae automation script. The script is reviewed and requires user approval.",
            {"type": "object", "properties": {"name": {"type": "string"}, "parameters": {"type": "object"}}, "required": ["name"], "additionalProperties": False},
            self._RunExecutable,
            RequiresApproval=True,
        )
        self.Register(
            "list_events",
            "List configured AEX events and their trigger descriptions.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._ListEvents,
        )
        self.Register(
            "load_event",
            "Read a configured event's complete AEX definition.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False},
            self._LoadEvent,
        )
        self.Register(
            "delegate_task",
            "Delegate a task to a subagent (TempAgent: isolated volatile memory, restricted tool copy, no nesting, auto-cleanup). To delegate to ANOTHER AGENT PROFILE, pass target_agent (e.g. 'TestAgent') and NO provider/model — it inherits that profile's provider. Pass provider+model ONLY to force a specific backend. Stance: neutral (default), adversarial (red-team the context: finds holes with evidence), reviewer (verdict-first PASS/FAIL audit). The worker reports back to YOU with a detailed debrief — you relay to the user.",
            {"type": "object", "properties": {"goal": {"type": "string", "description": "The task goal for the subagent"}, "context": {"type": "string", "description": "Additional context for the subagent"}, "target_agent": {"type": "string", "description": "Name of agent profile to delegate to (uses that profile's provider/model; preferred for profile-to-profile work)"}, "provider": {"type": "string", "description": "Chat provider ONLY to force a backend (e.g., 'groq'). Must have a configured API key — check the providers command first"}, "model": {"type": "string", "description": "Model ID for forced backend"}, "max_tokens": {"type": "integer", "description": "Max tokens for subagent response (50-100 for self-delegate, higher for full delegation)"}, "toolsets": {"type": "array", "items": {"type": "string"}, "description": "Allowed tool categories: web|file|terminal|memory|git|skill|event|executable|delegation(check_subagent)"}, "timeout_seconds": {"type": "integer", "description": "Timeout for subagent execution"}, "stance": {"type": "string", "enum": ["neutral", "adversarial", "reviewer"], "description": "Worker stance: neutral does the task, adversarial red-teams the context, reviewer audits verdict-first"}, "background": {"type": "boolean", "description": "If true, run in background and return immediately with a delegation id; steer with steer_subagent, collect with check_subagent, stop with stop_subagent"}}, "required": ["goal"], "additionalProperties": False},
            self._DelegateTask,
            RequiresApproval=True,
            Effect="execute",
        )
        self.Register(
            "steer_subagent",
            "Redirect a running background delegation by id. The text lands in the worker's next round. Fails if the delegation already finished.",
            {"type": "object", "properties": {"id": {"type": "string", "description": "Delegation id (dlg_...)"}, "text": {"type": "string", "description": "Steering instruction"}}, "required": ["id", "text"], "additionalProperties": False},
            self._SteerSubagent,
            Effect="write",
        )
        self.Register(
            "check_subagent",
            "Check a background delegation: status, elapsed, and result (full result once done).",
            {"type": "object", "properties": {"id": {"type": "string", "description": "Delegation id (dlg_...)"}}, "required": ["id"], "additionalProperties": False},
            self._CheckSubagent,
        )
        self.Register(
            "stop_subagent",
            "Stop a running background delegation by id. Sibling delegations keep running.",
            {"type": "object", "properties": {"id": {"type": "string", "description": "Delegation id (dlg_...)"}}, "required": ["id"], "additionalProperties": False},
            self._StopSubagent,
            Effect="write",
        )
        self.Register(
            "revive_subagent",
            "Re-run a finished background delegation with its original goal, model, and toolsets, plus extra guidance. Returns a new delegation id. Only for terminal (done/error/killed) delegations.",
            {"type": "object", "properties": {"id": {"type": "string", "description": "Finished delegation id (dlg_...)"}, "extra_context": {"type": "string", "description": "What to do differently this time"}}, "required": ["id"], "additionalProperties": False},
            self._ReviveSubagent,
            RequiresApproval=True,
            Effect="execute",
        )
        # Agent Communication & Self-Prompting (memory/inbox syscalls: no approval, like memory_retain)
        # NOTE: self notes are skills now (Skills/self/self_prompt.ae,
        # Skills/self/self_message.ae) — run them via run_skill. The Python
        # _SelfPrompt handler below remains as the syscall those skills use.
        self.Register(
            "skill_stats",
            "How many times a skill (or tool) ran: session total plus usage inside a recent window. Use to avoid repeating yourself.",
            {"type": "object", "properties": {"name": {"type": "string", "description": "Skill or tool name; empty lists the top tracked names"}, "window_minutes": {"type": "integer", "description": "Recent window in minutes (default 60)"}}, "additionalProperties": False},
            self._SkillStats,
        )
        self.Register(
            "session_usage",
            "Live harness stats for this session: tool calls total/errors, per-tool top counts, token + USD budget state. Read-only; use to pace yourself before heavy fan-outs.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._SessionUsage,
        )
        self.Register(
            "fetch_many",
            "Fetch up to 10 public http/https pages in parallel (same Jina-reader pipeline as web_fetch). One call returns [{url, text} or {url, error}] in input order. Use instead of N serial web_fetch calls.",
            {"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}, "description": "Up to 10 public http/https URLs"}}, "required": ["urls"], "additionalProperties": False},
            self._FetchMany,
            Effect="external",
        )
        self.Register(
            "wait_for_file",
            "Block until a workspace file appears/changes (or timeout). For waits on user drops, build output, or downloads. Polls mtime/size; never burns LLM rounds while waiting.",
            {"type": "object", "properties": {"path": {"type": "string", "description": "Workspace-relative file (may not exist yet) or directory to watch"}, "timeout_seconds": {"type": "integer", "description": "Max wait, 5-600 (default 60)"}, "poll_seconds": {"type": "number", "description": "Poll interval, 0.5-10 (default 1)"}}, "required": ["path"], "additionalProperties": False},
            self._WaitForFile,
            Effect="execute",
        )
        self.Register(
            "checkpoint",
            "Named snapshot of todos + long-term memory + summary under .ae_sessions/checkpoints/. save before risky refactors, restore to roll back agent state (restore/delete ask approval).",
            {"type": "object", "properties": {"action": {"type": "string", "enum": ["save", "restore", "list", "delete"]}, "name": {"type": "string", "description": "Snapshot name (letters, numbers, _-)"}}, "required": ["action"], "additionalProperties": False},
            self._Checkpoint,
            RequiresApproval=True,
            Effect="write",
        )
        self.Register(
            "remind",
            "One-shot reminder to yourself: after delay_seconds the text lands in your inbox and surfaces next turn. Survives nothing but the process — for within-session nudges, not cron.",
            {"type": "object", "properties": {"text": {"type": "string", "description": "Reminder text"}, "delay_seconds": {"type": "integer", "description": "Delay 10-86400s (default 300)"}, "target_agent": {"type": "string", "description": "Agent profile inbox (default: you)"}}, "required": ["text"], "additionalProperties": False},
            self._Remind,
            Effect="write",
        )
        self.Register(
            "agent_note",
            "Your private scratchpad: notes you write to yourself, injected into your system prompt every turn (NOT long-term memory). add/list/clear. Lessons must be about the WORLD/task ('weather.com blocks anonymous fetches — use accuweather'), NEVER about tool mechanics ('X tool requires Y', 'file Z does not exist') — mechanics go stale and pollute every future turn. Max 20 notes, 300 chars each.",
            {"type": "object", "properties": {"action": {"type": "string", "enum": ["add", "list", "clear"]}, "text": {"type": "string", "description": "Note text (required for add, ≤300 chars)"}}, "required": ["action"], "additionalProperties": False},
            self._AgentNote,
        )
        self.Register(
            "update_being",
            "Rewrite your own Being/system prompt (identity-level corrections only — use agent_note for lessons). Requires user approval. Full replacement text, 50-6000 chars.",
            {"type": "object", "properties": {"text": {"type": "string", "description": "Complete new Being prompt"}}, "required": ["text"], "additionalProperties": False},
            self._UpdateBeing,
            RequiresApproval=True,
            Effect="write",
        )
        self.Register(
            "turn_stats",
            "Live usage ledger: per-tool call counts for THIS turn plus session tops. Read-only; check before repeating a tool.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._TurnStats,
        )
        self.Register(
            "clock",
            "Current date and time (server local + UTC, ISO + human). Use for ANY current date/time/month/year question BEFORE web_search — search snippets cannot tell time.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._Clock,
        )
        self.Register(
            "stage_gate",
            "Milestone gate: present what was achieved + demo, and the user picks continue / redirect / stop (main thread only). Use at phase boundaries of long tasks instead of running silent to the end.",
            {"type": "object", "properties": {"milestone": {"type": "string", "description": "What was just achieved"}, "demo": {"type": "string", "description": "Evidence: paths, outputs, results (≤2000 chars)"}}, "required": ["milestone"], "additionalProperties": False},
            self._StageGate,
        )
        self.Register(
            "handoff",
            "Handoff packet for continuity: save goal/tried/remaining/key-files (+ todos, notes, summary auto-captured) under .ae_sessions/handoffs/; load one to resume with zero re-discovery. save/list/load/delete.",
            {"type": "object", "properties": {"action": {"type": "string", "enum": ["save", "load", "list", "delete"]}, "id": {"type": "string", "description": "Packet id (letters, numbers, _-)"}, "goal": {"type": "string"}, "tried": {"type": "string"}, "remaining": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}}}, "required": ["action"], "additionalProperties": False},
            self._Handoff,
            Effect="write",
        )
        self.Register(
            "executions",
            "Live .ae executions (loops, trackers): id, name, elapsed, last report() progress. Pair with stop_execution.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._ListExecutions,
        )
        self.Register(
            "stop_execution",
            "Cooperatively stop a running .ae execution by id (its next should_stop()/report() check exits). No force-kill exists in-process — .ae code must poll.",
            {"type": "object", "properties": {"id": {"type": "string", "description": "Execution id from `executions`"}}, "required": ["id"], "additionalProperties": False},
            self._StopExecution,
            Effect="write",
        )
        self.Register(
            "send_message",
            "Send a message to another agent profile. The message is added to their conversation history.",
            {"type": "object", "properties": {"target_agent": {"type": "string", "description": "Name of the target agent profile"}, "content": {"type": "string", "description": "Message content"}}, "required": ["target_agent", "content"], "additionalProperties": False},
            self._SendMessage,
        )
        self.Register(
            "ask_user",
            "Ask the user 1-6 questions and wait. questions[] items: string or {question, options[2-8], default, required}. Returns {answers, answer}.",
            {"type": "object", "properties": {"question": {"type": "string", "description": "Single question (or use `questions`)"}, "questions": {"type": "array", "description": "Up to 6 batched questions", "items": {"type": "object"}}}, "additionalProperties": False},
            self._AskUser,
        )
        self.Register(
            "task_complete",
            "Signal that all work is complete and the task is finished. Call this when all todos are done and you're ready to stop.",
            {"type": "object", "properties": {"summary": {"type": "string", "description": "Brief summary of what was accomplished"}}, "required": ["summary"], "additionalProperties": False},
            self._TaskComplete,
        )
        self.Register(
            "undo",
            "Restore the workspace file changed by the most recent write_file/patch_file (or undo to N steps back). Use after a bad edit.",
            {"type": "object", "properties": {"steps": {"type": "integer", "description": "How many file changes to roll back (default 1)"}}, "additionalProperties": False},
            self._UndoLast,
            RequiresApproval=True,
        )
        # Automation-layer syscalls: scoped file access under AutomatableExecutables/.
        # These are the ONLY Python file primitives allowed outside the workspace.
        # Agents compose them via .ae tools (create_tool, update_entity, ...), which
        # are shipped as editable .ae files — see Tools/create_*.ae.
        _ScopeProp = {
            "type": "string",
            "enum": ["local", "shared", "global"],
            "description": ("Which capability tree: 'local' is this agent's own (default when it has "
                            "one), 'shared' is the common tree, 'global' the repo-level tree."),
        }
        self.Register(
            "ae_read",
            "Read a .ae file from a capability tree (path relative to it, e.g. 'Tools/foo.ae', 'Events/bar.ae', 'Skills/x/skill.ae').",
            {"type": "object", "properties": {"path": {"type": "string", "description": "Tree-relative file path"}, "scope": dict(_ScopeProp)}, "required": ["path"], "additionalProperties": False},
            self._AeRead,
        )
        self.Register(
            "ae_write",
            "Create or overwrite a .ae file in a capability tree. Requires user approval.",
            {"type": "object", "properties": {"path": {"type": "string", "description": "Tree-relative file path"}, "content": {"type": "string", "description": "Complete UTF-8 file content"}, "scope": dict(_ScopeProp)}, "required": ["path", "content"], "additionalProperties": False},
            self._AeWrite,
            RequiresApproval=True,
        )
        self.Register(
            "ae_delete",
            "Delete a .ae file from a capability tree. Requires user approval.",
            {"type": "object", "properties": {"path": {"type": "string", "description": "Tree-relative file path"}, "scope": dict(_ScopeProp)}, "required": ["path"], "additionalProperties": False},
            self._AeDelete,
            RequiresApproval=True,
        )
        self.Register(
            "ae_list",
            "Inventory all Tools, Skills, Executables, and Events with metadata. Backs list_all_entities.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            self._AeList,
        )
        self.Register(
            "list_capabilities",
            "Everything you can use: tools, skills, workflows, executables, events, knowledge, notes and "
            "data, grouped by type and category. Filter by type/category/tag/scope, or pass query to "
            "search names and descriptions. This is your authoritative inventory — a capability not "
            "listed here is not callable.",
            {"type": "object", "properties": {
                "type": {"type": "string", "enum": sorted(AEXScript.Types), "description": "Restrict to one capability type."},
                "category": {"type": "string", "description": "Restrict to one category (e.g. 'core', 'research')."},
                "tag": {"type": "string", "description": "Restrict to capabilities carrying this tag."},
                "scope": {"type": "string", "enum": ["local", "shared", "global"], "description": "Restrict to one capability tree."},
                "query": {"type": "string", "description": "Search names, descriptions and tags."},
                "limit": {"type": "integer", "description": "Cap the number of results."},
            }, "additionalProperties": False},
            self._ListCapabilities,
        )
        self.Register(
            "share_capability",
            "Publish one of your OWN capabilities so other agents can use it. Your capabilities start "
            "private; sharing is the explicit opt-in. Copies the file, so your local copy keeps working. "
            "Requires user approval.",
            {"type": "object", "properties": {
                "name": {"type": "string", "description": "Capability name as you know it."},
                "type": {"type": "string", "enum": sorted(AEXScript.Types), "description": "Capability type; omit to search all types."},
                "scope": {"type": "string", "enum": ["shared", "global"], "description": "Where to publish. Default 'shared'."},
                "category": {"type": "string", "description": "Publish under this category instead of the current one."},
                "overwrite": {"type": "boolean", "description": "Replace an existing capability of the same name."},
            }, "required": ["name"], "additionalProperties": False},
            self._ShareCapability,
            RequiresApproval=True,
            Effect="write",
        )
        self.LoadToolPlugins()
        # Cognitive primitives (Scripts/tools/cognitive.py): observable reasoning,
        # self-critique, meaning search, runtime capability discovery.
        try:
            if __package__:
                from .tools.cognitive import register_cognitive_tools
            else:
                from tools.cognitive import register_cognitive_tools
            register_cognitive_tools(self)
        except Exception as Ex:
            try:
                if self.Tracer is not None:
                    self.Tracer.log("error", where="cognitive_register", error=str(Ex)[:200])
            except Exception:
                pass


    def _ResolvePermissions(self):
        """Read the active agent's `permissions:` field and compile it to a
        set of capability names. Accepted forms:
            permissions: all                 -> every capability (opt into global)
            permissions: meta                -> bootstrap set only (default)
            permissions: [web_search, skill] -> exactly these names
            permissions: {tool: web_search, category: research}
        """
        if self.Permissions is not None:
            return self.PermissionCache
        if self._Resolving:
            # Reentrant call (e.g. category: expansion listing capabilities).
            # Scope-gated lookups must not recurse back into resolution.
            return self.PermissionCache
        self._Resolving = True
        try:
            return self._ResolvePermissionsInner()
        finally:
            self._Resolving = False

    @staticmethod
    def _truthy(Value):
        return Value is True or str(Value).strip().lower() in ("true", "yes", "all", "on")

    def _ExpandTaxonomyGrant(self, Kind, Value):
        """Names granted by a `category:` / `tag:` permission entry.

        Scope-agnostic on purpose: during resolution the permission gate is not
        bound yet, so this reads every root directly rather than _VisibleRoots.
        """
        Want = Taxonomy.Normalize(Value)
        Granted = set()
        for Type in sorted(AEXScript.Types):
            for Directory in self._AllRoots(Type):
                if not Directory.is_dir():
                    continue
                for AEXFile in Directory.rglob("*.ae"):
                    try:
                        Script = AEXScript.FromFile(AEXFile)
                    except (AEXError, OSError):
                        continue
                    if Script.Type != Type:
                        continue
                    if (Script.Category == Want) if Kind == "category" else (Want in Script.Tags):
                        Granted.add(Script.Name)
        return Granted

    def _ResolvePermissionsInner(self):
        Pending = getattr(self, "_PendingPermissions", None)
        Raw = Pending
        Source = getattr(self, "_PendingSource", "default")
        Declared = Pending is not None
        # An explicit SetPermissions call wins over the profile file: that is
        # how .switch / a freshly created agent rebinds the whitelist.
        Root = self.AgentRoot() if not Declared else None
        if Root is not None:
            Parent = Root.parent
            for Candidate in (Parent / "Being.ae", Parent / "agent.json"):
                if not Candidate.is_file():
                    continue
                try:
                    if Candidate.suffix == ".json":
                        Data = json.loads(Candidate.read_text(encoding="utf-8"))
                    else:
                        Data = AEXScript.FromFile(Candidate).Metadata
                except Exception:
                    continue
                if "permissions" in Data:
                    Raw = Data.get("permissions")
                    Source = Candidate.name
                    Declared = True
                    break
        Allowed = set()
        # No `permissions:` field anywhere means the profile predates the
        # whitelist: grant everything so an existing agent never silently loses
        # capabilities. Restriction is opt-in (`permissions: meta` or a list).
        if Raw is None and not Declared:
            self.PermissionMode = "all"
            self.Permissions = []
            self.PermissionsSource = Source
            self.PermissionCache = set()
            return self.PermissionCache
        if isinstance(Raw, str):
            Raw = [Raw] if Raw.strip().lower() not in ("all", "meta") else Raw.strip().lower()
        if isinstance(Raw, str) and Raw == "all":
            self.PermissionMode = "all"
            self.Permissions = []
            self.PermissionsSource = Source
            self.PermissionCache = set()
            return self.PermissionCache
        # `permissions: [all]` (YAML turns the bare word into a one-item list)
        # must mean all-mode, not a capability literally named "all".
        if isinstance(Raw, (list, tuple, set)) and any(
            isinstance(Item, str) and Item.strip().lower() in ("all", "*") for Item in Raw
        ):
            self.PermissionMode = "all"
            self.Permissions = []
            self.PermissionsSource = Source
            self.PermissionCache = set()
            return self.PermissionCache
        if Raw == "meta" or Raw is None:
            self.PermissionMode = "meta"
            Allowed = set(self.MetaCapabilities)
        elif isinstance(Raw, (list, tuple, set)):
            self.PermissionMode = "list"
            # YAML turns `[category: core]` into [{'category': 'core'}], so a
            # list may carry mappings as well as plain names.
            Flat = []
            for Item in Raw:
                if isinstance(Item, dict):
                    for Key, Value in Item.items():
                        if self._truthy(Value):
                            Flat.append(str(Key))
                        elif isinstance(Value, (str, list, tuple)):
                            Flat.append(f"{Key}: {Value if isinstance(Value, str) else ','.join(map(str, Value))}")
                elif isinstance(Item, str):
                    Flat.append(Item)
            Raw = Flat
            for Item in Raw:
                if not isinstance(Item, str):
                    continue
                Key = Item.strip()
                if ":" in Key:
                    Kind, _, Value = Key.partition(":")
                    Kind = Kind.strip().lower()
                    if Kind in ("category", "tag"):
                        Allowed |= self._ExpandTaxonomyGrant(Kind, Value.strip())
                        continue
                    Key = Value.strip()
                Allowed.add(Key)
        elif isinstance(Raw, dict):
            # `permissions: {web_search: true, category: research}` and the
            # YAML shorthand `permissions: {category: research}` both land here.
            self.PermissionMode = "dict"
            Items = []
            for Key, Value in Raw.items():
                if str(Key).strip().lower() == "all" and self._truthy(Value):
                    self.PermissionMode = "all"
                    continue
                if self._truthy(Value):
                    Items.append(str(Key))
                elif isinstance(Value, (str, list, tuple)):
                    Items.append(f"{Key}: {Value if isinstance(Value, str) else ','.join(map(str, Value))}")
            Raw = Items
            for Item in Raw:
                if ":" in Item:
                    Kind, _, Value = Item.partition(":")
                    if Kind.strip().lower() in ("category", "tag"):
                        Allowed |= self._ExpandTaxonomyGrant(Kind.strip().lower(), Value.strip())
                        continue
                Allowed.add(Item.strip())
        # Capabilities present on disk but absent from the whitelist stay usable
        # only through the bootstrap set -- an agent never silently exceeds it.
        for Name in self.MetaCapabilities:
            Allowed.add(Name)
        self.Permissions = []
        self.PermissionsSource = Source
        self.PermissionCache = Allowed
        self.Permissions = []
        return Allowed

    def PermissionAllows(self, Name):
        Allowed = self._ResolvePermissions()
        if self.PermissionMode == "all":
            return True
        return str(Name) in Allowed

    def _EnsureCapabilityAllowed(self, Name, Scope=None):
        """Raise unless this agent may use the named capability.

        Local capabilities are the agent's own tree and are always usable;
        everything else is governed by the `permissions:` whitelist. This keeps
        the declared whitelist from being advisory only.
        """
        if Scope == "local" or self.PermissionAllows(Name):
            return
        raise PermissionError(
            f"Capability '{Name}' is not in this agent's permissions. Use "
            f"list_capabilities to see what you actually have, or ask the user "
            f"to add it to `permissions:` in the profile."
        )

    def SetPermissions(self, Raw, Source="runtime"):
        """Rebind permissions for a switched/created agent profile.

        Accepts a YAML scalar/sequence ("all", "meta", "[a, b]") or an already
        parsed list, matching what Being.ae holds.
        """
        if isinstance(Raw, str):
            try:
                Raw = yaml.safe_load(Raw)
            except yaml.YAMLError:
                Raw = [Raw.strip()]
        self.Permissions = None
        self.PermissionMode = "meta"
        self._PendingPermissions = Raw
        self._PendingSource = Source
        self.PermissionsSource = Source
        self._ResolvePermissions()
        self.Permissions = []
        return self.PermissionCache

    def _VisibleCapabilities(self):
        """Capability entries this agent is allowed to see, filtered by
        permissions. Used for the prompt listing so an agent is never told
        about a tool it cannot call."""
        Allowed = self._ResolvePermissions()
        Entries = self.ListCapabilities()
        if self.PermissionMode == "all":
            return Entries
        # Local capabilities are the agent's own; a whitelist restricts which
        # shared/global capabilities it may reach, not what it created itself.
        return [E for E in Entries if E.get("scope") == "local" or E["name"] in Allowed]

    def LoadToolPlugins(self):
        if not self.ToolsPath.is_dir():
            return

        for PluginPath in sorted(self.ToolsPath.glob("*.py")):
            if PluginPath.name.startswith("_"):
                continue
            ModuleSpec = importlib.util.spec_from_file_location(f"agent_tool_{PluginPath.stem}", PluginPath)
            if ModuleSpec is None or ModuleSpec.loader is None:
                raise ImportError(f"Could not load tool plugin: {PluginPath}")
            Module = importlib.util.module_from_spec(ModuleSpec)
            ModuleSpec.loader.exec_module(Module)
            RegisterTools = getattr(Module, "register_tools", None)
            if not callable(RegisterTools):
                raise AttributeError(f"Tool plugin must define register_tools(registry): {PluginPath}")
            ExistingTools = set(self.Tools)
            RegisterTools(self)
            for Name in set(self.Tools) - ExistingTools:
                Tool = self.Tools[Name]
                if not Tool["effect_explicit"]:
                    Tool["effect"] = "execute"
                    Tool["requires_approval"] = True
                    Tool["parallel_safe"] = False

    def _AEXFiles(self, Directory):
        if not Directory.is_dir():
            return []
        return sorted(Directory.rglob("*.ae"))

    # ---------------------------------------------------------------- capability
    # paths: every capability type resolves through here, so a new type is one
    # folder in Taxonomy.TYPE_FOLDERS rather than a new code path.
    def _TypePath(self, ScriptType):
        for Folder, Type in Taxonomy.TYPE_FOLDERS.items():
            if Type == ScriptType:
                return self.AutomationRoot / Folder
        return self.AutomationRoot / ScriptType.capitalize()

    def AgentRoot(self, AgentName=None):
        """An agent's own capability tree, or None when it has none."""
        Name = AgentName or self.CurrentAgentName
        if not Name or Name == "*":
            return None
        Candidate = self.RepoRoot / "Agents" / str(Name) / "AutomatableExecutables"
        return Candidate if Candidate.is_dir() else None

    def _AllRoots(self, ScriptType):
        """Every root on disk for a type (local, shared, global), unfiltered by
        permissions. Used during permission resolution itself."""
        Roots = []
        LocalRoot = self.AgentRoot()
        if LocalRoot:
            Local = self._TypePathFrom(LocalRoot, ScriptType)
            if Local.is_dir():
                Roots.append(Local)
        Shared = self._TypePathFrom(self.AutomationRoot, ScriptType)
        if Shared.is_dir() and Shared not in Roots:
            Roots.append(Shared)
        GlobalRoot = self.RepoRoot / "AutomatableExecutables"
        if GlobalRoot != self.AutomationRoot:
            SharedGlobal = self._TypePathFrom(GlobalRoot, ScriptType)
            if SharedGlobal.is_dir() and SharedGlobal not in Roots:
                Roots.append(SharedGlobal)
        return Roots

    def _VisibleRoots(self, ScriptType):
        """(directory, scope) pairs for one capability type.

        Scope 'local' outranks 'shared' outranks 'global', so an agent that
        defines its own tool shadows the shared one of the same name without a
        flag fight. `permissions` gates which scopes an agent may read.
        """
        Roots = []
        LocalRoot = self.AgentRoot()
        if LocalRoot:
            Local = self._TypePathFrom(LocalRoot, ScriptType)
            if Local.is_dir():
                Roots.append((Local, "local"))
        # Every root is listed so a capability named in `permissions:` stays
        # resolvable; the per-name gate (`PermissionAllows`, plus the local-scope
        # exemption) decides what may actually run. Gating whole trees on a
        # capability literally named "global" made shared/global capabilities
        # unreachable for any restricted profile.
        SharedRoot = self._TypePathFrom(self.AutomationRoot, ScriptType)
        if SharedRoot.is_dir() and all(SharedRoot.resolve() != P.resolve() for P, _ in Roots):
            Roots.append((SharedRoot, "shared"))
        GlobalRoot = self.RepoRoot / "AutomatableExecutables"
        if GlobalRoot != self.AutomationRoot and GlobalRoot.is_dir():
            Global = self._TypePathFrom(GlobalRoot, ScriptType)
            if Global.is_dir() and all(Global.resolve() != P.resolve() for P, _ in Roots):
                Roots.append((Global, "global"))
        return Roots

    def _TypePathFrom(self, Root, ScriptType):
        for Folder, Type in Taxonomy.TYPE_FOLDERS.items():
            if Type == ScriptType:
                return Root / Folder
        return Root / ScriptType.capitalize()

    def FindCapability(self, Name, ScriptType=None, Required=True):
        """Resolve one capability by name across every visible root.

        Later roots never win: the list is ordered local -> shared -> global and
        the first hit returns, so shadowing is deterministic.
        """
        Types = [ScriptType] if ScriptType else sorted(AEXScript.Types)
        for Type in Types:
            for Directory, Scope in self._VisibleRoots(Type):
                if not Directory.is_dir():
                    continue
                for AEXFile in sorted(Directory.rglob("*.ae")):
                    try:
                        Script = AEXScript.FromFile(AEXFile)
                    except (AEXError, OSError):
                        continue
                    if Script.Name == Name and Script.Type == Type:
                        Script.Metadata.setdefault("_scope", Scope)
                        return Script
            # Events and knowledge also live in nested Active/Inactive trees;
            # the rglob above covers those, so nothing extra is needed here.
        if Required:
            Known = ", ".join(sorted(set(AEXScript.Types)))
            raise ValueError(f"No capability named '{Name}' is visible to you. Searched types: {Known}. "
                             f"Use list_capabilities to see what you actually have.")
        return None

    def ListCapabilities(self, ScriptType=None, Category=None, Tag=None, Scope=None):
        """Every visible capability, grouped-ready and deterministically sorted."""
        Types = [ScriptType] if ScriptType else sorted(AEXScript.Types)
        Found = {}
        for Type in Types:
            for Directory, InScope in self._VisibleRoots(Type):
                if not Directory.is_dir():
                    continue
                for AEXFile in sorted(Directory.rglob("*.ae")):
                    if AEXFile.name.startswith("."):
                        continue
                    try:
                        Script = AEXScript.FromFile(AEXFile)
                    except (AEXError, OSError):
                        continue
                    if Script.Type != Type:
                        continue
                    if Category and Script.Category != Taxonomy.Normalize(Category, Type):
                        continue
                    if Tag and Taxonomy.Normalize(Tag, Type) not in Script.Tags:
                        continue
                    if Scope and InScope != Scope:
                        continue
                    if Script.Name in Found and InScope not in ("local",):
                        continue  # higher-scope entry already won
                    Found[Script.Name] = (Script, InScope)
        Results = []
        for Name, (Script, InScope) in Found.items():
            Results.append({
                "name": Name,
                "type": Script.Type,
                "category": Script.Category,
                "subcategory": Script.Subcategory,
                "tags": Script.Tags,
                "description": Script.Description,
                "active": Script.Active,
                "scope": InScope,
                "path": str(Script.SourcePath),
                "version": Script.Version,
            })
        Results.sort(key=lambda E: Taxonomy.SortKey(E["type"], E["category"], E["name"]))
        return Results

    def _FindAEX(self, Name, ScriptType, Directories=None):
        """Back-compatible single-directory lookup. Prefer FindCapability."""
        if Directories:
            Matches = []
            for Directory in Directories:
                for AEXFile in self._AEXFiles(Directory):
                    try:
                        Script = AEXScript.FromFile(AEXFile)
                    except (AEXError, OSError):
                        continue
                    if Script.Name == Name and Script.Type == ScriptType:
                        Matches.append(Script)
            if len(Matches) > 1:
                raise ValueError(f"Multiple {ScriptType} scripts use the name '{Name}'.")
            return Matches[0] if Matches else None
        return self.FindCapability(Name, ScriptType, Required=False)

    def _FindCoreTool(self, Name):
        Script = self._FindAEX(Name, "tool", [self.CoreToolsPath])
        if Script is not None and Script.Metadata.get("core") is not True:
            raise ValueError(f"Core tool override '{Name}' must declare core: true.")
        return Script

    def _LoadEventState(self):
        try:
            with self.EventStatePath.open("r", encoding="utf-8") as File:
                State = json.load(File)
            return State if isinstance(State, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _SaveEventState(self):
        with self._lock:
            self.EventsPath.mkdir(parents=True, exist_ok=True)
            TemporaryFile = self.EventStatePath.with_suffix(".tmp")
            TemporaryFile.write_text(json.dumps(self.EventState, indent=2), encoding="utf-8")
            TemporaryFile.replace(self.EventStatePath)

    def Register(self, Name, Description, Parameters, Handler, RequiresApproval=False, Effect=None):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", Name):
            raise ValueError(f"Invalid tool name: {Name}")
        if Name in self.Tools:
            raise ValueError(f"Tool already registered: {Name}")
        if Effect not in {None, "read", "write", "execute", "external"}:
            raise ValueError("Effect must be read, write, execute, or external.")
        EffectExplicit = Effect is not None
        if Effect is None:
            Effect = "write" if RequiresApproval else "read"
        self.Tools[Name] = {
            "description": Description,
            "parameters": Parameters,
            "handler": Handler,
            "requires_approval": RequiresApproval,
            "effect": Effect,
            "effect_explicit": EffectExplicit,
            "parallel_safe": Effect == "read",
        }

    def GetOpenAITools(self):
        Allowed = self._ResolvePermissions()
        InAllMode = self.PermissionMode == "all"
        LocalRoot = self.AgentRoot()
        LocalTools = self._TypePathFrom(LocalRoot, "tool") if LocalRoot else None

        def MayUse(Name, IsLocal):
            # Local own-tree capabilities are always callable; shared/global
            # ones must be named in `permissions:` unless the profile is all-mode.
            return IsLocal or InAllMode or str(Name) in Allowed

        Definitions = [
            {
                "type": "function",
                "function": {
                    "name": Name,
                    "description": Tool["description"],
                    "parameters": Tool["parameters"],
                },
            }
            for Name, Tool in self.Tools.items()
            if MayUse(Name, False)
        ]
        Registered = {Definition["function"]["name"]: Index for Index, Definition in enumerate(Definitions)}
        Directories = [self.ToolsPath, self.ExecutablesPath]
        if LocalTools is not None and LocalTools.is_dir():
            Directories.append(LocalTools)
        for Directory in Directories:
            for AEXFile in self._AEXFiles(Directory):
                try:
                    Script = AEXScript.FromFile(AEXFile)
                except AEXError:
                    continue
                if Script.Type != "tool":
                    continue
                IsLocal = (
                    LocalTools is not None
                    and LocalTools.resolve() in Path(Script.SourcePath).resolve().parents
                )
                if not MayUse(Script.Name, IsLocal):
                    continue
                IsCoreOverride = (
                    self.CoreToolsPath.resolve() in Path(Script.SourcePath).resolve().parents
                    and Script.Metadata.get("core") is True
                )
                if Script.Name in Registered:
                    if IsCoreOverride:
                        Definitions[Registered[Script.Name]] = Script.FunctionSchema()
                    continue
                Registered[Script.Name] = len(Definitions)
                Definitions.append(Script.FunctionSchema())
        return Definitions

    def ListMCPTools(self):
        return [
            {
                "name": Definition["function"]["name"],
                "description": Definition["function"]["description"],
                "inputSchema": Definition["function"]["parameters"],
            }
            for Definition in self.GetOpenAITools()
        ]

    def CallMCPTool(self, Name, Arguments):
        try:
            Result = self.Execute(Name, Arguments)
            return {"content": [{"type": "text", "text": Result}], "isError": False}
        except Exception as Ex:
            return {"content": [{"type": "text", "text": str(Ex)}], "isError": True}

    def _AskApproval(self, Name, Arguments):
        """Single approval gate: session denial cache + strong do-not-retry message."""
        try:
            # Subagent execution runs pre-approved: the delegator's goal is the
            # authorization, and the sandbox still jails paths/commands. The
            # subagent reports back to the DELEGATOR when done — never the user.
            if getattr(getattr(self, "_local", None), "auto_approve", False):
                return True
        except Exception:
            pass
        try:
            Key = (Name, json.dumps(Arguments, sort_keys=True, ensure_ascii=False, default=str))
        except Exception:
            Key = (Name, str(Arguments))
        # Allow explicit session allow-list to override previous denial for the same tool
        if Name in self.AllowedCache:
            # Remove any cached denials for this tool so future checks pass immediately
            self.DeniedCache = {k for k in self.DeniedCache if k[0] != Name}
            return True
        if Key in self.DeniedCache:
            raise PermissionError("Denied by user earlier in this session — not asking again. Do NOT retry this exact call; change arguments or strategy, use '.allow <tool>' to bypass, or proceed without it.")
        # Memo: a YES covers identical repeats for 5 minutes (10 identical
        # delegate_task prompts in one batch ask once, not 10 times).
        try:
            import time as _time
            Now = _time.time()
            self._ApprovedCache = {k: ts for k, ts in getattr(self, "_ApprovedCache", {}).items() if Now - ts < 300}
            if Key in self._ApprovedCache:
                return True
        except Exception:
            pass
        if self.ApprovalCallback is None or not self.ApprovalCallback(Name, Arguments):
            self.DeniedCache.add(Key)
            raise PermissionError("Denied by user. Do NOT retry this exact call with identical arguments; change arguments or strategy, use '.allow <tool>' to always allow this tool this session, or proceed without it.")
        try:
            self._ApprovedCache[Key] = _time.time()
        except Exception:
            pass
        return True

    def _NormalizeName(self, Name):
        Name = (Name or "").strip()
        Match = re.match(r"^([A-Za-z0-9_-]+)\s*\(.*\)\s*$", Name, re.DOTALL)
        return Match.group(1) if Match else Name

    def ClearDeniedCache(self, Name=None):
        """Remove denied entries. If Name is provided, clear denials for that tool only; otherwise clear all."""
        if Name is None:
            self.DeniedCache.clear()
        else:
            self.DeniedCache = {k for k in self.DeniedCache if k[0] != Name}

    def Execute(self, Name, Arguments):
        Name = self._NormalizeName(Name)
        if not isinstance(Arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        try:
            if self.Tracer is not None:
                self.Tracer.log("tool_call", tool=Name, arguments=Arguments,
                                agent=getattr(self, "CurrentAgentName", "*"))
        except Exception:
            pass
        CoreScript = self._FindCoreTool(Name)
        if CoreScript is not None:
            self._EnsureCapabilityAllowed(Name)
            Effect = CoreScript.Metadata.get("effect", "execute")
            RequiresApproval = CoreScript.Metadata.get("requires_approval", Effect != "read")
            Approved = False
            if RequiresApproval:
                self._AskApproval(Name, Arguments)
                Approved = True
            _Eid = self._ExecBegin(Name, "core")
            try:
                Res = CoreScript.Execute(Arguments, self, self.Memory, Approved=Approved)
            finally:
                self._ExecEnd(_Eid)
            return Res if isinstance(Res, str) else json.dumps(Res, ensure_ascii=False)
        if Name not in self.Tools:
            # Resolve across every visible tree (local -> shared -> global), so a
            # tool this agent created in its own folder is callable by name.
            Script = self.FindCapability(Name, "tool", Required=False)
            if Script is None:
                raise ValueError(f"Unknown tool: {Name}. Use list_capabilities to see what you actually have; do NOT retry this name.")
            self._EnsureCapabilityAllowed(Name, Scope=Script.Metadata.get("_scope"))
            # Reads never prompt (core or custom): effect:read declares no side effects.
            NeedsApproval = Script.Metadata.get("effect", "execute") != "read" or bool(Script.Metadata.get("requires_approval", False))
            if NeedsApproval:
                self._AskApproval(Name, Arguments)
            # Approval propagates inward: the outer call was just approved, so the
            # script's own call_builtin() composition does not re-prompt (same as core tools).
            _Eid = self._ExecBegin(Name, "tool")
            try:
                Res = Script.Execute(Arguments, self, self.Memory, Approved=True)
            finally:
                self._ExecEnd(_Eid)
            # Vision grounding: remember the packet's palette + mean brightness
            # (with timestamp) so task_complete can reject claims the data contradicts.
            if Name == "vision" and isinstance(Res, str):
                try:
                    import re as _re, time as _time
                    _m = _re.search(r"COLORS \(real, by coverage\): ([^\n]{1,300})", Res)
                    _lm = _re.search(r"LIGHT GEOMETRY \(mean brightness (\d+)/255\)", Res)
                    if _m:
                        self._LastPalette = (_m.group(1).strip(), _time.time(),
                                             int(_lm.group(1)) if _lm else 255)
                except Exception:
                    pass
            return Res if isinstance(Res, str) else json.dumps(Res, ensure_ascii=False)

        Tool = self.Tools[Name]
        self._EnsureCapabilityAllowed(Name)
        # deep_reason is pure diagnosis (no side effects) — never spend an
        # approval prompt on the loop-crisis exit, batch or otherwise.
        _NoAsk = (Name == "run_skill" and isinstance(Arguments, dict)
                  and str(Arguments.get("name", "")) == "deep_reason")
        if Tool["requires_approval"] and not _NoAsk:
            self._AskApproval(Name, Arguments)

        Result = Tool["handler"](**Arguments)
        return Result if isinstance(Result, str) else json.dumps(Result, ensure_ascii=False)

    def ExecutePrimitive(self, Name, Arguments, ApprovalGranted=False):
        if Name not in self.Tools:
            raise ValueError(f"Unknown builtin primitive: {Name}")
        Tool = self.Tools[Name]
        _NoAsk = (Name == "run_skill" and isinstance(Arguments, dict)
                  and str(Arguments.get("name", "")) == "deep_reason")
        if Tool["requires_approval"] and not ApprovalGranted and not _NoAsk:
            self._AskApproval(Name, Arguments)
        Result = Tool["handler"](**Arguments)
        return Result if isinstance(Result, str) else json.dumps(Result, ensure_ascii=False)

    def _ToolEffect(self, Name):
        Name = self._NormalizeName(Name)
        CoreScript = self._FindCoreTool(Name)
        if CoreScript is not None:
            if CoreScript.Metadata.get("effect") == "read" and CoreScript.Metadata.get("parallel_safe") is True:
                return "read"
            return CoreScript.Metadata.get("effect", "execute")
        Tool = self.Tools.get(Name)
        if Tool is not None:
            return Tool["effect"]
        Script = self._FindAEX(Name, "tool", [self.ToolsPath, self.ExecutablesPath])
        if Script is not None and Script.Metadata.get("effect") == "read" and not Script.Metadata.get("requires_approval", False):
            return "read"
        return "execute"

    def ExecuteBatch(self, ToolCalls, MaxWorkers=4):
        Results = [None] * len(ToolCalls)

        def ExecuteOne(Index):
            ToolCall = ToolCalls[Index]
            Function = ToolCall.get("function", {})
            Name = Function.get("name", "")
            Arguments = {}
            try:
                Arguments = json.loads(Function.get("arguments") or "{}")
                if not isinstance(Arguments, dict):
                    Arguments = {}
                # Repair: models sometimes mangle the name field itself —
                # "list_skills()" or "glob({\"pattern\": ...})". Recover the
                # real name (and inline args) instead of failing the call.
                Name = (Name or "").strip()
                if Name.startswith("{"):
                    try:
                        Blob, _JsonEnd = json.JSONDecoder().raw_decode(Name)
                        if isinstance(Blob, dict):
                            _Tail = Name[_JsonEnd:].strip()
                            _TailName = re.match(r"([A-Za-z0-9_-]+)", _Tail)
                            _RecognizedTail = not _Tail or (_TailName and _TailName.group(1) in {"web_fetch", "web_search"})
                            if _RecognizedTail and isinstance(Blob.get("url"), str) and Blob["url"].strip():
                                Name, Arguments = "web_fetch", {"url": Blob["url"].strip()}
                            elif _RecognizedTail and isinstance(Blob.get("query"), str) and Blob["query"].strip():
                                Name, Arguments = "web_search", {"query": Blob["query"].strip()}
                    except Exception:
                        pass
                if "(" in Name:
                    Match = re.match(r"^([A-Za-z0-9_-]+)\s*\((.*)\)\s*$", Name, re.DOTALL)
                    if Match:
                        Name = Match.group(1)
                        try:
                            Inline = json.loads(Match.group(2))
                            if isinstance(Inline, dict):
                                Merged = dict(Inline)
                                Merged.update(Arguments)
                                Arguments = Merged
                        except Exception:
                            pass
                    else:
                        Name = re.sub(r"\(\)$", "", Name).strip()
                # Leniency for zero-param tools: models sometimes echo schema
                # keys; extras are ignored rather than failing the call.
                if Arguments:
                    Schema = None
                    Tool = self.Tools.get(Name)
                    if Tool is not None:
                        Schema = Tool["parameters"]
                    else:
                        Core = self._FindCoreTool(Name)
                        if Core is not None:
                            Schema = {"properties": {p: {} for p in Core.Parameters}}
                        else:
                            Found = self._FindAEX(Name, "tool", [self.ToolsPath, self.ExecutablesPath])
                            if Found is not None:
                                Schema = {"properties": {p: {} for p in Found.Parameters}}
                    if Schema is not None and not Schema.get("properties"):
                        Arguments = {}
                with self._lock:
                    Result = self.Execute(Name, Arguments)
                IsHardError = LooksLikeHardError(Result)
                if Name != "task_complete" and not IsHardError:
                    try:
                        self.TurnToolEvidence.append((Name, Arguments, Result))
                    except Exception:
                        pass
                if IsHardError:
                    try:
                        self.TurnToolFailures.append((Name, Arguments, Result))
                    except Exception:
                        pass
                Skill = ""
                if Name == "run_skill" and isinstance(Arguments, dict):
                    Skill = str(Arguments.get("name", ""))
                self.CallLog.append((time.time(), Name, Skill, IsHardError))
                return Name, Arguments, Result, IsHardError
            except Exception as Ex:
                try:
                    self.TurnToolFailures.append((Name, Arguments, str(Ex)))
                except Exception:
                    pass
                Skill = ""
                try:
                    if Name == "run_skill" and isinstance(Arguments, dict):
                        Skill = str(Arguments.get("name", ""))
                except Exception:
                    pass
                try:
                    # A rejected completion poisons auto-[OK]: the turn may no
                    # longer end as success without a fixed resubmit.
                    if self._NormalizeName(Name) == "task_complete":
                        self.TurnRejectedComplete = True
                except Exception:
                    pass
                self.CallLog.append((time.time(), Name, Skill, True))
                return Name, Arguments if isinstance(Arguments, dict) else {}, f"Tool error: {Ex}", True

        # Dedupe identical web_fetch URLs in one batch: execute once, reuse.
        # (Models re-send the same URL 2-3x when asked for "10 sources".)
        # Plus turn-level: URLs already fetched EARLIER this turn (kept in
        # _FetchedURLs by _FetchOneUrl) resolve from cache with a marker.
        try:
            _UrlFirst = {}
            _Cache = getattr(self, "_FetchedURLs", None) or {}
            for _Di, _Dc in enumerate(ToolCalls):
                try:
                    _Dn = self._NormalizeName((_Dc.get("function", {}) or {}).get("name", ""))
                    if _Dn not in ("web_fetch", "fetch_many"):
                        continue
                    _Da = json.loads((_Dc.get("function", {}) or {}).get("arguments") or "{}")
                    _Du = str(_Da.get("url", "") or "").strip().lower()
                    if not _Du:
                        continue
                    if _Du in _UrlFirst:
                        Results[_Di] = ("__duplicate__", _Di, _UrlFirst[_Du])
                    elif _Dn == "web_fetch" and _Du in _Cache:
                        Results[_Di] = ("web_fetch", _Da, str(_Cache[_Du]) + "\n[cached — fetched earlier this turn, not re-downloaded]", False)
                    else:
                        _UrlFirst[_Du] = _Di
                except Exception:
                    continue
        except Exception:
            pass
        Index = 0
        while Index < len(ToolCalls):
            if Results[Index] is not None:
                Index += 1
                continue
            Name = self._NormalizeName(ToolCalls[Index].get("function", {}).get("name", ""))
            if self._ToolEffect(Name) == "read":
                End = Index + 1
                while End < len(ToolCalls):
                    NextName = self._NormalizeName(ToolCalls[End].get("function", {}).get("name", ""))
                    if self._ToolEffect(NextName) != "read":
                        break
                    End += 1
                BatchIndices = [ _Bi for _Bi in range(Index, End) if Results[_Bi] is None ]
                if len(BatchIndices) > 1:
                    with ThreadPoolExecutor(max_workers=min(MaxWorkers, len(BatchIndices))) as Executor:
                        BatchResults = list(Executor.map(ExecuteOne, BatchIndices))
                elif len(BatchIndices) == 1:
                    BatchResults = [ExecuteOne(BatchIndices[0])]
                else:
                    BatchResults = []
                for BatchIndex, Result in zip(BatchIndices, BatchResults):
                    Results[BatchIndex] = Result
                Index = End
            else:
                Results[Index] = ExecuteOne(Index)
                Index += 1
        # Resolve deduped fetches by copying the first identical result.
        try:
            for _Ri, _Rv in enumerate(list(Results)):
                if isinstance(_Rv, tuple) and len(_Rv) == 3 and _Rv[0] == "__duplicate__":
                    _Src = Results[_Rv[2]]
                    if isinstance(_Src, tuple) and len(_Src) == 4:
                        _N, _A, _R, _E = _Src
                        Results[_Ri] = (_N, _A, f"{_R}\n[duplicate URL in same batch — fetched once]", _E)
        except Exception:
            pass
        return Results

    def SkillUsage(self, name="", window_minutes=60):
        """How many times a skill (or tool) ran: total + inside the window."""
        try:
            Window = max(1, int(window_minutes))
        except (TypeError, ValueError):
            Window = 60
        Now = time.time()
        Cutoff = Now - Window * 60
        Total = 0
        Recent = 0
        LastTs = 0.0
        Key = (name or "").strip().lower()
        for Ts, Tool, Skill, _Err in list(self.CallLog):
            Hit = (Skill or Tool or "").strip().lower()
            if Key and Hit != Key and Skill.lower() != Key and Tool.lower() != Key:
                continue
            Total += 1
            if Ts >= Cutoff:
                Recent += 1
            if Ts > LastTs:
                LastTs = Ts
        return {"name": name or "(all)", "total": Total, f"last_{Window}m": Recent,
                "last_used": datetime.datetime.fromtimestamp(LastTs).isoformat(timespec="seconds") if LastTs else None}

    def _SkillStats(self, name="", window_minutes=60):
        """Usage stats for one skill/tool, or every tracked name when empty."""
        if (name or "").strip():
            return self.SkillUsage(name, window_minutes)
        Seen = {}
        for _Ts, Tool, Skill, _Err in list(self.CallLog):
            Label = Skill or Tool
            if Label:
                Seen[Label] = Seen.get(Label, 0) + 1
        Top = sorted(Seen.items(), key=lambda kv: kv[1], reverse=True)[:20]
        return {"tracked": [{"name": N, "total": C,
                             f"last_{window_minutes}m": self.SkillUsage(N, window_minutes)[f"last_{window_minutes}m"]}
                            for N, C in Top]}

    def GetSkills(self):
        # Skills are raw .ae ONLY (SKILL.md was migrated away). Any .ae file
        # with `type: skill` anywhere under Skills/ is a skill, keyed by name.
        if self._skills_cache is not None:
            return self._skills_cache
        Skills = []
        if not self.SkillsPath.is_dir():
            self._skills_cache = Skills
            return Skills
        ae_skills = {}
        for SkillFile in sorted(self.SkillsPath.rglob("*.ae")):
            try:
                Script = AEXScript.FromFile(SkillFile)
            except (AEXError, OSError):
                continue
            if Script.Type != "skill":
                continue
            ae_skills[Script.Name] = {
                "name": Script.Name,
                "description": Script.Description,
                "path": SkillFile.relative_to(self.SkillsPath).as_posix(),
                "type": "ae",
            }
        Skills.extend(ae_skills.values())
        self._skills_cache = Skills
        return Skills

    def _ListSkills(self):
        return self.GetSkills()

    def _LoadSkill(self, name):
        """Load a skill by name. Returns structured data with metadata and content."""
        # First try exact name match
        for Skill in self.GetSkills():
            if Skill["name"] == name:
                SkillFile = (self.SkillsPath / Skill["path"]).resolve()
                if self.SkillsPath.resolve() not in SkillFile.parents:
                    raise ValueError("Skill path must stay inside the skills directory.")
                Content = SkillFile.read_text(encoding="utf-8")
                return {"name": Skill["name"], "description": Skill["description"], "path": Skill["path"], "content": Content, "type": Skill.get("type", "ae")}
        
        # Try category/name format (legacy SKILL.md support)
        if "/" in name:
            SkillFile = (self.SkillsPath / name).resolve()
            if self.SkillsPath.resolve() in SkillFile.parents and SkillFile.name == "SKILL.md":
                Content = SkillFile.read_text(encoding="utf-8")
                return {"name": name, "description": "", "path": name, "content": Content, "type": "md"}
        
        # Try to find by partial name match
        matches = [s for s in self.GetSkills() if name.lower() in s["name"].lower()]
        if len(matches) == 1:
            SkillFile = (self.SkillsPath / matches[0]["path"]).resolve()
            if self.SkillsPath.resolve() in SkillFile.parents:
                Content = SkillFile.read_text(encoding="utf-8")
                return {"name": matches[0]["name"], "description": matches[0]["description"], "path": matches[0]["path"], "content": Content, "type": matches[0].get("type", "ae")}
        elif len(matches) > 1:
            names = [s["name"] for s in matches]
            raise ValueError(f"Ambiguous skill name '{name}'. Matches: {', '.join(names)}. Use exact name or category/name format.")
        
        # List available skills for better error message
        available = [s["name"] for s in self.GetSkills()]
        raise ValueError(f"Skill not found: {name}. Available skills: {', '.join(sorted(available))}")

    def _RunSkill(self, name, parameters=None):
        """Execute a skill with given parameters."""
        if parameters is None:
            parameters = {}
        if isinstance(parameters, str):
            # Models sometimes JSON-encode the object into a string — recover it.
            try:
                parameters = json.loads(parameters)
            except Exception:
                raise ValueError(f"Skill '{name}' parameters must be an object, got a string that is not valid JSON.")
            if not isinstance(parameters, dict):
                raise ValueError(f"Skill '{name}' parameters must be an object.")
        for Skill in self.GetSkills():
            if Skill["name"] == name:
                SkillFile = (self.SkillsPath / Skill["path"]).resolve()
                if self.SkillsPath.resolve() not in SkillFile.parents:
                    raise ValueError("Skill path must stay inside the skills directory.")
                if Skill.get("type") != "ae":
                    raise ValueError(f"Skill '{name}' is not an executable .ae skill. Only .ae skills can be executed.")
                Script = AEXScript.FromFile(SkillFile)
                if Script.Type != "skill":
                    raise ValueError(f"File {SkillFile} is not a skill type.")
                # Track usage for loop detection / skill_stats
                import time as _time
                with self._lock:
                    self.CallLog.append((_time.time(), "run_skill", name, False))
                _Eid = self._ExecBegin(name, "skill")
                try:
                    return Script.Execute(parameters, self, self.Memory, Approved=True)
                finally:
                    self._ExecEnd(_Eid)
        # Helpful failure: a file matching the name exists but doesn't parse
        # (otherwise the agent loops "create -> not found -> create" forever).
        for SkillFile in sorted(self.SkillsPath.rglob("*.ae")):
            if name.lower() in SkillFile.as_posix().lower():
                try:
                    AEXScript.FromFile(SkillFile)
                except Exception as Ex:
                    raise ValueError(f"Skill '{name}' exists at {SkillFile.relative_to(self.AutomationRoot).as_posix()} but FAILED TO LOAD: {Ex}. Fix it with read_entity/update_entity or delete it, then retry.") from Ex
        raise ValueError(f"Skill not found: {name}")

    def _Glob(self, pattern, path=".", max_results=100):
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("pattern must be workspace-relative and cannot contain '..'.")
        Root = self._ResolveProjectPath(path)
        if not Root.is_dir():
            raise ValueError(f"Directory not found: {path} (workspace root is '{self.ProjectRoot.name}/' — use path='.' or relative paths like 'story.txt')")
        Limit = max(1, min(int(max_results), 500))
        Ignored = {".git", "__pycache__", ".venv", "node_modules", "chroma_db", "hermes-import"}
        Matches = []
        for Match in Root.glob(pattern) if "**" in pattern or "/" in pattern else Root.rglob(pattern):
            if not Match.is_file() or Ignored.intersection(Match.parts):
                continue
            Resolved = Match.resolve()
            if self.ProjectRoot not in Resolved.parents:
                continue
            Matches.append(Resolved.relative_to(self.ProjectRoot).as_posix())
            if len(Matches) >= Limit:
                break
        return Matches

    def _RunGit(self, Arguments):
        Git = shutil.which("git")
        if not Git:
            raise RuntimeError("Git is not installed or not available on PATH.")
        Result = subprocess.run([Git, *Arguments], cwd=self.ProjectRoot, capture_output=True, text=True, timeout=30, check=False, shell=False)
        Output = (Result.stdout + Result.stderr).strip()
        if len(Output) > 12000:
            Output = Output[:12000] + "\n[output truncated]"
        return f"Exit code: {Result.returncode}\n{Output}"

    def _GitStatus(self):
        return self._RunGit(["status", "--short", "--branch"])

    def _GitDiff(self, staged=False):
        Arguments = ["diff"]
        if staged:
            Arguments.append("--cached")
        Arguments.append("--stat")
        return self._RunGit(Arguments)

    def _GitLog(self, limit=10):
        Count = max(1, min(int(limit), 50))
        return self._RunGit(["log", f"-{Count}", "--oneline", "--decorate"])

    def _ResolveProjectPath(self, RelativePath):
        # Models paste absolute Windows paths and stray quotes — normalize first.
        Clean = str(RelativePath or "").strip()
        if len(Clean) >= 2 and Clean[0] == Clean[-1] and Clean[0] in "\"'":
            Clean = Clean[1:-1].strip()
        Target = (self.ProjectRoot / Clean).resolve()
        if Target != self.ProjectRoot and self.ProjectRoot not in Target.parents:
            raise ValueError("Path must stay inside the workspace.")
        return Target

    def _ListFiles(self, path="."):
        Directory = self._ResolveProjectPath(path)
        if not Directory.is_dir():
            raise ValueError(f"Directory not found: {path} (workspace root is '{self.ProjectRoot.name}/' — use path='.' or relative paths like 'story.txt')")
        Files = []
        for Current, Directories, Names in os.walk(Directory):
            Directories[:] = [Name for Name in Directories if Name not in {".git", "__pycache__", ".venv"}]
            for Name in Names:
                Files.append((Path(Current) / Name).relative_to(self.ProjectRoot).as_posix())
                if len(Files) >= 200:
                    return Files
        return Files

    def _ReadFile(self, path):
        FilePath = self._ResolveProjectPath(path)
        if not FilePath.is_file():
            raise ValueError(f"File not found: {path}")
        if FilePath.stat().st_size > 1_000_000:
            raise ValueError("File exceeds the 1 MB read limit.")
        if FilePath.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico"}:
            raise ValueError(f"{FilePath.name} is an image, not text — use the vision tool to see it (never read_file, never claim blindness).")
        try:
            return FilePath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{FilePath.name} is binary, not readable text.")

    def _SearchFiles(self, query, path=".", glob="*", case_sensitive=False, max_results=25):
        SearchRoot = self._ResolveProjectPath(path)
        if not SearchRoot.is_dir():
            raise ValueError(f"Directory not found: {path} (workspace root is '{self.ProjectRoot.name}/' — use path='.' or relative paths like 'story.txt')")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string.")

        Needle = query if case_sensitive else query.casefold()
        Limit = max(1, min(int(max_results), 100))
        Matches = []
        Ignored = {".git", "__pycache__", ".venv", "node_modules", "chroma_db", "hermes-import"}
        for Current, Directories, Names in os.walk(SearchRoot):
            Directories[:] = [Name for Name in Directories if Name not in Ignored]
            for Name in Names:
                if not fnmatch.fnmatch(Name, glob):
                    continue
                FilePath = Path(Current) / Name
                try:
                    if FilePath.stat().st_size > 1_000_000:
                        continue
                    Lines = FilePath.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError):
                    continue
                for LineNumber, Line in enumerate(Lines, start=1):
                    Haystack = Line if case_sensitive else Line.casefold()
                    if Needle in Haystack:
                        Relative = FilePath.relative_to(self.ProjectRoot).as_posix()
                        Matches.append(f"{Relative}:{LineNumber}: {Line[:400]}")
                        if len(Matches) >= Limit:
                            return Matches
        return Matches

    def _SearchProject(self, query, path=".", glob="*", case_sensitive=False, max_results=25):
        """Search across the workspace (path defaults to workspace root)."""
        return self._SearchFiles(query, path=path, glob=glob, case_sensitive=case_sensitive, max_results=max_results)

    def _DDGFallbackSearch(self, query, max_results):
        """Stdlib-only DuckDuckGo search (no ddgs needed). Used when ddgs isn't installed."""
        import html as _html
        import re as _re
        Url = "https://html.duckduckgo.com/html/?q=" + parse.quote(query.strip())
        Req = request.Request(Url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}, method="GET")
        try:
            with request.urlopen(Req, timeout=30) as Response:
                Html = Response.read(500_000).decode("utf-8", errors="replace")
        except Exception as Ex:
            raise RuntimeError(f"DuckDuckGo fallback search failed: {Ex}") from Ex
        Links = _re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', _re.DOTALL).findall(Html)
        Snips = _re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', _re.DOTALL).findall(Html)
        Out = []
        for i, (Href, Title) in enumerate(Links[:max_results]):
            Href = _html.unescape(Href)
            if "uddg=" in Href:
                M = _re.search(r"uddg=([^&]+)", Href)
                if M:
                    Href = parse.unquote(M.group(1))
            Title = _re.sub(r"<[^>]+>", "", _html.unescape(Title)).strip() or "No title"
            Body = _re.sub(r"<[^>]+>", "", _html.unescape(Snips[i])).strip() if i < len(Snips) else ""
            Out.append({"title": Title, "href": Href, "body": Body})
        return Out

    def _Weather(self, location):
        """Current weather + 3-day outlook via Open-Meteo (free, no key)."""
        if not isinstance(location, str) or not location.strip():
            raise ValueError("location must be a non-empty place name (e.g. 'Paris', 'Clovis, California').")
        import json as _json

        def _Get(Url):
            Req = request.Request(Url, headers={"User-Agent": "AE-weather/1.0"}, method="GET")
            with request.urlopen(Req, timeout=25) as Resp:
                return _json.loads(Resp.read(200_000).decode("utf-8", errors="replace"))

        Geo = _Get("https://geocoding-api.open-meteo.com/v1/search?name="
                   + parse.quote(location.strip()) + "&count=1&language=en&format=json")
        Results = Geo.get("results") or []
        if not Results:
            raise ValueError(f"No matching place found for '{location.strip()}'. Be more specific (add state/country).")
        Top = Results[0]
        Lat, Lon = Top.get("latitude"), Top.get("longitude")
        Name = ", ".join(x for x in [Top.get("name"), Top.get("admin1"), Top.get("country")] if x)
        Fx = _Get("https://api.open-meteo.com/v1/forecast?latitude=" + str(Lat) + "&longitude=" + str(Lon)
                  + "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
                  + "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                  + "&temperature_unit=fahrenheit&timezone=auto&forecast_days=4")
        Codes = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                 45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
                 56: "Freezing drizzle", 57: "Freezing drizzle", 61: "Light rain", 63: "Rain",
                 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain", 71: "Light snow",
                 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light showers",
                 81: "Showers", 82: "Heavy showers", 85: "Snow showers", 86: "Snow showers",
                 95: "Thunderstorm", 96: "Storm + hail", 99: "Storm + hail"}
        Cur = Fx.get("current", {}) or {}
        Daily = Fx.get("daily", {}) or {}
        Days = Daily.get("time", [])
        Lines = [f"Weather for {Name} (as of {Cur.get('time', '?')})",
                 f"Now: {Cur.get('temperature_2m', '?')}F, feels {Cur.get('apparent_temperature', '?')}F, "
                 f"{Codes.get(Cur.get('weather_code'), 'n/a')}, humidity {Cur.get('relative_humidity_2m', '?')}%, "
                 f"wind {Cur.get('wind_speed_10m', '?')} mph"]
        for i in range(1, min(4, len(Days))):
            try:
                Lines.append(f"{Days[i]}: {Daily['temperature_2m_max'][i]}F / {Daily['temperature_2m_min'][i]}F, "
                             f"rain {Daily['precipitation_probability_max'][i]}%")
            except Exception:
                break
        return "\n".join(Lines)

    def _Wikipedia(self, query, max_results=3):
        """Wikipedia search + lead-section extract (free MediaWiki API, no key)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        import json as _json, re as _re
        Limit = max(1, min(int(max_results), 5))
        UA = {"User-Agent": "AE-agent/1.0 (contact: local)"}

        def _Get(Url):
            Req = request.Request(Url, headers=UA, method="GET")
            with request.urlopen(Req, timeout=25) as Resp:
                return _json.loads(Resp.read(500_000).decode("utf-8", errors="replace"))

        Found = _Get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
                     + parse.quote(query.strip()) + "&srlimit=" + str(Limit) + "&format=json")
        Hits = ((Found.get("query") or {}).get("search") or [])
        if not Hits:
            return f"No Wikipedia articles found for '{query.strip()}'."
        Titles = [h.get("title", "") for h in Hits if h.get("title")]
        Detail = _Get("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles="
                      + parse.quote("|".join(Titles[:2])) + "&format=json")
        Pages = ((Detail.get("query") or {}).get("pages") or {}).values()
        Texts = {p.get("title", ""): (p.get("extract", "") or "")[:1200] for p in Pages}
        Lines = ["## Wikipedia"]
        for T in Titles:
            Lines.append(f"- **{T}**: {(Texts.get(T, '') or 'no summary')[:600]}")
        return "\n".join(Lines)

    def _StackSearch(self, query, max_results=5):
        """Stack Overflow search (free Stack Exchange API, no key, 300 req/day)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        import json as _json, re as _re, html as _html
        Limit = max(1, min(int(max_results), 10))
        Url = ("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q="
               + parse.quote(query.strip()) + "&site=stackoverflow&pagesize=" + str(Limit) + "&filter=withbody")
        Req = request.Request(Url, headers={"User-Agent": "AE-agent/1.0"}, method="GET")
        with request.urlopen(Req, timeout=25) as Resp:
            Data = _json.loads(Resp.read(500_000).decode("utf-8", errors="replace"))
        Items = Data.get("items") or []
        if not Items:
            return f"No Stack Overflow questions found for '{query.strip()}'."
        Lines = ["## Stack Overflow"]
        for It in Items:
            Title = _html.unescape(str(It.get("title", "untitled")))
            Body = _re.sub(r"<[^>]+>", " ", str(It.get("body", "")))
            Body = _html.unescape(_re.sub(r"\s+", " ", Body)).strip()[:400]
            Tags = ",".join(It.get("tags", [])[:4])
            Lines.append(f"- **{Title}** (score {It.get('score', '?')}, answers {It.get('answer_count', '?')}"
                         + (", accepted" if It.get("is_answered") else "") + (f" [{Tags}]" if Tags else "") + ")")
            Lines.append(f"  {It.get('link', '')}")
            if Body:
                Lines.append(f"  {Body}")
        return "\n".join(Lines)

    def _NewsSearch(self, query, max_results=5):
        """World news search via GDELT (free, no key). Current events, coverage across outlets."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        import json as _json
        Limit = max(1, min(int(max_results), 10))
        Url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" + parse.quote(query.strip())
               + "&mode=artlist&maxrecords=" + str(Limit) + "&format=json")
        Req = request.Request(Url, headers={"User-Agent": "AE-agent/1.0"}, method="GET")
        try:
            with request.urlopen(Req, timeout=25) as Resp:
                Data = _json.loads(Resp.read(500_000).decode("utf-8", errors="replace"))
        except Exception as Ex:
            return f"News search temporarily unavailable ({Ex}). Retry once later, or use web_search instead."
        Arts = Data.get("articles") or []
        if not Arts:
            return f"No news found for '{query.strip()}'."
        Lines = ["## News (GDELT)"]
        for A in Arts:
            Lines.append(f"- **{A.get('title', 'untitled')}** ({A.get('sourcecountry', '?')} · {A.get('seendate', '')[:8]})")
            Lines.append(f"  {A.get('url', '')}")
        return "\n".join(Lines)

    def _ImageSearch(self, query, max_results=5):
        """Image URL search via the free ddgs library (no key)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        Limit = max(1, min(int(max_results), 10))
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                Raw = list(ddgs.images(query.strip(), max_results=Limit))
        except ImportError:
            raise RuntimeError("Image search needs the ddgs package (pip install ddgs).")
        except Exception as Ex:
            raise RuntimeError(f"Image search failed: {Ex}") from Ex
        if not Raw:
            return f"No images found for '{query.strip()}'."
        Lines = ["## Image results (direct URLs plus source pages)"]
        for i, r in enumerate(Raw, 1):
            Lines.append(f"{i}. {r.get('title', 'untitled')[:100]}")
            Lines.append(f"   image: {r.get('image', '')}")
            Lines.append(f"   page: {r.get('url', '')}")
        return "\n".join(Lines)

    def _SearchOne(self, provider, query, limit):
        """One provider -> [{title, href, body}]. Raises RuntimeError on missing key/failure."""
        provider = (provider or "duckduckgo").strip().lower()
        if provider == "duckduckgo":
            try:
                from ddgs import DDGS
                try:
                    raw = list(DDGS().text(query.strip(), max_results=limit))
                except Exception:
                    # Transient DDG failure (rate limit / network): one retry, then fallback.
                    import time as _time
                    _time.sleep(2)
                    try:
                        raw = list(DDGS().text(query.strip(), max_results=limit))
                    except Exception:
                        raw = []
                if not raw:
                    # Empty or still failing: stdlib HTML fallback instead of an error.
                    return self._DDGFallbackSearch(query.strip(), limit)
                return [{"title": r.get("title", "No title"),
                         "href": r.get("href", r.get("url", "")),
                         "body": r.get("body", r.get("snippet", "")) or ""} for r in raw]
            except ImportError:
                return self._DDGFallbackSearch(query.strip(), limit)
        if provider == "brave":
            key = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_KEY")
            if not key:
                raise RuntimeError("brave needs BRAVE_API_KEY (not set). Usable now: duckduckgo. Do NOT cycle other providers — they need keys too.")
            url = "https://api.search.brave.com/res/v1/web/search?q=" + parse.quote(query) + f"&count={limit}"
            req = request.Request(url, headers={"Accept": "application/json", "X-Subscription-Token": key}, method="GET")
            with request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [{"title": r.get("title", "No title"), "href": r.get("url", ""),
                     "body": r.get("description", "") or ""} for r in (data.get("web", {}) or {}).get("results", [])]
        if provider == "tavily":
            key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_KEY")
            if not key:
                raise RuntimeError("tavily needs TAVILY_API_KEY (not set). Usable now: duckduckgo. Do NOT cycle other providers — they need keys too.")
            payload = json.dumps({"query": query, "max_results": limit, "include_answer": False}).encode("utf-8")
            req = request.Request("https://api.tavily.com/search", data=payload,
                                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
            with request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [{"title": r.get("title", "No title"), "href": r.get("url", ""),
                     "body": r.get("content", "") or ""} for r in data.get("results", [])]
        if provider == "exa":
            key = os.getenv("EXA_API_KEY") or os.getenv("EXA_KEY")
            if not key:
                raise RuntimeError("exa needs EXA_API_KEY (not set). Usable now: duckduckgo. Do NOT cycle other providers — they need keys too.")
            payload = json.dumps({"query": query, "numResults": limit,
                                    "contents": {"text": {"maxCharacters": 800}}}).encode("utf-8")
            req = request.Request("https://api.exa.ai/search", data=payload,
                                  headers={"Content-Type": "application/json", "x-api-key": key}, method="POST")
            with request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = []
            for r in data.get("results", []):
                text = r.get("text", "") or ""
                if isinstance(text, str) and len(text) > 800:
                    text = text[:800] + "..."
                out.append({"title": r.get("title", "No title"), "href": r.get("url", ""), "body": text})
            return out
        if provider == "perplexity":
            key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
            if not key:
                raise RuntimeError("perplexity needs PERPLEXITY_API_KEY (not set). Usable now: duckduckgo. Do NOT cycle other providers — they need keys too.")
            model = os.getenv("PERPLEXITY_MODEL", "sonar")
            payload = json.dumps({"model": model, "messages": [
                {"role": "user", "content": f"Search the web for: {query}. Reply with the key facts and cite source URLs inline."}]}).encode("utf-8")
            req = request.Request("https://api.perplexity.ai/chat/completions", data=payload,
                                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
            with request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = ""
            return [{"title": "Perplexity answer", "href": "", "body": content or ""}]
        if provider == "jina":
            key = os.getenv("JINA_API_KEY") or os.getenv("JINA_KEY")
            if not key:
                raise RuntimeError("jina search needs JINA_API_KEY (not set). Usable now: duckduckgo. Do NOT cycle other providers — they need keys too.")
            url = "https://s.jina.ai/?q=" + parse.quote(query.strip())
            headers = {"Accept": "text/plain", "X-Return-Format": "markdown",
                       "Authorization": f"Bearer {key}"}
            req = request.Request(url, headers=headers, method="GET")
            with request.urlopen(req, timeout=45) as resp:
                text = resp.read(200_000).decode("utf-8", errors="replace")
            import re as _re
            results, cur = [], {"title": "", "href": "", "body": ""}
            for line in text.splitlines():
                m = _re.match(r"^(Title|URL(?: Source)?):\s*(.*)$", line.strip())
                if m:
                    if m.group(1).lower().startswith("title"):
                        if cur["title"] or cur["href"]:
                            results.append(cur)
                            cur = {"title": "", "href": "", "body": ""}
                        cur["title"] = m.group(2)
                    else:
                        cur["href"] = m.group(2)
                elif line.strip():
                    cur["body"] += line.strip() + " "
                if len(results) >= limit and not cur["title"]:
                    break
            if cur["title"] or cur["href"]:
                results.append(cur)
            if not results:
                results = [{"title": "Jina search", "href": "", "body": text[:2000]}]
            return results[:limit]
        if provider == "searxng":
            base = (os.getenv("SEARXNG_URL") or "").rstrip("/")
            if not base:
                raise RuntimeError("searxng needs SEARXNG_URL (not set). Usable now: duckduckgo. Do NOT cycle other providers — they need keys too.")
            url = base + "/search?q=" + parse.quote(query) + "&format=json"
            req = request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, method="GET")
            with request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [{"title": r.get("title", "No title"), "href": r.get("url", ""),
                     "body": r.get("content", "") or ""} for r in data.get("results", [])][:limit]
        raise ValueError(
            f"Unknown search provider '{provider}'. Supported: duckduckgo, brave, tavily, exa, "
            f"perplexity, jina, searxng. (ollama/kimi/kaga are models, not search APIs; "
            f"mojeek/ecosia/startpage need custom scraping and are not built in.)")

    def _WorkingSearchProviders(self):
        """Providers usable RIGHT NOW (key present, or keyless). Models never pick dead ones."""
        Working = ["duckduckgo"]
        if os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_KEY"):
            Working.append("brave")
        if os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_KEY"):
            Working.append("tavily")
        if os.getenv("EXA_API_KEY") or os.getenv("EXA_KEY"):
            Working.append("exa")
        if os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY"):
            Working.append("perplexity")
        if os.getenv("JINA_API_KEY") or os.getenv("JINA_KEY"):
            Working.append("jina")
        if (os.getenv("SEARXNG_URL") or "").strip():
            Working.append("searxng")
        return Working

    def _WebSearch(self, query, max_results=5, fetch_full=False, provider=None, providers=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        Limit = max(1, min(int(max_results), 10))
        # Auto default: local SearXNG when configured (multi-engine, free),
        # else duckduckgo. Explicit provider/providers always win.
        _HasSearx = bool((os.getenv("SEARXNG_URL") or "").strip())
        if providers:
            Wanted = [p.strip().lower() for p in providers if str(p).strip()]
        elif provider:
            Wanted = [str(provider).strip().lower()]
        else:
            Wanted = ["searxng" if _HasSearx else "duckduckgo"]
        if not Wanted:
            Wanted = ["searxng" if _HasSearx else "duckduckgo"]
        # Dedupe, preserve order.
        Seen, Plan = set(), []
        for P in Wanted:
            if P not in Seen:
                Seen.add(P)
                Plan.append(P)
        # Auto-filter dead providers: keyed backends without keys are dropped
        # (not errors) so the model never has to choose or cycle. Fall back
        # to duckduckgo rather than failing the whole call.
        Working = self._WorkingSearchProviders()
        Dropped = [P for P in Plan if P not in Working]
        Plan = [P for P in Plan if P in Working]
        Notes, Merged, SeenUrls = [], [], set()
        if Dropped:
            Notes.append(f"[providers unavailable (no key), skipped: {', '.join(Dropped)} — used {', '.join(Plan) if Plan else 'duckduckgo'}]")
        if not Plan:
            Plan = ["duckduckgo"]

        def _RunOne(P):
            try:
                return P, self._SearchOne(P, query.strip(), Limit), ""
            except ValueError as Ex:
                raise
            except Exception as Ex:
                return P, [], str(Ex)

        if len(Plan) == 1:
            P = Plan[0]
            try:
                Per, Note = self._SearchOne(P, query.strip(), Limit), ""
            except ValueError:
                raise
            except Exception as Ex:
                # Auto-chosen backend dead (e.g. SEARXNG_URL set but the
                # instance is down): degrade to duckduckgo, never fail bare.
                if P != "duckduckgo":
                    try:
                        Per = self._SearchOne("duckduckgo", query.strip(), Limit)
                        Notes.append(f"[{P} unreachable ({Ex}). Fell back to duckduckgo.]")
                    except Exception as Ex2:
                        raise RuntimeError(f"{P} search failed: {Ex}; fallback also failed: {Ex2}") from Ex2
                else:
                    raise RuntimeError(f"{P} search failed: {Ex}") from Ex
            Merged = [r for r in Per if r.get("href") not in SeenUrls or not r.get("href")]
            for r in Merged:
                if r.get("href"):
                    SeenUrls.add(r["href"])
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(Plan))) as Executor:
                for P, Per, Note in Executor.map(_RunOne, Plan):
                    if Note and "Unknown search provider" in Note:
                        raise ValueError(Note)
                    if Note and not Per:
                        Notes.append(f"[{P} skipped: {Note}]")
                    for r in Per:
                        Href = r.get("href", "")
                        if Href and Href in SeenUrls:
                            continue
                        if Href:
                            SeenUrls.add(Href)
                        Merged.append(r)
        results = Merged
        if not results:
            Detail = (" " + " ".join(Notes)) if Notes else ""
            return f"No results found.{Detail} Set provider keys (BRAVE_API_KEY, TAVILY_API_KEY, EXA_API_KEY, PERPLEXITY_API_KEY) or SEARXNG_URL to enable more providers."

        # Full snippet bodies (DDG gives ~1-2 paragraphs per result — keep all of it,
        # capped per-result so one page can't eat the whole context window).
        lines = ["## Web Search Results", ""]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href = r.get("href", r.get("url", ""))
            body = (r.get("body", r.get("snippet", "")) or "").strip()
            if len(body) > 800:
                body = body[:800] + "..."
            lines.append(f"{i}. **{title}**")
            lines.append(f"   URL: {href}")
            lines.append(f"   {body}" if body else "   (no snippet)")
            lines.append("")
        if fetch_full:
            # Fetch the top result's full page content so the agent has ACTUAL
            # facts (weather, time, prices) instead of just headlines.
            top = (results[0].get("href", results[0].get("url", "")) or "").strip()
            if top:
                try:
                    full = self._WebFetch(top)
                    lines.append(f"--- Full content of top result ({top}) ---")
                    lines.append(full[:6000] + ("\n[page truncated]" if len(full) > 6000 else ""))
                except Exception as Ex:
                    lines.append(f"[Could not fetch top result: {Ex}]")
        lines.append("Tip: use web_fetch(url) for the full content of any result above.")
        if Notes:
            lines.append("Skipped providers: " + "; ".join(Notes))
        out = "\n".join(lines)
        return out[:12000] + ("\n[results truncated]" if len(out) > 12000 else "")

    def _FetchOneUrl(self, url):
        """Single Jina-reader fetch (shared by web_fetch and fetch_many)."""
        Parsed = parse.urlparse(url)
        if Parsed.scheme not in {"http", "https"} or not Parsed.netloc:
            raise ValueError("url must be a public http or https URL.")
        Host = (Parsed.hostname or Parsed.netloc).casefold()
        BlockedHosts = getattr(self, "_BlockedFetchHosts", {})
        if Host in BlockedHosts:
            raise RuntimeError(f"Jina fetch was already blocked for {Host} this turn ({BlockedHosts[Host]}); use another source.")
        ApiKey = os.getenv("JINA_API_KEY") or os.getenv("JINA_KEY")
        Headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
        if ApiKey:
            Headers["Authorization"] = f"Bearer {ApiKey}"
        Req = request.Request("https://r.jina.ai/" + url, headers=Headers, method="GET")
        try:
            with request.urlopen(Req, timeout=45) as Response:
                Content = Response.read(1_000_001).decode("utf-8", errors="replace")
        except error.HTTPError as Ex:
            Body = Ex.read(2000).decode("utf-8", errors="replace")
            if Ex.code in (403, 451):
                Message = f"Jina fetch failed ({Ex.code}) for {Host}: this domain blocks anonymous reads — use another source, do NOT retry it this turn. ({Body[:200]})"
                try:
                    self._BlockedFetchHosts[Host] = Ex.code
                except Exception:
                    pass
                raise RuntimeError(Message) from Ex
            raise RuntimeError(f"Jina fetch failed ({Ex.code}): {Body}") from Ex
        if len(Content) > 1_000_000:
            Content = Content[:1_000_000] + "\n[content truncated]"
        Content = Content[:20000] + ("\n[content truncated]" if len(Content) > 20000 else "")
        # Turn-level fetch cache: identical URLs re-requested later this turn
        # (29-fetch loops) are served from here, never re-downloaded.
        try:
            _Cache = getattr(self, "_FetchedURLs", None)
            if _Cache is not None:
                _Cache[str(url).strip().lower()] = Content
        except Exception:
            pass
        return Content

    def _WebFetch(self, url):
        return self._FetchOneUrl(url)

    def _FetchMany(self, urls, **kwargs):
        """Fetch up to 10 URLs in parallel threads; per-URL errors stay in-band."""
        if not isinstance(urls, list) or not urls:
            raise ValueError("urls must be a non-empty array.")
        # Dedupe: models re-send the same URL twice in one batch — fetch once.
        _Seen, _Deduped = set(), []
        for _U in urls:
            _K = str(_U).strip().lower() if isinstance(_U, str) else _U
            if _K not in _Seen:
                _Seen.add(_K)
                _Deduped.append(_U)
        urls = _Deduped
        if len(urls) > 10:
            raise ValueError("fetch_many accepts at most 10 URLs per call; split into two calls.")
        for Url in urls:
            if not isinstance(Url, str) or not Url.strip():
                raise ValueError("Every url must be a non-empty string.")

        def FetchOne(Url):
            try:
                _Cache = getattr(self, "_FetchedURLs", None) or {}
                _Hit = _Cache.get(str(Url).strip().lower())
                if _Hit is not None:
                    return {"url": Url, "text": str(_Hit) + "\n[cached — fetched earlier this turn, not re-downloaded]"}
                return {"url": Url, "text": self._FetchOneUrl(Url)}
            except Exception as Ex:
                return {"url": Url, "error": str(Ex)[:300]}

        with ThreadPoolExecutor(max_workers=min(5, len(urls))) as Pool:
            return list(Pool.map(FetchOne, urls))

    def _TextToSpeech(self, text, voice_id=None, model_id=None, output_path=None):
        ApiKey = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_KEY")
        if not ApiKey:
            raise RuntimeError("Set ELEVENLABS_API_KEY to enable text-to-speech.")
        VoiceId = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        if not VoiceId:
            raise RuntimeError("Set ELEVENLABS_VOICE_ID or pass voice_id to select a voice.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty.")

        Filename = output_path or f"generated/audio/speech-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.mp3"
        AudioPath = self._ResolveProjectPath(Filename)
        if AudioPath.suffix.lower() != ".mp3":
            raise ValueError("output_path must use the .mp3 extension.")
        ModelId = model_id or "eleven_v4_turbo"
        Url = f"https://api.elevenlabs.io/v1/text-to-speech/{parse.quote(VoiceId, safe='')}?output_format=mp3_44100_128"
        Payload = json.dumps({"text": text, "model_id": ModelId}).encode("utf-8")
        Req = request.Request(Url, data=Payload, headers={"xi-api-key": ApiKey, "Content-Type": "application/json", "Accept": "audio/mpeg"}, method="POST")
        try:
            with request.urlopen(Req, timeout=90) as Response:
                Audio = Response.read(50_000_001)
        except error.HTTPError as Ex:
            Body = Ex.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(f"ElevenLabs API error ({Ex.code}): {Body}") from Ex
        if len(Audio) > 50_000_000:
            raise ValueError("Generated audio exceeded the 50 MB output limit.")
        AudioPath.parent.mkdir(parents=True, exist_ok=True)
        AudioPath.write_bytes(Audio)
        return {"path": AudioPath.relative_to(self.ProjectRoot).as_posix(), "bytes": len(Audio), "model_id": ModelId}

    def _PatchFile(self, path, old_text, new_text):
        FilePath = self._ResolveProjectPath(path)
        if not FilePath.is_file():
            raise ValueError(f"File not found: {path}")
        Content = FilePath.read_text(encoding="utf-8")
        Count = Content.count(old_text)
        if Count != 1:
            raise ValueError(f"Patch expected one exact match, found {Count}; read the current file and narrow the patch.")
        self._SnapshotFile(FilePath)
        FilePath.write_text(Content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Patched {FilePath.relative_to(self.ProjectRoot).as_posix()}"

    def AgentInterpreter(self):
        """Absolute path of the interpreter that runs code_exec/dependency_install.

        Shell commands must resolve `python`/`pip` to THIS file, otherwise a
        `pip install x` lands in a different environment than the one the agent
        imports from, and the install appears to have no effect.
        """
        return sys.executable

    def AgentShellEnv(self):
        """env for shell calls: the agent interpreter's bin dir first on PATH.

        Windows: a venv keeps python.exe/pip.exe together in Scripts\\.
        POSIX: python3/pip live beside the interpreter in bin/.
        """
        Env = os.environ.copy()
        InterpDir = str(Path(sys.executable).resolve().parent)
        Current = Env.get("PATH", "")
        Env["PATH"] = InterpDir + os.pathsep + Current if Current else InterpDir
        # Let child shells/tools find the interpreter by its bare name too.
        Env["AE_PYTHON"] = sys.executable
        return Env

    def _RunTerminal(self, command):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string.")
        if getattr(self, "Sandbox", None) is not None:
            self.Sandbox.check_command(command)
        Env = self.AgentShellEnv()
        if os.name == "nt":
            Shell = shutil.which("pwsh") or shutil.which("powershell") or os.environ.get("COMSPEC", "cmd.exe")
            Arguments = [Shell, "-NoProfile", "-Command", command] if Path(Shell).name.lower().startswith("pwsh") or "powershell" in Path(Shell).name.lower() else [Shell, "/d", "/s", "/c", command]
        else:
            Arguments = ["/bin/sh", "-lc", command]
        Result = subprocess.run(Arguments, cwd=self.ProjectRoot, capture_output=True, text=True, timeout=120, check=False, shell=False, env=Env)
        Output = (Result.stdout + Result.stderr).strip()
        if len(Output) > 12000:
            Output = Output[:12000] + "\n[output truncated]"
        return f"Exit code: {Result.returncode}\n{Output}"

    def _RunBash(self, command):
        if getattr(self, "Sandbox", None) is not None:
            self.Sandbox.check_command(command)
        Bash = shutil.which("bash")
        if not Bash:
            raise RuntimeError("Bash is not installed or not available on PATH. Use terminal on this system.")
        Result = subprocess.run([Bash, "-lc", command], cwd=self.ProjectRoot, capture_output=True, text=True, timeout=120, check=False, shell=False, env=self.AgentShellEnv())
        Output = (Result.stdout + Result.stderr).strip()
        if len(Output) > 12000:
            Output = Output[:12000] + "\n[output truncated]"
        return f"Exit code: {Result.returncode}\n{Output}"

    def _RetainMemory(self, text, category="learned_lesson"):
        if self.Memory is None:
            raise RuntimeError("Memory is not enabled.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        Category = str(category or "learned_lesson").strip()
        Category = {"lesson": "learned_lesson", "fact": "user_fact", "project fact": "project_fact", "user fact": "user_fact"}.get(Category.casefold(), Category)
        Categories = {"user_fact", "preference", "instruction", "project_fact", "learned_lesson"}
        if Category not in Categories:
            return {"stored": False, "reason": f"Unsupported category '{Category}'. Use one of: {', '.join(sorted(Categories))}."}
        if getattr(self, "TurnMemoryRetainBlocked", False):
            return {"stored": False, "reason": "A non-durable memory write was already declined this turn. Keep task output in the current response; do not retry memory_retain."}
        Request = str(getattr(self, "CurrentUserRequest", "") or "")
        ExplicitRemember = bool(re.search(r"\b(?:remember|retain|store|save)\b.{0,60}\b(?:this|that|for (?:later|future)|in memory|permanently|my|me|it)?\b", Request, re.I))
        DurablePreference = Category == "preference" and bool(re.search(r"\b(?:i prefer|i like|i dislike|from now on|always|never)\b", Request, re.I))
        DurableUserFact = Category == "user_fact" and bool(re.search(r"\b(?:i am|i'm|my (?:name|role|timezone|language|lucky number|job|location))\b", Request, re.I))
        DurableInstruction = Category == "instruction" and bool(re.search(r"\b(?:from now on|always|never|do not|don't)\b", Request, re.I))
        StableProjectFact = Category == "project_fact" and bool(re.search(r"\b(?:repository|repo|workspace|codebase|project architecture|configured|installed|uses? (?:python|node|react|django|flask|sqlite|postgres|mongodb))\b", text, re.I))
        ReusableLesson = Category == "learned_lesson" and len(text.strip()) <= 300 and bool(re.search(r"\b(?:fail(?:ed|ure)?|error|never|always|instead|because|lesson|avoid|prefer)\b", text, re.I))
        if not (ExplicitRemember or DurablePreference or DurableUserFact or DurableInstruction or StableProjectFact or ReusableLesson):
            self.TurnMemoryRetainBlocked = True
            return {"stored": False, "reason": "This is temporary task/educational content, not durable memory. Keep it in the current response and do not retry memory_retain this turn."}
        Entry = self.Memory.AddLongTerm(text.strip(), {"type": category, "source": "agent"})
        return {"stored": True, "id": Entry["id"], "text": Entry["text"]}

    def _RecallMemory(self, query, limit=5):
        if self.Memory is None:
            raise RuntimeError("Memory is not enabled.")
        return self.Memory.SearchLongTerm(query, Limit=max(1, min(int(limit), 10)))

    def _ExecEnter(self, name, kind):
        """Enter a tracked .ae execution. Nested executions (tool calling a
        skill) reuse the outer record so stop/report always hit one id;
        the record drops when the outermost frame exits."""
        try:
            stack = list(getattr(self._local, "exec_stack", None) or [])
        except Exception:
            stack = []
        if stack:
            Eid = stack[-1]
        else:
            with self._lock:
                self._ExecSeq += 1
                Eid = f"ex_{self._ExecSeq:04d}"
                self.RunningExecutions[Eid] = {"id": Eid, "name": name, "kind": kind,
                    "started": time.time(), "progress": "", "stop": threading.Event()}
        try:
            self._local.exec_stack = [*stack, Eid]
        except Exception:
            pass
        return Eid

    def _ExecExit(self, Eid):
        try:
            stack = list(getattr(self._local, "exec_stack", None) or [])
            if not (stack and stack[-1] == Eid):
                return
            stack = stack[:-1]
            self._local.exec_stack = stack
        except Exception:
            return
        if not stack:
            with self._lock:
                self.RunningExecutions.pop(Eid, None)

    # Back-compat aliases (flat tracking; prefer Enter/Exit for nesting).
    def _ExecBegin(self, name, kind):
        return self._ExecEnter(name, kind)

    def _ExecEnd(self, Eid):
        return self._ExecExit(Eid)

    def _ExecStopRequested(self, Eid):
        try:
            with self._lock:
                Rec = self.RunningExecutions.get(Eid)
                return bool(Rec and Rec["stop"].is_set())
        except Exception:
            return False

    def _ExecReport(self, Eid, Data):
        try:
            Text = Data if isinstance(Data, str) else json.dumps(Data, ensure_ascii=False, default=str)
            with self._lock:
                Rec = self.RunningExecutions.get(Eid)
                if Rec is not None:
                    Rec["progress"] = Text[:500]
        except Exception:
            pass

    def _ListExecutions(self, **kwargs):
        Now = time.time()
        with self._lock:
            Recs = list(self.RunningExecutions.values())
        return [{"id": r["id"], "name": r["name"], "kind": r["kind"],
                 "elapsed_seconds": round(Now - r["started"], 1),
                 "progress": r["progress"], "stop_requested": r["stop"].is_set()} for r in Recs]

    def _StopExecution(self, id, **kwargs):
        with self._lock:
            Rec = self.RunningExecutions.get((id or "").strip())
            if Rec is None:
                raise ValueError(f"No running execution: {id}. Check `executions` for live ids.")
            Rec["stop"].set()
            return {"stopped": Rec["id"], "name": Rec["name"]}

    def ToolHistory(self, limit=30):
        """Recent tool usage for .ae reasoning skills: [{age_seconds, tool, skill, error}]."""
        try:
            limit = max(1, min(int(limit), 200))
        except Exception:
            limit = 30
        now = time.time()
        out = []
        for ts, tool, skill, err in list(self.CallLog)[-limit:]:
            try:
                out.append({"age_seconds": round(now - float(ts), 1), "tool": str(tool),
                            "skill": str(skill or ""), "error": bool(err)})
            except Exception:
                continue
        return out

    def _KnowledgeQuery(self, query, limit=5):
        """Search inactive knowledge. First try semantic search over long-term
        memory; fall back to naive long-term search. Also indexes inactive .ae
        knowledge files (BM25-lite via TF-IDF over their body text)."""
        Results = []
        try:
            if __package__:
                from .memory.semantic import search as semantic_search
            else:
                from memory.semantic import search as semantic_search
        except Exception:
            semantic_search = None

        # Inactive knowledge from .ae files (local → shared → global)
        try:
            from AEX import AEXScript, AEXError
            from pathlib import Path
            Entries = []
            for Type in ("knowledge",):
                for Directory, Scope in self._VisibleRoots(Type):
                    if not Directory.is_dir():
                        continue
                    for AEXFile in sorted(Directory.rglob("*.ae")):
                        try:
                            Script = AEXScript.FromFile(AEXFile)
                        except (AEXError, OSError):
                            continue
                        if Script.Type != "knowledge":
                            continue
                        if Script.Active:
                            continue  # inactive only
                        Body = (Script.Body or "").strip()
                        if not Body:
                            continue
                        Text = Body
                        Entries.append({
                            "text": Text,
                            "metadata": {
                                "name": Script.Name,
                                "type": Script.Type,
                                "category": Script.Category,
                                "tags": Script.Tags,
                                "scope": Scope,
                                "path": str(Script.SourcePath),
                            },
                        })
            if Entries and semantic_search:
                Results.extend(semantic_search(Entries, query, limit=10))
            elif Entries:
                ql = (query or "").lower()
                for E in Entries:
                    if ql and ql in E["text"].lower():
                        Results.append({**E, "score": 0.8, "method": "substring"})
        except Exception:
            pass

        # Long-term memory (existing behavior)
        if self.Memory is not None:
            try:
                if semantic_search:
                    MEntries = [{"text": e.get("text", ""), "metadata": e.get("metadata", {})}
                                for e in (getattr(self.Memory, "LongTerm", []) or [])]
                    if MEntries:
                        Results.extend(semantic_search(MEntries, query, limit=10))
                else:
                    Results.extend(self.Memory.SearchLongTerm(query, Limit=max(1, min(int(limit), 10))))
            except Exception:
                pass

        # Deduplicate by text (keep highest score)
        Seen = {}
        for R in Results:
            Key = R.get("text", "")[:80]
            if Key not in Seen or R.get("score", 0) > Seen[Key].get("score", 0):
                Seen[Key] = R
        Out = sorted(Seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return Out[: max(1, min(int(limit), 10))]

    def _CapabilitySearch(self, query, limit=8):
        """Runtime capability discovery: match query against tool/skill/exe names+descriptions."""
        try:
            limit = max(1, min(int(limit), 20))
        except Exception:
            limit = 8
        q = (query or "").lower()
        toks = [t for t in __import__("re").findall(r"[a-z0-9_]+", q) if t]
        cands = []
        try:
            for key, t in self.Tools.items():
                cands.append({"kind": "tool", "name": t.get("name", key),
                              "description": t.get("description", "")})
        except Exception:
            pass
        try:
            for s in self.GetSkills():
                cands.append({"kind": "skill", "name": s.get("name", ""),
                              "description": s.get("description", "")})
        except Exception:
            pass
        try:
            inv = self._AeList() if hasattr(self, "_AeList") else {}
            for e in (inv.get("executables", []) or []):
                cands.append({"kind": "executable", "name": e.get("name", ""),
                              "description": e.get("description", "")})
        except Exception:
            pass
        scored = []
        for c in cands:
            hay = (c["name"] + " " + c["description"]).lower()
            score = sum(2 if t == c["name"].lower() else (1 if t in hay else 0) for t in toks)
            if score > 0:
                scored.append({**c, "score": score})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]

    def _ForgetMemory(self, memory_id_or_text):
        if self.Memory is None:
            raise RuntimeError("Memory is not enabled.")
        return {"forgotten": self.Memory.ForgetLongTerm(memory_id_or_text)}

    def _ManageTodos(self, action, task_id=None, title=None, description=None, status=None, priority=None, assignee=None, parent=None, plan=None, tasks=None):
        if self.Memory is None:
            raise RuntimeError("Shared memory is not enabled.")
        # Alias leniency: models invent tasks= for plan= — accept it.
        if plan is None and tasks is not None:
            plan = tasks
        # Subagent threads stamp their todos with their delegation id so the
        # board can render per-worker lanes from the shared store.
        try:
            Dlg = getattr(getattr(self, "_local", None), "dlg_id", None)
        except Exception:
            Dlg = None
        return self.Memory.ManageTodos(
            action,
            TaskId=task_id,
            Title=title,
            Description=description,
            Status=status,
            Priority=priority,
            Assignee=assignee,
            Parent=parent,
            Plan=plan,
            Dlg=Dlg,
        )

    def _SnapshotFile(self, FilePath):
        """Remember pre-write state for undo (cap 25)."""
        try:
            rel = FilePath.relative_to(self.ProjectRoot).as_posix()
        except ValueError:
            return
        old = FilePath.read_text(encoding="utf-8") if FilePath.is_file() else None
        self._Snapshots.append((rel, old))
        del self._Snapshots[:-25]

    def _UndoLast(self, steps=1):
        try:
            count = max(1, min(int(steps or 1), 25))
        except (TypeError, ValueError):
            count = 1
        if not self._Snapshots:
            return "Nothing to undo: no tracked file changes this session."
        restored = []
        for _ in range(min(count, len(self._Snapshots))):
            rel, old = self._Snapshots.pop()
            target = (self.ProjectRoot / rel).resolve()
            if target != self.ProjectRoot and self.ProjectRoot not in target.parents:
                continue
            if old is None:
                if target.is_file():
                    target.unlink()
                restored.append(f"{rel} (removed, was new)")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(old, encoding="utf-8")
                restored.append(rel)
        return "Undone: " + ", ".join(restored) if restored else "Nothing to undo."

    def _WriteFile(self, path, content):
        # Binary formats can't be written as text: refuse with a redirect.
        _Ext = ("." + str(path).rsplit(".", 1)[-1].lower()) if "." in str(path) else ""
        if _Ext in {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".exe", ".mp3", ".mp4"}:
            raise ValueError(
                f"Refusing text write to *{_Ext} (would produce a corrupt file). "
                "Generate it with code_exec instead (reportlab/fpdf2 for PDF, Pillow for images), "
                "then verify with list_files.")
        # Safe hands: resolve inside workspace, snapshot for undo.
        if getattr(self, "Sandbox", None) is not None:
            try:
                FilePath = self.Sandbox.resolve(path)
            except ValueError:
                raise
        else:
            FilePath = self._ResolveProjectPath(path)
        FilePath.parent.mkdir(parents=True, exist_ok=True)
        self._SnapshotFile(FilePath)
        FilePath.write_text(content, encoding="utf-8")
        return f"Wrote {FilePath.relative_to(self.ProjectRoot).as_posix()}"

    def _ExecutableFiles(self):
        return [
            (AEXFile, Script)
            for AEXFile in self._AEXFiles(self.ExecutablesPath)
            for Script in [self._TryLoadAEX(AEXFile)]
            if Script is not None and Script.Type == "ae"
        ]

    def _TryLoadAEX(self, AEXFile):
        try:
            return AEXScript.FromFile(AEXFile)
        except (AEXError, OSError):
            return None

    def _ListExecutables(self):
        return [
            {"name": Script.Name, "description": Script.Description, "parameters": Script.Parameters}
            for _, Script in self._ExecutableFiles()
        ]

    def _LoadExecutable(self, name):
        Script = self._FindAEX(name, "ae", [self.ExecutablesPath])
        if Script is None:
            raise ValueError(f"AE executable not found: {name}")
        Content = Path(Script.SourcePath).read_text(encoding="utf-8")
        return {"name": Script.Name, "description": Script.Description, "parameters": Script.Parameters, "content": Content, "type": Script.Type}

    def _RunExecutable(self, name, parameters=None):
        Script = self._FindAEX(name, "ae", [self.ExecutablesPath])
        if Script is None:
            raise ValueError(f"AE executable not found: {name}")
        # run_executable itself is approval-gated; propagate inward (see Execute).
        _Eid = self._ExecBegin(name, "executable")
        try:
            return Script.Execute(parameters or {}, self, self.Memory, Approved=True)
        finally:
            self._ExecEnd(_Eid)

    def _ListEvents(self):
        Events = []
        for AEXFile in self._AEXFiles(self.EventsPath):
            Script = self._TryLoadAEX(AEXFile)
            if Script is not None and Script.Type == "event":
                Events.append({
                    "name": Script.Name,
                    "description": Script.Description,
                    "trigger": Script.Trigger,
                    "target_agent": Script.Metadata.get("target_agent", "*"),
                    "enabled": Script.Metadata.get("enabled", True),
                })
        return Events

    def _LoadEvent(self, name):
        Script = self._FindAEX(name, "event", [self.EventsPath])
        if Script is None:
            raise ValueError(f"Event not found: {name}")
        Content = Path(Script.SourcePath).read_text(encoding="utf-8")
        return {"name": Script.Name, "description": Script.Description, "trigger": Script.Trigger, "content": Content, "type": Script.Type}

    def _DelegateTask(self, goal, context="", target_agent=None, provider=None, model=None, max_tokens=None, toolsets=None, timeout_seconds=120, background=False, stance="neutral"):
        """Delegate a task to an isolated TempAgent (task-oriented lifecycle).
        Delegator-chosen toolsets restrict the copy. Volatile memory only, auto-cleanup on completion.
        background=True returns immediately with a delegation id (steer/check/stop); False blocks with panels.
        """
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string.")
        if stance not in {"neutral", "adversarial", "reviewer"}:
            raise ValueError("stance must be neutral, adversarial (red-team the context), or reviewer (verdict-first audit).")
        if provider and model:
            subagent_model = f"{provider}:{model}"
        elif target_agent:
            agent_folder = self.RepoRoot / "Agents" / target_agent
            if not agent_folder.is_dir():
                try:
                    Known = sorted(p.name for p in (self.RepoRoot / "Agents").iterdir() if p.is_dir() and (p / "agent.json").is_file())
                except Exception:
                    Known = []
                raise ValueError(f"Agent not found: '{target_agent}'. Available profiles: {', '.join(Known) or 'none'}. Pass target_agent with one of these exact names — not a role description.")
            info_file = agent_folder / "agent.json"
            if not info_file.exists():
                raise ValueError(f"Agent not found: {target_agent}")
            try:
                agent_config = json.loads(info_file.read_text(encoding="utf-8"))
                subagent_model = agent_config.get("model", "mistral:codestral-latest")
            except Exception:
                subagent_model = "mistral:codestral-latest"
        else:
            subagent_model = os.getenv("AE_DEFAULT_MODEL", "mistral:codestral-latest")
        if background:
            running = sum(1 for d in self.Delegations.values() if d.get("status") == "running")
            if running >= 4:
                raise ValueError(f"Delegation cap reached ({running} running, max 4). Wait for one with check_subagent or stop one with stop_subagent.")
            self._DelegationSeq += 1
            dlg_id = f"dlg_{self._DelegationSeq:04d}"
            rec = {"id": dlg_id, "goal": goal.strip()[:160], "model": subagent_model,
                "toolsets": toolsets or "all", "status": "running", "result": "",
                "elapsed": 0.0, "started": time.time(), "notified": True,
                "steer_box": [], "stop_flag": threading.Event(), "thread": None,
                "spawn": {"context": context or "", "max_tokens": max_tokens,
                    "timeout_seconds": timeout_seconds, "target_agent": target_agent, "stance": stance}}
            worker = TempAgent(goal.strip(), context or "", subagent_model, self, Toolsets=toolsets,
                MaxTokens=max_tokens, TimeoutSeconds=timeout_seconds, Background=True,
                DelegatorName=getattr(self, "CurrentAgentName", "*") or target_agent or "*",
                SteerBox=rec["steer_box"], StopFlag=rec["stop_flag"], Stance=stance)
            thread = threading.Thread(target=self._RunBackground, args=(dlg_id, worker), daemon=True)
            rec["thread"] = thread
            self.Delegations[dlg_id] = rec
            thread.start()
            try:
                self.start_dashboard()
            except Exception:
                pass
            # Prose receipt, not JSON: models echo JSON tool results back into later
            # calls' arguments (merging started/status/hint keys). Prose is read, not merged.
            scope = ", ".join(toolsets) if toolsets else "all tools"
            return (f"Background delegation {dlg_id} started (model {subagent_model}, scope {scope}). "
                f"Steer it: steer_subagent(id={dlg_id}, text=...). Collect it: check_subagent(id={dlg_id}). "
                f"Stop it: stop_subagent(id={dlg_id}). Its completion auto-prints.")
        worker = TempAgent(goal.strip(), context or "", subagent_model, self, Toolsets=toolsets, MaxTokens=max_tokens, TimeoutSeconds=timeout_seconds,
            DelegatorName=getattr(self, "CurrentAgentName", "*") or target_agent or "*", Stance=stance)
        import time as _time
        _t0 = _time.time()
        outcome = worker.run()
        elapsed = round(_time.time() - _t0, 1)
        result = outcome.get("result", "Subagent completed with no output.")
        self.LastDelegation = {"model": subagent_model, "elapsed": elapsed,
            "status": outcome.get("status", "error"), "toolsets": toolsets or "all",
            "goal": goal.strip()[:160], "tokens": getattr(worker, "LastUsage", {}) or {}}
        # Callback: alert delegator via shared memory tool-event log (no TempAgent persistence)
        if self.Memory is not None:
            try:
                self.Memory.AddToolEvent("delegate_task", {"goal": goal[:200], "model": subagent_model, "toolsets": toolsets or []}, str(result)[:1200], IsError=(outcome.get("status") != "done"))
            except Exception:
                pass
        # Automatic cleanup: worker holds only volatile state; drop reference
        del worker
        if outcome.get("status") != "done":
            return f"Subagent error:\n{result}"
        return f"Subagent result:\n{result}"

    def start_dashboard(self):
        """Persistent ticker: every 15s, one dim line while runners exist.
        Plain newline prints (input-safe) — no cursor fights with the prompt."""
        try:
            if self._DashboardThread is not None and self._DashboardThread.is_alive():
                return
        except Exception:
            pass

        def _loop():
            import time as _time
            while True:
                _time.sleep(15)
                try:
                    with self._lock:
                        Run = [(d.get("id", "?"), str(d.get("goal", ""))[:60], _time.time() - float(d.get("started") or _time.time()))
                               for d in list(self.Delegations.values()) if d.get("status") == "running"]
                    if not Run:
                        continue
                    from rich.console import Console as _Console
                    _bits = " · ".join(f"{_i} {round(_e)}s" for _i, _g, _e in Run)
                    _Console().print(f"[dim]workers {len(Run)}: {_bits} — results auto-print; .board for the full picture[/]")
                except Exception:
                    pass

        try:
            self._DashboardThread = threading.Thread(target=_loop, daemon=True)
            self._DashboardThread.start()
        except Exception:
            pass

    def _RunBackground(self, dlg_id, worker):
        rec = self.Delegations.get(dlg_id)
        try:
            self._local.dlg_id = dlg_id  # worker todos stamp this lane
        except Exception:
            pass
        try:
            outcome = worker.run()
        except Exception as Ex:
            outcome = {"status": "error", "result": f"Subagent crashed: {Ex}"}
        if rec is not None:
            rec["status"] = outcome.get("status", "error")
            rec["result"] = outcome.get("result", "")
            rec["elapsed"] = round(time.time() - rec.get("started", time.time()), 1)
            rec["notified"] = False
            rec["tokens"] = getattr(worker, "LastUsage", {}) or {}
            if self.Memory is not None:
                try:
                    with self._lock:
                        self.Memory.AddToolEvent("delegate_task", {"background": dlg_id, "goal": rec.get("goal", "")[:200]}, str(rec["result"])[:1200], IsError=(rec["status"] != "done"))
                except Exception:
                    pass
            # Fire subagent_complete triggers so event-driven follow-ups can react.
            try:
                self.DispatchEvent("subagent_complete", {"delegation_id": dlg_id, "status": rec["status"], "goal": rec.get("goal", ""), "result": str(rec["result"])[:2000]}, AgentName="*")
            except Exception:
                pass
            # Immediate surfacing: the main thread usually sits blocked on
            # input() here, so the input-loop fallback would print this only
            # after the user types. Print now (Rich print is thread-safe).
            try:
                with self._lock:
                    if not rec.get("notified"):
                        from rich.panel import Panel as _Panel
                        from rich.markdown import Markdown as _Markdown
                        from rich.console import Console as _Console
                        _Body = str(rec.get("result", ""))[:4000] or "(no output)"
                        _Border = "green" if rec.get("status") == "done" else "red"
                        _Console().print(_Panel(_Markdown(_Body), title="* Subagent " + str(rec.get("status")) + ": " + str(rec.get("goal", ""))[:80], border_style=_Border, subtitle="[dim]" + dlg_id + " · " + str(rec.get("model", "?")) + " · " + str(rec.get("elapsed", 0.0)) + "s — check_subagent " + dlg_id + "[/]"))
                        rec["notified"] = True
            except Exception:
                pass
            try:
                self._local.dlg_id = None
            except Exception:
                pass

    def _FindDelegation(self, dlg_id):
        rec = self.Delegations.get(dlg_id)
        if rec is None:
            # Leniency: models shorten dlg_0001 to dlg_1 — zero-pad and retry.
            import re as _re
            m = _re.fullmatch(r"(dlg_)(\d+)", str(dlg_id or "").strip())
            if m:
                rec = self.Delegations.get(f"{m.group(1)}{int(m.group(2)):04d}")
        if rec is None:
            raise ValueError(f"Unknown delegation: {dlg_id}. Use the delegations command to list running and finished work.")
        return rec

    def _SteerSubagent(self, id, text, **kwargs):
        rec = self._FindDelegation(id)
        if rec.get("status") != "running":
            raise ValueError(f"Delegation {id} already {rec.get('status')} — steering only reaches running workers.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty steering instruction.")
        rec["steer_box"].append(text.strip())
        return {"steered": id, "queued": len(rec["steer_box"])}

    def _CheckSubagent(self, id, **kwargs):
        rec = self._FindDelegation(id)
        out = {"id": id, "status": rec.get("status"), "model": rec.get("model"),
            "elapsed": rec.get("elapsed", 0.0), "goal": rec.get("goal", "")}
        if rec.get("status") != "running":
            out["result"] = rec.get("result", "")
        else:
            out["result"] = "(still running — steer with steer_subagent, stop with stop_subagent)"
        return out

    def _StopSubagent(self, id, **kwargs):
        rec = self._FindDelegation(id)
        if rec.get("status") != "running":
            return {"id": id, "status": rec.get("status"), "note": "already finished"}
        rec["stop_flag"].set()
        return {"id": id, "stopping": True, "note": "Stop requested; worker halts at its next checkpoint without disturbing siblings."}

    def _ReviveSubagent(self, id, extra_context="", **kwargs):
        rec = self._FindDelegation(id)
        if rec.get("status") == "running":
            raise ValueError(f"Delegation {id} is still running — steer it with steer_subagent or stop it first; revive is for finished work.")
        spawn = rec.get("spawn", {}) or {}
        prev = str(rec.get("result", ""))[:800]
        context = (spawn.get("context", "") + f"\n\n[Revive of {id}, which ended as {rec.get('status')}. Previous result: {prev}." + (f" Extra guidance: {extra_context.strip()}" if extra_context and extra_context.strip() else " Do it better this time.")) + "]"
        toolsets = rec.get("toolsets")
        if spawn.get("target_agent"):
            receipt = self._DelegateTask(rec.get("goal", ""), context=context,
                target_agent=spawn.get("target_agent"),
                max_tokens=spawn.get("max_tokens"), toolsets=None if toolsets in (None, "all") else toolsets,
                timeout_seconds=spawn.get("timeout_seconds", 120), background=True,
                stance=spawn.get("stance", "neutral"))
        else:
            prov, _, mod = (rec.get("model", "") or "").partition(":")
            receipt = self._DelegateTask(rec.get("goal", ""), context=context,
                provider=prov or None, model=mod or None,
                max_tokens=spawn.get("max_tokens"), toolsets=None if toolsets in (None, "all") else toolsets,
                timeout_seconds=spawn.get("timeout_seconds", 120), background=True,
                stance=spawn.get("stance", "neutral"))
        import re as _re
        m = _re.search(r"Background delegation (dlg_\d+) started", str(receipt))
        return {"revived": m.group(1) if m else "", "receipt": receipt}

    def _SelfPrompt(self, content, memory=True, **kwargs):
        """Add a prompt to the agent's own conversation history."""
        if self.Memory is None:
            raise RuntimeError("Memory is not enabled.")
        if memory:
            self.Memory.AddMessage("user", f"[Self-prompt] {content}")
            return {"status": "added_to_memory", "content": content}
        self.Memory.AddMessage("user", f"[Transient self-prompt] {content}")
        return {"status": "added_transient", "content": content}

    def _SendMessage(self, target_agent, content, **kwargs):
        """Send a message to another agent profile's inbox."""
        if str(target_agent or "").strip().lower() in {"user", "self", "me", "human", "operator"}:
            raise ValueError("send_message is agent-to-agent only — there is no 'User' profile. To address the user, just write your reply text; to note yourself, use run_skill self_message.")
        TargetFolder = self.RepoRoot / "Agents" / target_agent
        if not TargetFolder.is_dir():
            raise ValueError(f"Agent not found: {target_agent}")

        InboxDir = TargetFolder / "inbox"
        InboxDir.mkdir(parents=True, exist_ok=True)

        Timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        MsgFile = InboxDir / f"msg_{Timestamp}.json"
        MsgFile.write_text(json.dumps({
            "from": "agent",
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        }, indent=2), encoding="utf-8")

        try:
            Rel = MsgFile.relative_to(self.RepoRoot).as_posix()
        except ValueError:
            Rel = MsgFile.name
        return {"status": "sent", "target": target_agent, "message_file": Rel}

    def _CoerceQuestions(self, question=None, questions=None):
        """Pure helper: normalize ask_user args into [{question, options, default, required}]."""
        Items = []
        if questions is not None:
            if not isinstance(questions, list) or not questions:
                raise ValueError("questions must be a non-empty array.")
            if len(questions) > 6:
                raise ValueError("questions accepts at most 6 items per call; split into two calls.")
            for Item in questions:
                if isinstance(Item, str):
                    Item = {"question": Item}
                if not isinstance(Item, dict) or not isinstance(Item.get("question", ""), str) or not Item["question"].strip():
                    raise ValueError("Each question must be a string or {question, ...} with non-empty question text.")
                Options = Item.get("options")
                if Options is not None:
                    if not isinstance(Options, list) or not 2 <= len(Options) <= 8 or not all(isinstance(o, str) and o.strip() for o in Options):
                        raise ValueError(f"options for '{Item['question'][:40]}' must be 2-8 non-empty strings.")
                Default = Item.get("default", "")
                if Default is None:
                    Default = ""
                if not isinstance(Default, str):
                    raise ValueError("default must be a string.")
                if Options is not None and Default and Default not in Options:
                    raise ValueError(f"default '{Default}' must be one of options for '{Item['question'][:40]}'.")
                Items.append({"question": Item["question"].strip(), "options": Options or None,
                              "default": Default, "required": bool(Item.get("required", False))})
        elif isinstance(question, str) and question.strip():
            Items.append({"question": question.strip(), "options": None, "default": "", "required": False})
        else:
            raise ValueError("ask_user needs `question` (string) or `questions` (array of up to 6).")
        return Items

    def _AskUser(self, question=None, questions=None, **kwargs):
        """Ask one or more questions, waiting for each answer (main thread only)."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("ask_user is not available inside background delegations; report what you need and continue with what needs no approval.")
        if getattr(self, "SuppressAskUserThisTurn", False):
            raise RuntimeError("The user explicitly deferred this question to the next turn. Do not call ask_user now; finish the current task and ask on the next user turn.")
        Items = self._CoerceQuestions(question=question, questions=questions)
        from rich.prompt import Prompt as RichPrompt
        from rich.panel import Panel
        from rich.console import Console
        Console().print(Panel("\n\n".join(f"{i}. {it['question']}" + (f"  [options: {' / '.join(it['options'])}]" if it["options"] else "") for i, it in enumerate(Items, 1)),
                              title=f"Agent questions ({len(Items)})", border_style="cyan"))
        Answers = []
        for Item in Items:
            if Item["options"]:
                Lines = "\n".join(f"  {i}. {o}" for i, o in enumerate(Item["options"], 1))
                while True:
                    Raw = RichPrompt.ask(f"{Item['question']}\n{Lines}\nReply with number, exact text, or your own answer" + (f" [default: {Item['default']}]" if Item["default"] else "")).strip()
                    if not Raw and Item["default"]:
                        Answers.append(Item["default"])
                        break
                    if Raw.isdigit() and 1 <= int(Raw) <= len(Item["options"]):
                        Answers.append(Item["options"][int(Raw) - 1])
                        break
                    if Raw in Item["options"]:
                        Answers.append(Raw)
                        break
                    if Raw:
                        # Free text: never lock the user into the listed options.
                        Answers.append(Raw)
                        break
                    Console().print("[yellow]Pick one of the listed options (number or exact text).[/]")
            else:
                while True:
                    Raw = RichPrompt.ask(Item["question"] + (f" [default: {Item['default']}]" if Item["default"] else "")).strip()
                    if Raw or Item["default"] or not Item["required"]:
                        Answers.append(Raw or Item["default"])
                        break
                    Console().print("[yellow]An answer is required for this one.[/]")
        return {"answers": [{"question": it["question"], "answer": an} for it, an in zip(Items, Answers)], "answer": Answers[0]}

    def _SessionUsage(self, **kwargs):
        """Live session stats: call counts, per-tool tops, token/USD budget."""
        from collections import Counter
        Calls, Errors = Counter(), Counter()
        FirstTs, LastTs = None, None
        for Ts, Tool, _Skill, IsErr in list(getattr(self, "CallLog", []) or []):
            try:
                Calls[str(Tool)] += 1
                if IsErr:
                    Errors[str(Tool)] += 1
                Ts = float(Ts)
                FirstTs = Ts if FirstTs is None else min(FirstTs, Ts)
                LastTs = Ts if LastTs is None else max(LastTs, Ts)
            except Exception:
                continue
        Total = sum(Calls.values())
        Top = [{"tool": t, "calls": c, "errors": Errors.get(t, 0)} for t, c in Calls.most_common(10)]
        Budget = None
        try:
            Budget = self.Budget.status() if getattr(self, "Budget", None) is not None else None
        except Exception:
            pass
        Elapsed = round(LastTs - FirstTs, 1) if FirstTs and LastTs else 0.0
        return {"total_calls": Total, "total_errors": sum(Errors.values()), "elapsed_seconds": Elapsed,
                "by_tool": Top, "budget": Budget or {"tracking": "off"}}

    def _WaitForFile(self, path, timeout_seconds=60, poll_seconds=1, **kwargs):
        """Block until a workspace file appears/changes or the timeout hits."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a workspace-relative file or directory.")
        try:
            Timeout = max(5, min(int(timeout_seconds), 600))
        except Exception:
            Timeout = 60
        try:
            Poll = max(0.5, min(float(poll_seconds), 10))
        except Exception:
            Poll = 1.0
        Target = (self.ProjectRoot / path.strip()).resolve()
        if Target != self.ProjectRoot and self.ProjectRoot not in Target.parents:
            raise ValueError("Path must stay inside the workspace.")

        def Snapshot():
            try:
                if Target.is_file():
                    Stat = Target.stat()
                    return ("file", Stat.st_mtime, Stat.st_size)
                if Target.is_dir():
                    Entries = sorted(p.name for p in Target.iterdir())
                    return ("dir", Target.stat().st_mtime, len(Entries))
            except OSError:
                pass
            return ("missing", 0, 0)

        Start, Before = time.time(), Snapshot()
        while True:
            Now = Snapshot()
            if Now != Before and Now[0] != "missing":
                return {"changed": True, "path": path.strip(), "kind": Now[0],
                        "waited_seconds": round(time.time() - Start, 1)}
            if time.time() - Start >= Timeout:
                return {"changed": False, "path": path.strip(),
                        "waited_seconds": round(time.time() - Start, 1), "timeout_seconds": Timeout}
            time.sleep(Poll)

    def _CheckpointDir(self):
        Dir = self.RepoRoot / ".ae_sessions" / "checkpoints"
        Dir.mkdir(parents=True, exist_ok=True)
        return Dir

    def _Checkpoint(self, action, name=None, **kwargs):
        """Named snapshots of todos + long-term memory + summary."""
        if action == "list":
            return {"checkpoints": sorted(p.stem for p in self._CheckpointDir().glob("*.json"))}
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name or ""):
            raise ValueError("name must contain only letters, numbers, underscores, or hyphens.")
        File = self._CheckpointDir() / f"{name}.json"
        if action == "save":
            if self.Memory is None or not hasattr(self.Memory, "Save"):
                raise RuntimeError("checkpoints need persistent memory; not available for volatile subagents.")
            File.write_text(json.dumps({
                "name": name, "agent": getattr(self, "CurrentAgentName", "*"),
                "saved_at": datetime.datetime.now().isoformat(),
                "todos": getattr(self.Memory, "Todos", []) or [],
                "long_term": getattr(self.Memory, "LongTerm", []) or [],
                "summary": getattr(self.Memory, "Summary", "") or "",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"saved": name, "todos": len(getattr(self.Memory, "Todos", []) or []),
                    "long_term": len(getattr(self.Memory, "LongTerm", []) or [])}
        if action == "delete":
            if not File.is_file():
                raise ValueError(f"Checkpoint not found: {name}")
            File.unlink()
            return {"deleted": name}
        if action == "restore":
            if not File.is_file():
                raise ValueError(f"Checkpoint not found: {name}")
            if self.Memory is None or not hasattr(self.Memory, "Save"):
                raise RuntimeError("checkpoints need persistent memory; not available for volatile subagents.")
            try:
                Data = json.loads(File.read_text(encoding="utf-8"))
            except json.JSONDecodeError as Ex:
                raise ValueError(f"Checkpoint '{name}' is corrupt: {Ex}") from Ex
            self.Memory.Todos = Data.get("todos", []) or []
            self.Memory.LongTerm = Data.get("long_term", []) or []
            self.Memory.Summary = Data.get("summary", "") or ""
            try:
                self.Memory._todos_count = len(self.Memory.Todos)
                self.Memory._long_term_count = len(self.Memory.LongTerm)
            except Exception:
                pass
            self.Memory.Save()
            return {"restored": name, "saved_at": Data.get("saved_at", "?"),
                    "todos": len(self.Memory.Todos), "long_term": len(self.Memory.LongTerm)}
        raise ValueError("action must be save, restore, list, or delete.")

    def _Remind(self, text, delay_seconds=300, target_agent=None, **kwargs):
        """One-shot inbox reminder after a delay (daemon timer, within-session)."""
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("text must be a non-empty string up to 2000 chars.")
        try:
            Delay = max(10, min(int(delay_seconds), 86400))
        except Exception:
            Delay = 300
        Target = (target_agent or getattr(self, "CurrentAgentName", "") or "").strip()
        Folder = self.RepoRoot / "Agents" / Target
        if not Target or Target == "*" or not Folder.is_dir():
            raise ValueError("No resolvable agent inbox (are you '*'?). Pass target_agent with a real profile name.")

        def Deliver():
            try:
                InboxDir = Folder / "inbox"
                InboxDir.mkdir(parents=True, exist_ok=True)
                Stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                (InboxDir / f"msg_reminder_{Stamp}.json").write_text(json.dumps({
                    "from": f"remind ({Delay}s timer)", "content": f"[Reminder] {text.strip()}",
                    "timestamp": datetime.datetime.now().isoformat(),
                }, indent=2), encoding="utf-8")
            except Exception:
                pass

        Timer = threading.Timer(Delay, Deliver)
        Timer.daemon = True
        Timer.start()
        FiresAt = (datetime.datetime.now() + datetime.timedelta(seconds=Delay)).isoformat(timespec="seconds")
        return {"scheduled": True, "target": Target, "delay_seconds": Delay, "fires_at": FiresAt}

    def _NotesFile(self):
        """Per-agent scratchpad path. Resolves from whoever is asking (delegator for subagents)."""
        Name = (getattr(self, "CurrentAgentName", "") or "").strip() or "*"
        Safe = "".join(ch for ch in Name if ch.isalnum() or ch in {"_", "-"})
        if not Safe:
            raise ValueError("No resolvable agent identity for notes.")
        return self.RepoRoot / "Agents" / Safe / "Notes.md"

    def _AgentNote(self, action, text=None, notes=None, **kwargs):
        """Add/list/clear private self-notes (system-prompt scratchpad, not memory)."""
        # Alias leniency: models send notes=/note= instead of text= — accept it.
        if text is None:
            for _Alias in (notes, kwargs.get("notes"), kwargs.get("note")):
                if isinstance(_Alias, str) and _Alias.strip():
                    text = _Alias
                    break
                if isinstance(_Alias, list) and _Alias:
                    text = " ".join(str(x) for x in _Alias if str(x).strip())
                    break
        File = self._NotesFile()
        Notes = []
        if File.is_file():
            Notes = [l[2:].strip() for l in File.read_text(encoding="utf-8").splitlines() if l.startswith("- ") and l[2:].strip()]
        if action == "list":
            return {"notes": Notes, "count": len(Notes)}
        if action == "clear":
            # Permanent store — never silent. Main thread + explicit approval.
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("Clearing notes is not available inside background delegations.")
            self._AskApproval("agent_note", {"action": "clear"})
            if File.is_file():
                File.unlink()
            return {"cleared": True, "count": 0}
        if action == "add":
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string (≤300 chars).")
            Clean = re.sub(r"\s+", " ", text.strip())
            if len(Clean) > 300:
                raise ValueError(f"Note is {len(Clean)} chars; keep it ≤300 (one lesson per note).")
            if any(n.casefold() == Clean.casefold() for n in Notes):
                raise ValueError("That note is already recorded (check agent_note list) — add a different lesson, not a duplicate.")
            # Lesson-gate: standing notes are lessons/strategies, never result
            # echoes ("the result was 580.5") or tool-mechanics navel-gazing
            # ("X tool requires Y"). Facts go to memory_retain, not here.
            import re as _re2
            if not _re2.search(r"\b(fail(?:ed|ing)?|errors?|wrong|never|always|instead|because|avoid|prefer|rule|lesson|must|should|don't|do not|won't|can't use)\b", Clean, _re2.I):
                raise ValueError("Not a lesson: standing notes record strategies and failure-lessons (must contain a signal word like fail/error/never/always/instead/because/avoid/prefer). "
                                 "Plain results go in your reply, durable facts in memory_retain — not here.")
            if len(Notes) >= 20:
                raise ValueError("Notes full (20). Clear or rewrite them via clear + add before adding more.")
            Notes.append(Clean)
            File.parent.mkdir(parents=True, exist_ok=True)
            File.write_text("\n".join(f"- {n}" for n in Notes) + "\n", encoding="utf-8")
            return {"added": True, "count": len(Notes), "note": Clean}
        raise ValueError("action must be add, list, or clear.")

    def _UpdateBeing(self, text, **kwargs):
        """Rewrite the agent's own Being prompt (main thread only, approval-gated)."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("update_being is not available inside background delegations.")
        if not isinstance(text, str) or not 50 <= len(text.strip()) <= 6000:
            raise ValueError("text must be the complete new Being prompt (50-6000 chars).")
        Name = (getattr(self, "CurrentAgentName", "") or "").strip()
        Folder = self.RepoRoot / "Agents" / Name
        if not Name or Name == "*" or not (Folder / "agent.json").is_file():
            raise ValueError("No resolvable agent profile for update_being.")
        Clean = text.strip()
        (Folder / "Being.md").write_text(Clean, encoding="utf-8")
        try:
            Data = json.loads((Folder / "agent.json").read_text(encoding="utf-8"))
            Data["default_prompt"] = Clean
            (Folder / "agent.json").write_text(json.dumps(Data, indent=2), encoding="utf-8")
        except Exception as Ex:
            raise RuntimeError(f"Being.md updated but agent.json failed: {Ex}") from Ex
        return {"updated": True, "agent": Name, "chars": len(Clean)}

    def _Clock(self, **kwargs):
        """Current date/time: local, UTC, ISO, human. Ground truth for 'now'."""
        Now = datetime.datetime.now()
        Utc = datetime.datetime.now(datetime.timezone.utc)
        return {"local": Now.strftime("%A, %B %d, %Y %H:%M:%S"),
                "utc": Utc.strftime("%A, %B %d, %Y %H:%M:%S UTC"),
                "iso_local": Now.isoformat(timespec="seconds"),
                "iso_utc": Utc.isoformat(timespec="seconds"),
                "year": Now.year, "month": Now.month, "month_name": Now.strftime("%B"),
                "day": Now.day, "weekday": Now.strftime("%A")}

    def _TurnStats(self, **kwargs):
        """Live per-turn ledger + session tops."""
        from collections import Counter
        Turn = dict(getattr(self, "TurnCounts", {}) or {})
        Calls = Counter()
        for _Ts, Tool, _Skill, _Err in list(getattr(self, "CallLog", []) or []):
            try:
                Calls[str(Tool)] += 1
            except Exception:
                continue
        Out = {"turn": Turn, "turn_total": sum(Turn.values()),
                "session_top": [{"tool": t, "calls": c} for t, c in Calls.most_common(10)],
                "session_total": sum(Calls.values())}
        if kwargs:
            Out["warning"] = "turn_stats takes NO parameters — call it bare. Stop passing arguments."
        return Out

    def _StageGate(self, milestone, demo="", **kwargs):
        """Milestone gate: user picks continue / redirect / stop (main thread only)."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("stage_gate needs the user at the terminal; not available inside background delegations. Report the milestone and continue.")
        if not isinstance(milestone, str) or not milestone.strip():
            raise ValueError("milestone must be a non-empty string.")
        Demo = str(demo or "")[:2000]
        from rich.prompt import Prompt as RichPrompt
        from rich.panel import Panel
        from rich.console import Console
        Body = f"[bold]{milestone.strip()}[/]\n" + (f"\n{Demo}" if Demo else "\n[dim](no demo evidence given)[/]")
        Console().print(Panel(Body, title="Stage gate — your call", border_style="magenta"))
        Console().print("  1. continue   2. redirect   3. stop")
        while True:
            Raw = (RichPrompt.ask("Decision (1/2/3)") or "").strip().lower()
            if Raw in {"1", "continue"}:
                return {"decision": "continue", "detail": ""}
            if Raw in {"3", "stop"}:
                return {"decision": "stop", "detail": ""}
            if Raw in {"2", "redirect"}:
                Detail = RichPrompt.ask("Redirect — what should change").strip()
                if Detail:
                    return {"decision": "redirect", "detail": Detail}
                Console().print("[yellow]Say what should change, or pick 1/3.[/]")
                continue
            Console().print("[yellow]Pick 1, 2, or 3.[/]")

    def _HandoffDir(self):
        Dir = self.RepoRoot / ".ae_sessions" / "handoffs"
        Dir.mkdir(parents=True, exist_ok=True)
        return Dir

    def _Handoff(self, action, id=None, goal="", tried="", remaining="", files=None, **kwargs):
        """Continuity packets: save/load/list/delete goal-state snapshots."""
        if action == "list":
            return {"handoffs": sorted(p.stem for p in self._HandoffDir().glob("*.json"))}
        if not isinstance(id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", id or ""):
            raise ValueError("id must contain only letters, numbers, underscores, or hyphens.")
        File = self._HandoffDir() / f"{id}.json"
        if action == "delete":
            if not File.is_file():
                raise ValueError(f"Handoff not found: {id}")
            File.unlink()
            return {"deleted": id}
        if action == "load":
            if not File.is_file():
                raise ValueError(f"Handoff not found: {id}")
            try:
                return json.loads(File.read_text(encoding="utf-8"))
            except json.JSONDecodeError as Ex:
                raise ValueError(f"Handoff '{id}' is corrupt: {Ex}") from Ex
        if action == "save":
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("save needs goal (what is being achieved).")
            if files is not None and (not isinstance(files, list) or len(files) > 20 or not all(isinstance(f, str) for f in files)):
                raise ValueError("files must be an array of up to 20 path strings.")
            Notes = []
            try:
                _NF = self._NotesFile()
                if _NF.is_file():
                    Notes = [l[2:].strip() for l in _NF.read_text(encoding="utf-8").splitlines() if l.startswith("- ")][:20]
            except Exception:
                pass
            Packet = {
                "id": id, "agent": getattr(self, "CurrentAgentName", "*"),
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "goal": goal.strip(), "tried": str(tried or "")[:2000],
                "remaining": str(remaining or "")[:2000], "files": files or [],
                "todos": list(getattr(self.Memory, "Todos", []) or []) if self.Memory is not None else [],
                "notes": Notes,
                "summary": str(getattr(self.Memory, "Summary", "") or "")[:2000] if self.Memory is not None else "",
            }
            File.write_text(json.dumps(Packet, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"saved": id, "todos": len(Packet["todos"]), "notes": len(Notes)}
        raise ValueError("action must be save, load, list, or delete.")

    def _TaskComplete(self, summary, **kwargs):
        """Signal that all work is complete. Vacuous claim-summaries are rejected."""
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string.")
        Clean = summary.strip()
        _TodoStart = getattr(self, "TurnTodoIdsAtStart", None)
        if self.Memory is not None and _TodoStart is not None:
            _OpenNewTodos = [
                t for t in (getattr(self.Memory, "Todos", []) or [])
                if str(t.get("id")) not in _TodoStart and str(t.get("status", "todo")) not in {"done", "cancelled"}
            ]
            if _OpenNewTodos:
                _Titles = ", ".join(str(t.get("title", "untitled")) for t in _OpenNewTodos[:4])
                raise ValueError(f"Cannot complete yet: current-turn todos remain open ({_Titles}). Finish or cancel them, then submit task_complete again.")
        import re as _re
        _Request = str(getattr(self, "CurrentUserRequest", "") or "")
        _AskedToExplain = bool(_re.search(r"\b(?:explain|teach|show|example)\b", _Request, _re.I))
        _ClaimOnly = bool(_re.match(r"\s*(?:provided|created|completed|finished|the task is complete|here is the summary)\b", Clean, _re.I))
        _ContainsExample = bool(
            "```" in Clean
            or _re.search(r"`[^`]+`", Clean)
            or _re.search(r"\b(?:local|function|print|prints|returns?)\b.{0,100}(?:=|\btrue\b|\bfalse\b|\d+)", Clean, _re.I)
            or _re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", Clean)
        )
        if _AskedToExplain and _ClaimOnly and not _ContainsExample:
            raise ValueError("The user requested an explanation/example, but this summary only claims one was provided. Include the explanation and actual example/output before completing.")
        if re.search(r"\b(?:outline|learning path)\b", _Request, re.I) and len(Clean) < 500:
            _HasStructure = bool(re.search(r"(?m)^\s*(?:[-*]|\d+[.)]|#{1,6})\s+", Clean))
            if not _HasStructure:
                raise ValueError("This request needs the actual outline, not a completion claim. Include the requested levels and topics in the final answer.")
        _DetailedReport = bool(
            _re.search(r"\b(?:detailed|comprehensive)\s+(?:report|review)\b", _Request, _re.I)
            or _re.search(r"\b(?:generate|write|create|prepare|produce)\s+(?:me\s+)?(?:a\s+)?(?:detailed\s+)?(?:report|review)\b", _Request, _re.I)
        )
        if _DetailedReport:
            _NeedsCitations = bool(_re.search(r"\b(search|research|sources?|citations?|forums?)\b", _Request, _re.I))
            _FetchedPages = set()
            _HasSearchEvidence = False
            for _EvidenceName, _EvidenceArgs, _EvidenceResult in getattr(self, "TurnToolEvidence", []) or []:
                if str(_EvidenceName).casefold() == "web_search":
                    _HasSearchEvidence = True
                elif str(_EvidenceName).casefold() == "web_fetch":
                    _Url = str((_EvidenceArgs or {}).get("url", "")).strip()
                    if _Url:
                        _FetchedPages.add(_Url)
                elif str(_EvidenceName).casefold() == "fetch_many":
                    try:
                        _FetchResults = json.loads(_EvidenceResult) if isinstance(_EvidenceResult, str) else _EvidenceResult
                    except Exception:
                        _FetchResults = []
                    if isinstance(_FetchResults, list):
                        for _FetchResult in _FetchResults:
                            if not isinstance(_FetchResult, dict) or _FetchResult.get("error") or not _FetchResult.get("text"):
                                continue
                            _Url = str(_FetchResult.get("url", "")).strip()
                            if _Url:
                                _FetchedPages.add(_Url)
            if _NeedsCitations and (not _HasSearchEvidence or len(_FetchedPages) < 2):
                raise ValueError(
                    "Gather search evidence and successfully fetch at least two distinct source pages first. "
                    "Then call task_complete with a short completion marker; write the report as normal final-answer text, not as a tool argument."
                )
            self.TaskCompleteRequiresFinal = True
            self.TaskCompleteNeedsCitations = _NeedsCitations
        _CreationClaim = _re.search(
            r"\b(?:has been|was|is now|successfully)\s+(?:created|saved|built|generated|armed)\b|"
            r"\b(?:created|saved|built|generated|armed)\s+(?:the\s+)?(?:skill|event|executable|tool)\b",
            Clean,
            _re.I,
        )
        if _CreationClaim:
            for _FailedName, _FailedArgs, _Failure in getattr(self, "TurnToolFailures", []) or []:
                if str(_FailedName).casefold() not in {"create_skill", "create_event", "create_executable", "create_tool"}:
                    continue
                _EntityName = str((_FailedArgs or {}).get("name", "")).strip()
                if not _EntityName or _EntityName.casefold() not in Clean.casefold():
                    continue
                _Created = any(
                    str(_SuccessName).casefold() == str(_FailedName).casefold()
                    and isinstance(_SuccessArgs, dict)
                    and str(_SuccessArgs.get("name", "")).casefold() == _EntityName.casefold()
                    for _SuccessName, _SuccessArgs, _SuccessResult in getattr(self, "TurnToolEvidence", []) or []
                )
                if not _Created:
                    raise ValueError(
                        f"Cannot claim '{_EntityName}' was created: {_FailedName} failed and no successful creation result exists this turn. "
                        "Report the failure or retry with corrected inputs first."
                    )
        from decimal import Decimal, InvalidOperation
        for _Name, _Arguments, _Result in getattr(self, "TurnToolEvidence", []) or []:
            if str(_Name).casefold() != "calculate":
                continue
            try:
                _Payload = json.loads(_Result) if isinstance(_Result, str) else _Result
                _Expected = _Payload.get("result") if isinstance(_Payload, dict) else None
                if isinstance(_Expected, bool) or not isinstance(_Expected, (int, float, str)):
                    continue
                _Expected = Decimal(str(_Expected))
                _Reported = {
                    Decimal(_Token.replace(",", ""))
                    for _Token in _re.findall(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?", Clean)
                }
            except (InvalidOperation, ValueError, TypeError, AttributeError):
                continue
            if _Expected not in _Reported:
                raise ValueError(
                    f"Calculation result mismatch: calculate returned {_Expected}, but the completion summary does not contain that value. "
                    "Use the exact result from the tool output before completing."
                )
        if (_re.search(r"has been (rendered|described|provided|displayed|retrieved|summarized|completed)", Clean, _re.I)
                or _re.search(r"using the \w+ tool", Clean, _re.I)
                or _re.search(r"successfully (rendered|retrieved|displayed|described|completed)", Clean, _re.I)):
            raise ValueError("Vacuous summary: it claims work was done without containing any of it. "
                             "Put the ACTUAL content in the summary (the ASCII lines, temps, facts, description) — then complete.")
        # Palette grounding: a fresh vision packet contradicts ungrounded
        # nature claims (brown/black palette = no water/forest). Reject them.
        # Scoped: only summaries ABOUT imagery answer to vision rules —
        # a weather summary with a packet in context is not describing it.
        try:
            import time as _time
            _Saved = getattr(self, "_LastPalette", ("", 0, 255)) or ("", 0, 255)
            _Pal, _Ts = _Saved[0], _Saved[1]
            _Mean = _Saved[2] if len(_Saved) > 2 else 255
            _Low = Clean.lower()
            _AboutImage = bool(_re.search(r"\b(image|picture|photo|depict|render|ascii|visual|shows|looks|landscape|forest|water|chains|sky)\b", _Low))
            if _Pal and _time.time() - float(_Ts) < 600 and _AboutImage:
                _Claims = {"water": ("blue", "cyan"), "ocean": ("blue", "cyan"), "sea": ("blue", "cyan"),
                           "lake": ("blue", "cyan"), "river": ("blue", "cyan"), "forest": ("green",),
                           "trees": ("green",), "jungle": ("green",)}
                _Low = Clean.lower()
                for _Word, _Fams in _Claims.items():
                    if _re.search(r"\b" + _Word + r"\b", _Low) and not any(_F in _Pal.lower() for _F in _Fams):
                        # Ruling OUT ("ruled out water", "no blue") is the desired
                        # behavior — only claims ("it is water") get rejected.
                        _Sent = _re.split(r"[.!?]\s*", _Low)
                        _RuledOut = any(_Word in _S and any(_M in _S for _M in ("ruled out", "rules out", "no " + _Fams[0], "not " + _Word, "rather than", "instead of")) for _S in _Sent)
                        if _RuledOut:
                            continue
                        raise ValueError(f"Palette contradicts '{_Word}': { _Pal }. "
                                         f"A { _Word } claim needs { '/'.join(_Fams) } in the palette and there is none. "
                                         "Drop the claim or cite the view that shows the missing color.")
                # Brightness grounding: a dark frame cannot host daylight scenes.
                if _Mean < 70 and _re.search(r"clear sky|blue sky|daylight|sunny|sunlit|bright day", _Low):
                    raise ValueError(f"Brightness contradicts daylight: frame mean is {_Mean}/255 (dark). "
                                     "A clear-sky/sunny scene cannot be this dark. Describe the dark scene as it is.")
                # Vision grounding: a fresh vision packet means the verdict must
                # carry packet data (a color, %, hex, or mass) AND a commitment
                # (confidence level + ruled-out alternative). Packet-echo with
                # neither is recitation, not a verdict. Reject it.
                _Grounded = (_re.search(r"#[0-9a-f]{6}", _Low)
                             or _re.search(r"\d+%", Clean)
                             or _re.search(r"\bmass\b", _Low)
                             or any(_C in _Low for _C in ("black", "white", "gray", "grey", "red", "orange",
                                                          "yellow", "green", "cyan", "blue", "purple", "brown", "pink")))
                if not _Grounded:
                    raise ValueError(f"Ungrounded verdict: a fresh vision packet exists but the summary quotes none of its data. "
                                     f"Name a palette color, % figure, or mass from the packet ({ _Pal[:120] }).")
                _Committed = (_re.search(r"\b(high|medium|low)\b[^.]{0,40}\bconfiden", _Low)
                              or _re.search(r"\bconfiden\w*\s*:\s*(high|medium|low)", _Low)
                              or any(_W in _Low for _W in ("appears", "possibly", "likely", "unlikely", "seems",
                                                           "suggests", "uncertain", "ruled out", "rules out",
                                                           "rather than", "instead of")))
                if not _Committed:
                    raise ValueError("Uncommitted verdict: state your confidence (high/medium/low) and name one ruled-out "
                                     "alternative with its reason — a verdict without commitment is recitation.")
        except ValueError:
            raise
        except Exception:
            pass
        return {"status": "complete", "summary": Clean}

    def ScopeRoot(self, Scope=None):
        """Root directory for a capability scope.

        'local'  the calling agent's own tree (Agents/<name>/AutomatableExecutables)
        'shared' the shared tree (AutomationRoot)
        'global' the repo-level tree, when that differs from shared

        `None`/`'auto'` picks local when the agent has its own tree and shared
        otherwise, so an agent bootstraps into its own space by default while a
        scope-less caller (no identity) keeps the historical shared behavior.
        """
        Key = (Scope or "auto").strip().lower()
        if Key in ("", "auto"):
            Key = "local" if self.AgentRoot() else "shared"
        if Key == "local":
            Root = self.AgentRoot()
            if Root is None:
                raise ValueError(
                    "No local capability tree for this agent. Write with scope='shared' instead.")
            return Root
        if Key == "shared":
            return self.AutomationRoot
        if Key == "global":
            Global = self.RepoRoot / "AutomatableExecutables"
            return Global if Global.is_dir() else self.AutomationRoot
        raise ValueError("scope must be one of: local, shared, global.")

    def _ResolveAutomationPath(self, RelativePath, Scope=None):
        Root = self.ScopeRoot(Scope).resolve()
        Target = (Root / RelativePath).resolve()
        if Target != Root and Root not in Target.parents:
            raise ValueError("Path must stay inside the capability tree.")
        if Target.suffix.lower() not in (".ae", ".md"):
            raise ValueError("Only .ae and .md files may be accessed in the automation layer.")
        return Target

    def _AeRead(self, path, scope=None):
        FilePath = self._ResolveAutomationPath(path, scope)
        if not FilePath.is_file():
            raise ValueError(f"Automation file not found: {path}")
        if FilePath.stat().st_size > 1_000_000:
            raise ValueError("File exceeds the 1 MB read limit.")
        return FilePath.read_text(encoding="utf-8")

    def _AeWrite(self, path, content, scope=None):
        FilePath = self._ResolveAutomationPath(path, scope)
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string.")
        FilePath.parent.mkdir(parents=True, exist_ok=True)
        FilePath.write_text(content, encoding="utf-8", newline="")
        # Invalidate skills cache if writing to Skills
        try:
            if FilePath.resolve().parent.name == Taxonomy.TYPE_FOLDERS.get("skill"):
                self._skills_cache = None
        except Exception:
            pass
        return f"Wrote {FilePath.relative_to(self.RepoRoot).as_posix()}"

    def _AeDelete(self, path, scope=None):
        FilePath = self._ResolveAutomationPath(path, scope)
        if not FilePath.is_file():
            raise ValueError(f"Automation file not found: {path}")
        FilePath.unlink()
        try:
            if FilePath.parent.name == Taxonomy.TYPE_FOLDERS.get("skill"):
                self._skills_cache = None
        except Exception:
            pass
        return f"Deleted {FilePath.relative_to(self.RepoRoot).as_posix()}"

    def _ShareCapability(self, name, type=None, scope="shared", category=None, overwrite=False):
        """Promote a capability this agent owns out of its local tree.

        Capabilities start agent-local; sharing is the explicit opt-in that makes
        one visible to other agents. Copies the file rather than moving it, so
        the local copy keeps working.
        """
        Source = self.AgentRoot()
        if Source is None:
            raise ValueError("This agent has no local capability tree to share from.")
        Script = None
        for Type in ([type] if type else sorted(AEXScript.Types)):
            for Directory in self._TypePathFrom(Source, Type),:
                if not Directory.is_dir():
                    continue
                for AEXFile in sorted(Directory.rglob("*.ae")):
                    try:
                        Candidate = AEXScript.FromFile(AEXFile)
                    except (AEXError, OSError):
                        continue
                    if Candidate.Name == name and Candidate.Type == Type:
                        Script = Candidate
                        break
                if Script:
                    break
            if Script:
                break
        if Script is None:
            raise ValueError(
                f"No local {type or 'capability'} named '{name}' to share. "
                "Only capabilities in this agent's own tree can be shared.")

        Target = self.ScopeRoot(scope).resolve()
        if Target == Source.resolve():
            raise ValueError(f"'{name}' is already at scope '{scope}'.")

        Sub = Script.Category if category is None else Taxonomy.Normalize(category, Script.Type)
        Destination = self._TypePathFrom(Target, Script.Type) / (Sub if Sub and Sub != Taxonomy.UNCATEGORIZED else "") / Path(Script.SourcePath).name
        Destination.parent.mkdir(parents=True, exist_ok=True)
        if Destination.exists() and not overwrite:
            raise ValueError(f"{Script.Type} '{name}' already exists in scope '{scope}' at "
                             f"{Destination.relative_to(self.RepoRoot).as_posix()}. Pass overwrite=true to replace it.")
        Content = Path(Script.SourcePath).read_text(encoding="utf-8")
        if category:
            # Keep the declared category in step with where it now lives.
            Content = re.sub(r"^category:.*$", f"category: {Sub}", Content, count=1, flags=re.MULTILINE)
        Destination.write_text(Content, encoding="utf-8", newline="")
        try:
            if Destination.parent.name == Taxonomy.TYPE_FOLDERS.get("skill"):
                self._skills_cache = None
        except Exception:
            pass
        return {
            "shared": Script.Name,
            "type": Script.Type,
            "category": Sub,
            "scope": scope,
            "path": str(Destination.relative_to(self.RepoRoot).as_posix()),
        }

    def _ListCapabilities(self, type=None, category=None, tag=None, scope=None, query=None, limit=None):
        """What this agent can actually use, grouped by type and category."""
        Entries = self._VisibleCapabilities()
        if type:
            Entries = [E for E in Entries if E["type"] == type]
        if category:
            Want = Taxonomy.Normalize(category)
            Entries = [E for E in Entries if E["category"] == Want]
        if tag:
            Want = Taxonomy.Normalize(tag)
            Entries = [E for E in Entries if Want in (E.get("tags") or [])]
        if scope:
            Entries = [E for E in Entries if E["scope"] == scope]
        if query:
            q = str(query).lower()
            toks = [t for t in re.findall(r"[a-z0-9_]+", q) if t]
            def Score(Entry):
                hay = (Entry["name"] + " " + (Entry.get("description") or "") + " "
                       + " ".join(Entry.get("tags") or [])).lower()
                return sum(2 if t == Entry["name"].lower() else (1 if t in hay else 0) for t in toks)
            Scored = [(Score(E), E) for E in Entries]
            Scored = [(S, E) for S, E in Scored if S > 0]
            Scored.sort(key=lambda Pair: (-Pair[0], Pair[1]["name"].lower()))
            Entries = [E for _, E in Scored]
        if limit:
            Entries = Entries[:max(1, int(limit))]
        Groups = []
        for Type in sorted(AEXScript.Types):
            OfType = [E for E in Entries if E["type"] == Type]
            if not OfType:
                continue
            Groups.append({
                "type": Type,
                "count": len(OfType),
                "categories": [
                    {"category": Cat, "capabilities": [E["name"] for E in Bucket]}
                    for Cat, Bucket in Taxonomy.PromptGroups(OfType)
                ],
            })
        return {"count": len(Entries), "types": Groups,
                "names": [E["name"] for E in Entries]}

    def _CapabilityPath(self, name, type=None, category=None, scope=None):
        Aliases = {"core_tool": "tool", "executable": "ae", "skill": "skill",
                   "tool": "tool", "event": "event", "workflow": "workflow",
                   "knowledge": "knowledge", "note": "note", "data": "data", "ae": "ae"}
        Want = None
        WantCategory = category
        if type:
            Key = str(type).strip().lower()
            if Key == "core_tool":
                Want, WantCategory = "tool", (WantCategory or "core")
            elif Key in Aliases:
                Want = Aliases[Key]
            else:
                raise ValueError(f"Unknown entity type: {type}.")
        Matches = [E for E in self._VisibleCapabilities()
                   if E["name"] == name and (Want is None or E["type"] == Want)]
        if not Matches:
            raise ValueError(
                f"No capability named '{name}'" + (f" of type '{type}'" if type else "") +
                " is visible to you. Use list_capabilities to see what you actually have.")
        if WantCategory:
            Matches = [E for E in Matches if E["category"] == Taxonomy.Normalize(WantCategory)]
        if scope:
            Matches = [E for E in Matches if E["scope"] == scope]
        if not Matches:
            if scope:
                raise ValueError(f"'{name}' is not in scope '{scope}'.")
            Matches = [E for E in self._VisibleCapabilities()
                       if E["name"] == name and (Want is None or E["type"] == Want)]
        Entry = Matches[0]
        Absolute = Path(Entry["path"])
        Root = self.ScopeRoot(Entry["scope"]).resolve()
        try:
            Relative = Absolute.resolve().relative_to(Root).as_posix()
        except ValueError:
            Relative = Absolute.as_posix()
        return Relative

    def _AeList(self):
        result = {"tools": [], "core_tools": [], "skills": [], "executables": [], "events": [], "inboxes": []}

        for AEXFile in self._AEXFiles(self.ToolsPath):
            try:
                if self.CoreToolsPath.resolve() in AEXFile.resolve().parents:
                    continue
            except Exception:
                if "core" in AEXFile.parts:
                    continue
            Script = self._TryLoadAEX(AEXFile)
            if Script and Script.Type == "tool":
                result["tools"].append({"name": Script.Name, "description": Script.Description, "parameters": Script.Parameters})

        for AEXFile in self._AEXFiles(self.CoreToolsPath):
            Script = self._TryLoadAEX(AEXFile)
            if Script and Script.Type == "tool":
                result["core_tools"].append({"name": Script.Name, "description": Script.Description, "parameters": Script.Parameters})

        for SkillFile in sorted(self.SkillsPath.rglob("*.ae")):
            Script = self._TryLoadAEX(SkillFile)
            if Script is None or Script.Type != "skill":
                # Report unloadable files instead of hiding them: invisible files
                # cause "created but not found" loops. Deletable via delete_entity.
                try:
                    Problem = ""
                    try:
                        AEXScript.FromFile(SkillFile)
                    except Exception as Ex:
                        Problem = str(Ex)[:200]
                    result["skills"].append({
                        "name": SkillFile.parent.name,
                        "category": SkillFile.parent.relative_to(self.SkillsPath).as_posix(),
                        "description": f"[BROKEN - failed to load: {Problem}]",
                        "path": SkillFile.relative_to(self.AutomationRoot).as_posix(),
                        "format": "ae",
                        "load_error": Problem,
                    })
                except Exception:
                    pass
                continue
            result["skills"].append({
                "name": Script.Name,
                "category": SkillFile.parent.relative_to(self.SkillsPath).as_posix(),
                "description": Script.Description,
                "path": SkillFile.relative_to(self.AutomationRoot).as_posix(),
                "format": "ae",
            })

        for AEXFile in self._AEXFiles(self.ExecutablesPath):
            Script = self._TryLoadAEX(AEXFile)
            if Script and Script.Type == "ae":
                result["executables"].append({"name": Script.Name, "description": Script.Description, "parameters": Script.Parameters})

        for AEXFile in self._AEXFiles(self.EventsPath):
            Script = self._TryLoadAEX(AEXFile)
            if Script and Script.Type == "event":
                result["events"].append({"name": Script.Name, "description": Script.Description, "trigger": Script.Trigger})

        AgentsDir = self.RepoRoot / "Agents"
        if AgentsDir.is_dir():
            for Folder in sorted(AgentsDir.iterdir()):
                if not Folder.is_dir():
                    continue
                Inbox = Folder / "inbox"
                Pending = len(list(Inbox.glob("msg_*.json"))) if Inbox.is_dir() else 0
                result["inboxes"].append({"agent": Folder.name, "pending": Pending})

        return result

    def EmitEvent(self, Name, Data):
        Script = self._FindAEX(Name, "event", [self.EventsPath])
        if Script is None:
            raise ValueError(f"Event not found: {Name}")
        return {"input": Script.Metadata.get("input_prompt", ""), "response": Script.Metadata.get("response"), "event": Data}

    def DispatchEvent(self, EventType, Data, AgentName="*"):
        Triggered = []
        for AEXFile in self._AEXFiles(self.EventsPath):
            Script = self._TryLoadAEX(AEXFile)
            if Script is None or Script.Type != "event":
                continue
            if not Script.Metadata.get("enabled", True):
                continue
            if Script.Metadata.get("target_agent", "*") not in {"*", AgentName}:
                continue
            Raw = Script.Trigger.get("on", []) if isinstance(Script.Trigger, dict) else []
            Entries = Raw if isinstance(Raw, list) else [Raw]
            for Trigger in Entries:
                if isinstance(Trigger, str):
                    Trigger = {"type": "message", "match": Trigger}
                if not isinstance(Trigger, dict):
                    continue
                if Trigger.get("type") != EventType:
                    continue
                EventData = {**Data, "agent": AgentName, "trigger": Trigger}
                Result = {"input": Script.Metadata.get("input_prompt", ""), "response": Script.Metadata.get("response"), "event": EventData}
                Triggered.append({"name": Script.Name, "result": Result})
                if self.Memory is not None:
                    self.Memory.AddToolEvent(f"event:{Script.Name}", EventData, Result)
                break
        return Triggered

    def PollEvents(self, UserInput, AgentName="*", OnlyTimed=False):
        """Poll event triggers. With OnlyTimed=True (scheduler ticks), message
        triggers are skipped so chat-only events never fire on a timer.
        Per-event opt-out: events with `scheduler: false` metadata never fire
        on scheduler ticks (chat polling still works)."""
        Triggered = []
        Now = datetime.datetime.now().timestamp()
        for AEXFile in self._AEXFiles(self.EventsPath):
            Script = self._TryLoadAEX(AEXFile)
            if Script is None or Script.Type != "event":
                continue
            if not Script.Metadata.get("enabled", True):
                continue
            if OnlyTimed and Script.Metadata.get("scheduler", True) is False:
                continue
            TargetAgent = Script.Metadata.get("target_agent", "*")
            if TargetAgent not in {"*", AgentName}:
                continue
            RawEntries = Script.Trigger.get("on", []) if isinstance(Script.Trigger, dict) else []
            TriggerEntries = RawEntries if isinstance(RawEntries, list) else [RawEntries]
            for Index, Trigger in enumerate(TriggerEntries):
                if isinstance(Trigger, str):
                    Trigger = {"type": "message", "match": Trigger}
                if not isinstance(Trigger, dict):
                    continue
                # Evaluate condition if present (e.g., "data.title is not None")
                Condition = Trigger.get("condition")
                if Condition:
                    try:
                        # Simple eval with limited scope: data, event, agent, now
                        EventDataForCond = {"agent": AgentName, "text": UserInput, "timestamp": Now}
                        CondResult = _SafeEvalCondition(Condition, {"data": EventDataForCond, "event": EventDataForCond, "agent": AgentName, "now": Now})
                        if not CondResult:
                            continue
                    except Exception:
                        continue  # condition error -> skip trigger
                TriggerType = Trigger.get("type", "message")
                if OnlyTimed and TriggerType == "message":
                    continue
                StateKey = f"{Script.Name}:{Index}"
                EventData = {"agent": AgentName, "text": UserInput, "timestamp": Now}
                ShouldRun = False
                if TriggerType == "message":
                    Match = Trigger.get("match", "")
                    ShouldRun = not Match or Match.casefold() in UserInput.casefold()
                elif TriggerType == "interval":
                    Interval = self._ParseInterval(Trigger.get("every", ""))
                    Previous = self.EventState.get(StateKey)
                    if Previous is None:
                        self.EventState[StateKey] = Now
                    elif Now - Previous >= Interval:
                        ShouldRun = True
                        self.EventState[StateKey] = Now
                elif TriggerType == "file_change":
                    FilePath = self._ResolveProjectPath(Trigger.get("path", ""))
                    Signature = FilePath.stat().st_mtime_ns if FilePath.is_file() else None
                    Previous = self.EventState.get(StateKey)
                    self.EventState[StateKey] = Signature
                    ShouldRun = Previous is not None and Signature != Previous
                if ShouldRun:
                    EventData["trigger"] = Trigger
                    Result = {"input": Script.Metadata.get("input_prompt", ""), "response": Script.Metadata.get("response"), "event": EventData}
                    Triggered.append({"name": Script.Name, "result": Result})
                    if self.Memory is not None:
                        self.Memory.AddToolEvent(f"event:{Script.Name}", EventData, Result)
                    break
        self._SaveEventState()
        return Triggered

    def _ParseInterval(self, Value):
        Match = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", str(Value).lower())
        if not Match or int(Match.group(1)) < 1:
            raise ValueError("Interval must look like 30s, 5m, 2h, or 1d.")
        Multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[Match.group(2)]
        return int(Match.group(1)) * Multiplier

    def start_scheduler(self, callback, interval_seconds=20):
        """Background tick for timed events (interval/file_change only — never
        message triggers). Fired hits are delivered to callback(name, result).
        Per-event opt-out: events with `scheduler: false` in metadata are skipped.
        Global kill-switch: `scheduler off` CLI command sets SchedulerEnabled=False."""
        if self._SchedulerThread is not None and self._SchedulerThread.is_alive():
            return
        self._SchedulerStop.clear()

        def _loop():
            while not self._SchedulerStop.wait(interval_seconds):
                if not getattr(self, "SchedulerEnabled", True):
                    continue
                try:
                    for Hit in self.PollEvents("__scheduler_tick__", "*", OnlyTimed=True):
                        try:
                            callback(Hit["name"], Hit["result"])
                        except Exception:
                            pass
                except Exception:
                    pass

        self._SchedulerThread = threading.Thread(target=_loop, daemon=True)
        self._SchedulerThread.start()

    def stop_scheduler(self):
        self._SchedulerStop.set()
