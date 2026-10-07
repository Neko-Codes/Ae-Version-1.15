import datetime
import json
import re
import time

import yaml

try:
    from Taxonomy import (Normalize as _TaxNormalize, NormalizeTags as _TaxTags,
                          FromPath as _TaxFromPath, UNCATEGORIZED as _TAX_UNCAT)
except ImportError:  # package-relative import when Scripts/ is a package
    from .Taxonomy import (Normalize as _TaxNormalize, NormalizeTags as _TaxTags,
                           FromPath as _TaxFromPath, UNCATEGORIZED as _TAX_UNCAT)


class AEXError(ValueError):
    pass


class AEXStopped(Exception):
    """Cooperative stop: raised inside .ae via report()/should_stop() checks."""
    pass


class AEXScript:
    # Every capability kind is a `type:` value. Nothing is implicit.
    Types = {"tool", "skill", "workflow", "ae", "event", "knowledge", "note", "data"}
    # Callable: has parameters + execute, exposed to the model as a function.
    CallableTypes = {"tool", "skill", "ae"}
    # Text-first: frontmatter + body, where the BODY is the content.
    # No execute, no parameters -- prose is the payload.
    TextTypes = {"workflow", "knowledge", "note", "data"}
    # Capability folder name -> type stored in each file's frontmatter.
    TypeFolders = {
        "Tools": "tool",
        "Skills": "skill",
        "Workflows": "workflow",
        "Executables": "ae",
        "Events": "event",
        "Knowledge": "knowledge",
        "Notes": "note",
    }
    ParameterTypes = {"string", "number", "integer", "boolean", "array", "object", "json"}

    def __init__(self, Metadata, SourcePath="", Body=""):
        self.Metadata = Metadata
        self.SourcePath = SourcePath
        self.Body = Body or ""
        self.Name = Metadata["name"]
        self.Type = Metadata["type"]
        self.Description = Metadata.get("description", "") or ""
        self.Parameters = Metadata.get("parameters", {}) or {}
        self.ExecuteCode = Metadata.get("execute", "") or ""
        self.Trigger = Metadata.get("trigger", {}) or {}
        self.Permissions = Metadata.get("permissions", []) or []
        # Extended fields for richer AE scripts
        self.DependsOn = Metadata.get("depends_on", []) or []
        self.Emits = Metadata.get("emits", []) or []
        self.Agents = Metadata.get("agents", []) or []
        self.Version = Metadata.get("version", "1.0.0")
        self.Author = Metadata.get("author", "")
        self.Tags = Metadata.get("tags", []) or []
        # knowledge only: active=True is injected into the prompt every turn,
        # active=False (the default) is searchable but never injected.
        self.Active = bool(Metadata.get("active", False))

        # Taxonomy: category is explicit frontmatter first, else inferred from
        # the folder path. Tags are normalized (lowercase, sorted, unique) and
        # always include the category, so tag filters are always total.
        self.Category = _TaxNormalize(Metadata.get("category", ""), self.Type)
        if not Metadata.get("category") and SourcePath:
            Inferred, Sub = _TaxFromPath(SourcePath, self.Type)
            if Inferred and Inferred != "uncategorized":
                self.Category = Inferred
            self.Subcategory = Sub
        else:
            self.Subcategory = _TaxNormalize(Metadata.get("subcategory", ""), self.Type) or ""
        self.Tags = _TaxTags(Metadata.get("tags", []), self.Category)

    @property
    def IsCallable(self):
        return self.Type in self.CallableTypes

    @property
    def IsText(self):
        return self.Type in self.TextTypes

    @classmethod
    def FromFile(cls, FilePath):
        PathText = str(FilePath)
        with open(FilePath, "r", encoding="utf-8") as File:
            Content = File.read()
        return cls.FromString(Content, SourcePath=PathText)

    @classmethod
    def FromString(cls, Content, SourcePath=""):
        Lines = Content.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if not Lines:
            raise AEXError("An AE file cannot be empty.")
        if Lines[0].strip() != "---":
            # A header-only .ae (agent profiles like Being.ae) needs neither
            # the opening nor the closing marker -- it is pure YAML. Accept it
            # only when the whole file parses as a mapping with a valid type,
            # so a malformed script still reports the real syntax error.
            try:
                _Bare = yaml.safe_load("\n".join(Lines))
            except yaml.YAMLError as Ex:
                raise AEXError(f"An AE script must start with a YAML '---' frontmatter marker: {Ex}") from Ex
            if isinstance(_Bare, dict) and _Bare.get("type") in cls.Types:
                return cls(_Bare, SourcePath=SourcePath, Body="")
            raise AEXError("An AE script must start with a YAML '---' frontmatter marker.")

        EndIndex = next((Index for Index, Line in enumerate(Lines[1:], start=1) if Line.strip() == "---"), None)
        Body = ""
        if EndIndex is None:
            # Header-only data files (agent profiles like Being.ae) legitimately
            # have no body, so an unclosed frontmatter is not a syntax error --
            # but only when the whole rest of the file is valid YAML mapping.
            try:
                _Maybe = yaml.safe_load("\n".join(Lines[1:]))
            except yaml.YAMLError:
                raise AEXError("An AE script must close its YAML frontmatter with '---'.")
            if not isinstance(_Maybe, dict):
                raise AEXError("An AE script must close its YAML frontmatter with '---'.")
            EndIndex = len(Lines) - 1
        else:
            # Everything after the closing marker is the body: for text types
            # (workflow/knowledge/note/data) this IS the content; for callables
            # it is optional documentation.
            Body = "\n".join(Lines[EndIndex + 1:])

        try:
            Metadata = yaml.safe_load("\n".join(Lines[1:EndIndex])) or {}
        except yaml.YAMLError as Ex:
            raise AEXError(f"Invalid AE script YAML: {Ex}") from Ex
        if not isinstance(Metadata, dict):
            raise AEXError("AE script frontmatter must be a YAML object.")

        # YAML 1.1 parses an unquoted `on:` key as boolean True, which silently
        # breaks every event trigger. Normalize it back centrally so all existing
        # .ae files (and yaml.safe_dump output) keep working.
        Trigger = Metadata.get("trigger")
        if isinstance(Trigger, dict):
            # Handle YAML 1.1 boolean True quirk
            if "on" not in Trigger and True in Trigger:
                Trigger = {"on": Trigger[True], **{k: v for k, v in Trigger.items() if k is not True}}
            # Normalize `on` to always be a list of trigger dicts
            on_val = Trigger.get("on")
            top_condition = Trigger.get("condition")  # condition at top level applies to all triggers
            if isinstance(on_val, str):
                # Simple event name: convert to standard format
                Trigger["on"] = [{"type": "message", "match": on_val}]
            elif isinstance(on_val, dict):
                # Single trigger object: wrap in list
                Trigger["on"] = [on_val]
            elif isinstance(on_val, list):
                # Coerce bare-string entries: `- "name"` -> {"type": "message", ...}
                Trigger["on"] = [
                    {"type": "message", "match": t} if isinstance(t, str) else t
                    for t in on_val
                ]
            else:
                Trigger["on"] = []
            # Propagate top-level condition to each trigger
            if top_condition:
                for t in Trigger["on"]:
                    if isinstance(t, dict) and "condition" not in t:
                        t["condition"] = top_condition
            Metadata["trigger"] = Trigger

        Name = Metadata.get("name")
        ScriptType = Metadata.get("type")
        Description = Metadata.get("description")
        if not isinstance(Name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", Name):
            raise AEXError("AE script name must contain only letters, numbers, underscores, hyphens, or dots.")
        if ScriptType not in cls.Types:
            raise AEXError(f"AE script type must be one of: {', '.join(sorted(cls.Types))}.")

        if ScriptType not in cls.TextTypes and (not isinstance(Description, str) or not Description.strip()):
            raise AEXError("AE script requires a non-empty description.")

        Parameters = Metadata.get("parameters", {}) or {}
        if not isinstance(Parameters, dict):
            raise AEXError("AE script parameters must be a YAML object.")
        for ParameterName, Parameter in Parameters.items():
            if not isinstance(ParameterName, str) or not isinstance(Parameter, dict):
                raise AEXError("Each parameter must have a name and an object definition.")
            if Parameter.get("type", "string") not in cls.ParameterTypes:
                raise AEXError(f"Unsupported type for parameter '{ParameterName}'.")

        # Text types (workflow/knowledge/note/data) carry their content in the
        # body and must not smuggle in callables -- a skill is the automated
        # form of a workflow, so prose + execute in one file is the confusion
        # this separation exists to prevent.
        if ScriptType in cls.TextTypes:
            if str((Metadata.get("execute") or "")).strip():
                raise AEXError(f"A '{ScriptType}' file holds prose in its body, not code. Move the code into a tool/skill/ae file.")
            if Parameters:
                raise AEXError(f"A '{ScriptType}' file takes no parameters -- describe inputs in the body instead.")
            if ScriptType != "data" and not Body.strip():
                raise AEXError(f"A '{ScriptType}' file needs body text below the closing '---'.")

        Code = Metadata.get("execute", "") or ""
        if not isinstance(Code, str):
            raise AEXError("AE script execute must be a YAML literal string, usually written as 'execute: |'.")
        if Code.strip():
            try:
                compile(Code, SourcePath or f"<ae:{Name}>", "exec")
            except SyntaxError as Ex:
                raise AEXError(f"Invalid Python in AE script '{Name}': {Ex}") from Ex
            # Dead-function trap: `def execute(params): ... return ...` that is
            # never called leaves return_value None and the tool returns empty.
            # Module-level code must assign return_value (call it explicitly).
            import re as _re2
            if (_re2.search(r"^\s*def\s+execute\s*\(", Code, _re2.M)
                    and "return_value" not in Code):
                raise AEXError(f"AE script '{Name}' defines execute() but never assigns return_value — the tool would return empty. "
                               f"Either call it (return_value = execute(params)) or write module-level code ending in return_value = ... .")

        # Validate extended fields
        # depends_on: list of {name, type, version?}
        DependsOn = Metadata.get("depends_on", []) or []
        if not isinstance(DependsOn, list):
            raise AEXError("depends_on must be a list.")
        for dep in DependsOn:
            if not isinstance(dep, dict) or "name" not in dep or "type" not in dep:
                raise AEXError("Each depends_on entry must have 'name' and 'type'.")
            if dep["type"].lower() not in {t.lower() for t in cls.Types}:
                raise AEXError(f"depends_on type must be one of: {', '.join(sorted(cls.Types))}, got {dep['type']}.")

        # emits: list of event names
        Emits = Metadata.get("emits", []) or []
        if not isinstance(Emits, list):
            raise AEXError("emits must be a list.")
        for e in Emits:
            if not isinstance(e, str):
                raise AEXError("Each emits entry must be a string (event name).")

        # agents: list of {name, role} for raw AE type
        Agents = Metadata.get("agents", []) or []
        if not isinstance(Agents, list):
            raise AEXError("agents must be a list.")
        for a in Agents:
            if not isinstance(a, dict) or "name" not in a or "role" not in a:
                raise AEXError("Each agent entry must have 'name' and 'role'.")

        # permissions: list of strings
        Permissions = Metadata.get("permissions", []) or []
        if not isinstance(Permissions, list):
            raise AEXError("permissions must be a list.")
        for p in Permissions:
            if not isinstance(p, str):
                raise AEXError("Each permission must be a string.")

        return cls(Metadata, SourcePath=SourcePath, Body=Body)

    def FunctionSchema(self):
        Properties = {}
        Required = []
        TypeNames = {"json": "object"}
        for Name, Config in self.Parameters.items():
            Properties[Name] = {
                "type": TypeNames.get(Config.get("type", "string"), Config.get("type", "string")),
                "description": Config.get("description", ""),
            }
            if Config.get("required", False):
                Required.append(Name)
        return {
            "type": "function",
            "function": {
                "name": self.Name,
                "description": self.Description,
                "parameters": {
                    "type": "object",
                    "properties": Properties,
                    "required": Required,
                    "additionalProperties": False,
                },
            },
        }

    def ValidateArguments(self, Arguments):
        if not isinstance(Arguments, dict):
            raise AEXError("AE script arguments must be an object.")
        if not self.Parameters:
            return  # zero-param tools accept anything; extras are ignored, not errors
        Unknown = set(Arguments) - set(self.Parameters)
        if Unknown:
            raise AEXError(f"Unknown parameter(s): {', '.join(sorted(Unknown))}. Received keys: {sorted(Arguments)}. Valid keys: {sorted(self.Parameters)}. Resend with only valid keys.")
        for Name, Config in self.Parameters.items():
            if Config.get("required", False) and Name not in Arguments:
                raise AEXError(f"Missing required parameter: {Name}.")
            if Name in Arguments and not self._MatchesType(Arguments[Name], Config.get("type", "string")):
                raise AEXError(f"Parameter '{Name}' must have type {Config.get('type', 'string')}.")

    def Execute(self, Arguments, Registry, MemorySystem=None, Approved=False):
        self.ValidateArguments(Arguments)

        # Check permissions if needed
        if self.Permissions:
            for perm in self.Permissions:
                if not self._CheckPermission(perm, Registry, MemorySystem):
                    raise AEXError(f"Permission denied: {perm}")

        def CallTool(Name, **ToolArguments):
            """Call a tool through the registry (logs, approval, etc.).
            Falls back to skills: a TOOL may compose a SKILL by name —
            call_tool("my_skill", text="hi") == run_skill my_skill."""
            try:
                return Registry.Execute(Name, ToolArguments)
            except ValueError as _ex:
                if "Unknown tool" not in str(_ex):
                    raise
                # Skill fallback: is there a skill with this name?
                try:
                    _skills = {s.get("name") for s in Registry.GetSkills()}
                except Exception:
                    _skills = set()
                if Name not in _skills:
                    raise
                _res = Registry._RunSkill(Name, ToolArguments)
                if isinstance(_res, str):
                    return _res
                try:
                    return json.dumps(_res, ensure_ascii=False)
                except Exception:
                    return str(_res)

        def CallBuiltin(Name, **ToolArguments):
            """Call a builtin primitive directly (bypasses tool registration)."""
            return Registry.ExecutePrimitive(Name, ToolArguments, ApprovalGranted=Approved)

        def CallSkill(Name, parameters=None):
            """Execute a skill with parameters."""
            return Registry._RunSkill(Name, parameters or {})

        def LoadSkill(Name):
            """Load a skill's definition."""
            return Registry._LoadSkill(Name)

        def EmitEvent(Name, Data=None):
            """Emit an event to the event system."""
            return Registry.EmitEvent(Name, Data or {})

        def SpawnAgent(Name, Role, Prompt, Model=None):
            """Spawn a sub-agent for raw AE orchestration."""
            if not hasattr(Registry, "_SpawnAgent"):
                return {"error": "Agent spawning not available"}
            return Registry._SpawnAgent(Name, Role, Prompt, Model)

        def SendMessage(TargetAgent, Content):
            return Registry._SendMessage(TargetAgent, Content)

        def Retain(Text, Category="learned_lesson"):
            return Registry._RetainMemory(Text, Category) if hasattr(Registry, "_RetainMemory") else {"stored": False}

        def Recall(Query, Limit=5):
            return Registry._RecallMemory(Query, Limit) if hasattr(Registry, "_RecallMemory") else []

        def ToolHistory(Limit=30):
            """Recent tool usage: [{age_seconds, tool, skill, error}]."""
            try:
                return Registry.ToolHistory(Limit)
            except Exception:
                return []

        def WaitForEvent(EventName, Timeout=60):
            """Wait for an event to be emitted (blocking)."""
            if not hasattr(Registry, "_WaitForEvent"):
                return {"error": "Event waiting not available"}
            return Registry._WaitForEvent(EventName, Timeout)

        # Cooperative execution tracking: long-running .ae (loops, time
        # tracking) can report progress and honor stop requests from
        # stop_execution — polled from any thread (e.g. next turn).
        # Nested executions reuse the outer record (single stop/report id).
        _Enter = getattr(Registry, "_ExecEnter", None) or getattr(Registry, "_ExecBegin", None)
        try:
            _ExecId = _Enter(self.Name, self.Type) if callable(_Enter) else None
        except Exception:
            _ExecId = None

        def ShouldStop():
            """True when stop_execution() was called for this run."""
            try:
                return bool(Registry._ExecStopRequested(_ExecId)) if _ExecId else False
            except Exception:
                return False

        def Report(Data=None):
            """Relay progress (visible via `executions`); raises if stopped."""
            try:
                if _ExecId:
                    Registry._ExecReport(_ExecId, Data)
            except Exception:
                pass
            if ShouldStop():
                raise AEXStopped(f"stop requested: {self.Name}")
            return {"reported": True}

        Context = {
            "params": Arguments,
            "parameters": Arguments,  # alias: showcase convention uses `parameters`
            "data": Arguments,  # alias: event blocks use `data`
            "workspace": str(getattr(Registry, "ProjectRoot", "") or ""),
            "memory": MemorySystem,
            "agent_name": getattr(Registry, "CurrentAgentName", "*"),
            "tools": Registry.Tools,
            "skills": {Skill["name"]: Skill for Skill in Registry.GetSkills()},
            "events": {"emit": EmitEvent},
            "config": self.Metadata.get("config", {}),
            "context": None,
            "datetime": datetime,
            "time": time,
            "json": json,
            "call_tool": CallTool,
            "call_builtin": CallBuiltin,
            "call_skill": CallSkill,
            "load_skill": LoadSkill,
            "emit_event": EmitEvent,
            "spawn_agent": SpawnAgent,
            "send_message": SendMessage,
            "wait_for_event": WaitForEvent,
            "retain": Retain,
            "recall": Recall,
            "tool_history": ToolHistory,
            "exec_id": _ExecId,
            "should_stop": ShouldStop,
            "report": Report,
            "return_value": None,
        }
        Context["context"] = Context
        Context["body"] = self.Body
        try:
            exec(compile(self.ExecuteCode, self.SourcePath or f"<ae:{self.Name}>", "exec"), Context)
        except AEXStopped as _Stopped:
            return {"status": "stopped", "name": self.Name, "detail": str(_Stopped)}
        finally:
            try:
                if _ExecId:
                    _Exit = getattr(Registry, "_ExecExit", None) or getattr(Registry, "_ExecEnd", None)
                    if callable(_Exit):
                        _Exit(_ExecId)
            except Exception:
                pass
        if Context["return_value"] is not None:
            return Context["return_value"]
        return {"status": "executed", "name": self.Name}

    def _CheckPermission(self, perm, Registry, MemorySystem):
        """Check if the script has a required permission."""
        # For now, always allow. In future, integrate with approval system.
        allowed = {
            "network", "read_files", "write_files", "execute_commands",
            "read_memory", "write_memory", "spawn_agents"
        }
        return perm in allowed

    @classmethod
    def _MatchesType(cls, Value, TypeName):
        if TypeName == "string":
            return isinstance(Value, str)
        if TypeName == "number":
            return isinstance(Value, (int, float)) and not isinstance(Value, bool)
        if TypeName == "integer":
            return isinstance(Value, int) and not isinstance(Value, bool)
        if TypeName == "boolean":
            return isinstance(Value, bool)
        if TypeName == "array":
            return isinstance(Value, list)
        if TypeName in {"object", "json"}:
            return isinstance(Value, dict)
        return False