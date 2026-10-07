import datetime
import json
import re
import time

import yaml


class AEXError(ValueError):
    pass


class AEXScript:
    Types = {"tool", "skill", "event", "ae"}
    ParameterTypes = {"string", "number", "integer", "boolean", "array", "object", "json"}

    def __init__(self, Metadata, SourcePath=""):
        self.Metadata = Metadata
        self.SourcePath = SourcePath
        self.Name = Metadata["name"]
        self.Type = Metadata["type"]
        self.Description = Metadata["description"]
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

    @classmethod
    def FromFile(cls, FilePath):
        PathText = str(FilePath)
        with open(FilePath, "r", encoding="utf-8") as File:
            Content = File.read()
        return cls.FromString(Content, SourcePath=PathText)

    @classmethod
    def FromString(cls, Content, SourcePath=""):
        Lines = Content.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if not Lines or Lines[0].strip() != "---":
            raise AEXError("An AE script must start with a YAML '---' frontmatter marker.")

        EndIndex = next((Index for Index, Line in enumerate(Lines[1:], start=1) if Line.strip() == "---"), None)
        if EndIndex is None:
            raise AEXError("An AE script must close its YAML frontmatter with '---'.")

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
        if not isinstance(Name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", Name):
            raise AEXError("AE script name must contain only letters, numbers, underscores, or hyphens.")
        if ScriptType not in cls.Types:
            raise AEXError(f"AE script type must be one of: {', '.join(sorted(cls.Types))}.")
        if not isinstance(Description, str) or not Description.strip():
            raise AEXError("AE script requires a non-empty description.")

        Parameters = Metadata.get("parameters", {}) or {}
        if not isinstance(Parameters, dict):
            raise AEXError("AE script parameters must be a YAML object.")
        for ParameterName, Parameter in Parameters.items():
            if not isinstance(ParameterName, str) or not isinstance(Parameter, dict):
                raise AEXError("Each parameter must have a name and an object definition.")
            if Parameter.get("type", "string") not in cls.ParameterTypes:
                raise AEXError(f"Unsupported type for parameter '{ParameterName}'.")

        Code = Metadata.get("execute", "") or ""
        if not isinstance(Code, str):
            raise AEXError("AE script execute must be a YAML literal string, usually written as 'execute: |'.")
        if Code.strip():
            try:
                compile(Code, SourcePath or f"<ae:{Name}>", "exec")
            except SyntaxError as Ex:
                raise AEXError(f"Invalid Python in AE script '{Name}': {Ex}") from Ex

        # Validate extended fields
        # depends_on: list of {name, type, version?}
        DependsOn = Metadata.get("depends_on", []) or []
        if not isinstance(DependsOn, list):
            raise AEXError("depends_on must be a list.")
        for dep in DependsOn:
            if not isinstance(dep, dict) or "name" not in dep or "type" not in dep:
                raise AEXError("Each depends_on entry must have 'name' and 'type'.")
            if dep["type"] not in {"TOOL", "SKILL", "EVENT", "AE", "tool", "skill", "event", "ae"}:
                raise AEXError(f"depends_on type must be TOOL/SKILL/EVENT/AE, got {dep['type']}.")

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

        return cls(Metadata, SourcePath=SourcePath)

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
            """Call a tool through the registry (logs, approval, etc.)."""
            return Registry.Execute(Name, ToolArguments)

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
            "return_value": None,
        }
        Context["context"] = Context
        exec(compile(self.ExecuteCode, self.SourcePath or f"<ae:{self.Name}>", "exec"), Context)
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