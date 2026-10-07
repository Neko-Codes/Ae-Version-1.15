import datetime
import json
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path


_SAVE_LOCK = threading.RLock()  # shared memory.json across main thread + delegation threads


class Memory:
    def __init__(self, BasePath=".", ShortTermLimit=8, SummaryThreshold=5, RecentKeep=2, MemoryFile="memory.json"):
        self.BasePath = Path(BasePath)
        self.BasePath.mkdir(parents=True, exist_ok=True)

        self.ShortTermLimit = ShortTermLimit
        self.SummaryThreshold = SummaryThreshold
        self.RecentKeep = RecentKeep
        self.MemoryFile = self.BasePath / MemoryFile
        # Append-only write-ahead log: every mutation lands here first, so a
        # crash between saves loses nothing. Save() snapshots + truncates it.
        self.WalFile = self.BasePath / "memory-wal.jsonl"
        self._replaying = False

        self.ShortTerm = deque(maxlen=self.ShortTermLimit)
        self.ToolEvents = deque(maxlen=30)
        self.Todos = []
        self.Summary = ""
        self.LongTerm = []
        self._short_term_count = 0
        self._tool_events_count = 0
        self._long_term_count = 0
        self._todos_count = 0
        self._message_id_seq = 0
        self._last_message_time = None
        self._last_user_message_id = None
        self._last_agent_message_id = None
        self._last_tool_used = None
        self._last_tool_error = None
        self._last_action_taken = None
        self._last_error = None

        self.Load()

    @property
    def ShortTermCount(self):
        return self._short_term_count

    @property
    def ToolEventsCount(self):
        return self._tool_events_count

    @property
    def LongTermCount(self):
        return self._long_term_count

    @property
    def TodosCount(self):
        return self._todos_count

    def Load(self):
        if not self.MemoryFile.exists():
            return

        try:
            with self.MemoryFile.open("r", encoding="utf-8") as File:
                Data = json.load(File)
        except json.JSONDecodeError:
            return

        self.ShortTerm = deque(Data.get("short_term", []), maxlen=self.ShortTermLimit)
        self.ToolEvents = deque(Data.get("tool_events", []), maxlen=30)
        self.Todos = Data.get("todos", [])
        self.Summary = Data.get("summary", "")
        self.LongTerm = Data.get("long_term", [])
        self._ReplayWal()
        self._short_term_count = len(self.ShortTerm)
        self._tool_events_count = len(self.ToolEvents)
        self._long_term_count = len(self.LongTerm)
        self._todos_count = len(self.Todos)

    def _WalAppend(self, op, **fields):
        """Append one mutation record. Skipped during replay."""
        if self._replaying:
            return
        try:
            with self.WalFile.open("a", encoding="utf-8") as File:
                File.write(json.dumps({"op": op, **fields}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _ReplayWal(self, limit=5000):
        """Re-apply mutations logged after the last snapshot."""
        if not self.WalFile.is_file():
            return
        try:
            Lines = self.WalFile.read_text(encoding="utf-8").splitlines()[-limit:]
        except Exception:
            return
        self._replaying = True
        try:
            for Line in Lines:
                try:
                    Rec = json.loads(Line)
                except Exception:
                    continue
                Op = Rec.get("op")
                if Op == "message":
                    self.ShortTerm.append({"role": Rec.get("role", "user"), "content": Rec.get("content", "")})
                elif Op == "tool":
                    self.ToolEvents.append({
                        "name": Rec.get("name", ""), "arguments": Rec.get("arguments", {}),
                        "result": Rec.get("result", ""), "is_error": Rec.get("is_error", False),
                        "timestamp": Rec.get("timestamp", time.time()),
                    })
                elif Op == "longterm":
                    Entry = Rec.get("entry", {})
                    if Entry and not any(e.get("id") == Entry.get("id") for e in self.LongTerm):
                        self.LongTerm.append(Entry)
                elif Op == "longterm_meta":
                    for e in self.LongTerm:
                        if e.get("id") == Rec.get("id"):
                            e["metadata"].update(Rec.get("metadata", {}))
                            break
                elif Op == "forget":
                    Gone = set(Rec.get("ids", []))
                    self.LongTerm = [e for e in self.LongTerm if e.get("id") not in Gone]
                elif Op == "todos":
                    if isinstance(Rec.get("todos"), list):
                        self.Todos = Rec["todos"]
                elif Op == "compact":
                    self.Summary = Rec.get("summary", self.Summary)
                    Short = Rec.get("short", [])
                    if isinstance(Short, list):
                        self.ShortTerm = deque(Short, maxlen=self.ShortTermLimit)
        finally:
            self._replaying = False

    def Save(self):
        Data = {
            "short_term": list(self.ShortTerm),
            "tool_events": list(self.ToolEvents),
            "todos": self.Todos,
            "summary": self.Summary,
            "long_term": self.LongTerm,
        }
        with _SAVE_LOCK:
            Temporary = self.MemoryFile.with_suffix(".tmp")
            with Temporary.open("w", encoding="utf-8") as File:
                json.dump(Data, File, indent=2)
            Temporary.replace(self.MemoryFile)
            # Snapshot now subsumes the WAL — truncate it (bounded growth).
            if not self._replaying:
                try:
                    self.WalFile.write_text("", encoding="utf-8")
                except Exception:
                    pass

    def AddMessage(self, Role, Content):
        self._message_id_seq += 1
        MsgId = self._message_id_seq
        now = datetime.datetime.now()
        self._last_message_time = now
        RoleL = (Role or "").lower()
        if RoleL == "user":
            self._last_user_message_id = MsgId
        elif RoleL == "assistant":
            self._last_agent_message_id = MsgId
        Message = {"role": Role, "content": Content, "id": MsgId, "ts": now.isoformat()}
        self._WalAppend("message", role=Role, content=Content, id=MsgId)
        self.ShortTerm.append(Message)
        self._short_term_count = len(self.ShortTerm)
        if self._short_term_count >= self.SummaryThreshold:
            self.Compact()
        self.Save()

    def Clear(self):
        """Wipe all mutable memory: chat, digest, long-term, tool history, todos, WAL."""
        self.ShortTerm = deque(maxlen=self.ShortTermLimit)
        self.ToolEvents = deque(maxlen=30)
        self.Todos = []
        self.Summary = ""
        self.LongTerm = []
        self._short_term_count = 0
        self._tool_events_count = 0
        self._long_term_count = 0
        self._todos_count = 0
        self._message_id_seq = 0
        self._last_message_time = None
        self._last_user_message_id = None
        self._last_agent_message_id = None
        self._last_tool_used = None
        self._last_tool_error = None
        self._last_action_taken = None
        self._last_error = None
        try:
            if self.WalFile.is_file():
                self.WalFile.write_text("", encoding="utf-8")
        except Exception:
            pass
        self.Save()
        return {"cleared": True}

    def Compact(self, keep=2):
        """Fold older short-term messages into a structured digest.

        Keeps the newest `keep` messages live; everything older is distilled
        into Summary (decisions, tool outcomes, open threads) instead of a
        raw 300-char cut. Returns counts. Safe to call anytime (.compact)."""
        try:
            keep = max(0, min(int(keep), len(self.ShortTerm)))
        except Exception:
            keep = 2
        if len(self.ShortTerm) <= keep:
            return {"evicted": 0, "kept": len(self.ShortTerm)}
        Evicted = list(self.ShortTerm)[:len(self.ShortTerm) - keep]
        Kept = list(self.ShortTerm)[len(self.ShortTerm) - keep:]
        Digest = self._DigestMessages(Evicted)
        if Digest:
            if self.Summary and len(self.Summary) < 3000:
                self.Summary = (self.Summary + "\n" + Digest)[-3000:]
            else:
                self.Summary = Digest[-3000:]
        self.ShortTerm = deque(Kept, maxlen=self.ShortTermLimit)
        self._short_term_count = len(self.ShortTerm)
        self._WalAppend("compact", summary=self.Summary, short=list(self.ShortTerm))
        self.Save()
        return {"evicted": len(Evicted), "kept": len(Kept)}

    def _DigestMessages(self, Messages):
        """Extractive digest: user goals, tool outcomes, decisions, open threads."""
        Users, Tools, Assistants = [], [], []
        for Item in Messages:
            Role = Item.get("role", "")
            Text = re.sub(r"\s+", " ", str(Item.get("content", ""))).strip()
            if not Text:
                continue
            if Role == "user":
                if Text.startswith(("[System]", "[Steering", "[User steering", "Circuit breaker")):
                    continue  # scaffolding, not substance
                Users.append(Text[:300])
            elif Role == "tool":
                Tools.append(Text[:200])
            else:
                Assistants.append(Text[:300])
        Parts = []
        if Users:
            Parts.append("Goals/requests: " + " | ".join(Users[-5:]))
        if Tools:
            Parts.append(f"Tool activity ({len(Tools)} calls): " + " | ".join(Tools[:8]))
        if Assistants:
            Last = Assistants[-1]
            Parts.append("Latest conclusion: " + Last)
            if len(Assistants) > 1:
                Parts.append("Earlier: " + " | ".join(Assistants[:-1][:3]))
        Digest = "\n".join(f"- {P}" for P in Parts)
        return Digest[:2000]

    def _BuildSummary(self):
        # Legacy hook: delegate to the digest (kept for backward compat).
        return self._DigestMessages(list(self.ShortTerm))[:2000]

    def GetContext(self, AnchorTakeover=0):
        # AnchorTakeover: number of trailing messages the caller re-sends itself
        # (BuildPrompt's "Immediate previous exchange"). Skip them here or the
        # same exchange is transmitted twice, doubling its token cost.
        Recent = list(self.ShortTerm)
        if AnchorTakeover > 0:
            Recent = Recent[:-AnchorTakeover] if len(Recent) > AnchorTakeover else []
        Parts = []

        # The digest's "Latest conclusion" is just the final assistant message,
        # which the anchor also carries — exclude the takeover window so it is
        # not stated twice (the user saw it three times: digest + Recent chat +
        # anchor).
        if self.Summary:
            Summary = self._DigestMessages(Recent)
            if Summary:
                Parts.append("Summary:\n" + Summary)

        if Recent:
            RecentText = "\n".join(f"{Item['role']}: {Item['content']}" for Item in Recent[-self.RecentKeep:])
            Parts.append("Recent chat:\n" + RecentText)

        return "\n\n".join(Parts)

    def AddToolEvent(self, ToolName, Arguments, Result, IsError=False):
        SafeArguments = {}
        for Key, Value in Arguments.items():
            Text = Value if isinstance(Value, str) else json.dumps(Value, ensure_ascii=False)
            SafeArguments[Key] = Text if len(Text) <= 300 else Text[:300] + "..."

        ResultText = str(Result)
        if len(ResultText) > 1200:
            ResultText = ResultText[:1200] + "..."
        try:
            self._message_id_seq += 1
            tid = self._message_id_seq
        except Exception:
            tid = len(self.ToolEvents) + 1
        if IsError:
            try:
                err = str(Result)
                if len(err) > 100:
                    err = err[:100]
                self._last_tool_error = (err, tid)
                self._last_error = (err, tid)
            except Exception:
                pass
        else:
            try:
                self._last_tool_used = (ToolName, tid)
            except Exception:
                pass
        try:
            self._last_action_taken = (ToolName, tid)
        except Exception:
            pass
        self._WalAppend("tool", name=ToolName, arguments=SafeArguments,
                        result=ResultText, is_error=IsError, timestamp=time.time())
        self.ToolEvents.append({
            "name": ToolName,
            "arguments": SafeArguments,
            "result": ResultText,
            "is_error": IsError,
            "timestamp": time.time(),
            "ts": datetime.datetime.now().isoformat(),
            "id": tid,
        })
        self._tool_events_count = len(self.ToolEvents)
        self.Save()

    def GetRecentArtifacts(self, Limit=8):
        """Return paths from recent successful workspace writes, newest first."""
        Artifacts = []
        Seen = set()
        for Event in reversed(self.ToolEvents):
            if Event.get("is_error") or Event.get("name") not in {"write_file", "patch_file"}:
                continue
            Arguments = Event.get("arguments", {})
            PathText = str(Arguments.get("path", "")).strip() if isinstance(Arguments, dict) else ""
            if not PathText or PathText in Seen:
                continue
            Seen.add(PathText)
            Artifacts.append(PathText)
            if len(Artifacts) >= max(1, int(Limit)):
                break
        return Artifacts

    def GetToolContext(self, Limit=3):
        RecentEvents = list(self.ToolEvents)[-Limit:]
        # Filter out denied/permission errors to avoid re-prompting the model
        Filtered = []
        for Event in RecentEvents:
            Result = str(Event.get('result', ''))
            ToolName = Event.get('name', '')
            IsError = Event.get('is_error', False)
            if 'Denied' in Result or 'Denied by user' in Result or 'PermissionError' in Result:
                continue
            if ToolName == 'write_file' and IsError:
                continue
            Filtered.append(Event)
        # Hard caps: one bloated call (e.g. a whole skill file in `content`)
        # must never dominate the prompt.
        Lines = []
        for Event in Filtered:
            Args = json.dumps(Event['arguments'], ensure_ascii=False)
            if len(Args) > 200:
                Args = Args[:200] + "..."
            Res = str(Event['result'])
            if len(Res) > 300:
                Res = Res[:300] + "..."
            Lines.append(f"- {Event['name']}({Args}): {Res}")
        return "\n".join(Lines)

    def GetRecentResearch(self, Limit=6, MaxChars=6000):
        """Return bounded successful search/fetch evidence for explicit follow-ups."""
        Lines = []
        SeenUrls = set()
        for Event in reversed(self.ToolEvents):
            if Event.get("is_error") or Event.get("name") not in {"web_search", "web_fetch", "fetch_many"}:
                continue
            Name = Event.get("name", "")
            Arguments = Event.get("arguments", {})
            Result = str(Event.get("result", "")).strip()
            if Name == "web_search":
                Query = str(Arguments.get("query", "")) if isinstance(Arguments, dict) else ""
                Block = f"Search query: {Query}\n{Result[:900]}"
            else:
                RawUrls = Arguments.get("url", "") if Name == "web_fetch" and isinstance(Arguments, dict) else Arguments.get("urls", []) if isinstance(Arguments, dict) else []
                if isinstance(RawUrls, str):
                    try:
                        RawUrls = json.loads(RawUrls)
                    except Exception:
                        RawUrls = [RawUrls]
                if not isinstance(RawUrls, list):
                    RawUrls = [RawUrls]
                Urls = [str(Url).strip() for Url in RawUrls if str(Url).strip()]
                FreshUrls = [Url for Url in Urls if Url not in SeenUrls]
                if not FreshUrls:
                    continue
                SeenUrls.update(FreshUrls)
                Block = f"Fetched source(s): {', '.join(FreshUrls[:5])}\n{Result[:1200]}"
            Lines.append(Block)
            if len(Lines) >= max(1, int(Limit)) or sum(map(len, Lines)) >= MaxChars:
                break
        return "\n\n".join(Lines)[:MaxChars]

    def GetInstructionContext(self):
        Rules = []
        for Entry in self.LongTerm:
            Meta = Entry.get("metadata", {})
            if Meta.get("type") in {"instruction", "preference", "user_fact"}:
                Rules.append(Entry.get("text", ""))
        return "\n".join(f"- {Rule}" for Rule in Rules[:10])

    def AddLongTerm(self, Text, Metadata=None):
        NormalizedText = re.sub(r"\s+", " ", Text).strip().casefold()
        for Entry in self.LongTerm:
            ExistingText = re.sub(r"\s+", " ", Entry.get("text", "")).strip().casefold()
            if ExistingText == NormalizedText:
                Entry["metadata"].update(Metadata or {})
                self._WalAppend("longterm_meta", id=Entry["id"], metadata=Metadata or {})
                self.Save()
                return Entry

        Entry = {
            "id": uuid.uuid4().hex[:12],
            "text": Text.strip(),
            "metadata": Metadata or {},
        }
        self._WalAppend("longterm", entry=Entry)
        self.LongTerm.append(Entry)
        self._long_term_count = len(self.LongTerm)
        self.Save()
        return Entry

    def ForgetLongTerm(self, Query):
        NormalizedQuery = re.sub(r"\s+", " ", Query).strip().casefold()
        if not NormalizedQuery:
            return 0

        Matches = [
            Entry for Entry in self.LongTerm
            if Entry.get("id") == Query
            or re.sub(r"\s+", " ", Entry.get("text", "")).strip().casefold() == NormalizedQuery
        ]
        if not Matches:
            return 0

        MatchIds = {id(Entry) for Entry in Matches}
        MatchIds = {id(Entry) for Entry in Matches}
        Gone = [Entry.get("id") for Entry in Matches]
        self._WalAppend("forget", ids=Gone)
        self.LongTerm = [Entry for Entry in self.LongTerm if id(Entry) not in MatchIds]
        self._long_term_count = len(self.LongTerm)
        self.Save()
        return len(Matches)

    TodoStatuses = {"todo", "in_progress", "blocked", "done", "cancelled"}

    def _TodoChildren(self, TaskId):
        return [t for t in self.Todos if t.get("parent") == TaskId]

    def _TodoIsPhase(self, Todo):
        return bool(self._TodoChildren(Todo.get("id")))

    def _RollupTodos(self):
        """Phases derive their status: all children done/cancelled -> done."""
        changed = True
        while changed:
            changed = False
            for Todo in self.Todos:
                Kids = self._TodoChildren(Todo.get("id"))
                if Kids and all(k.get("status") in {"done", "cancelled"} for k in Kids):
                    if Todo.get("status") != "done":
                        Todo["status"] = "done"
                        changed = True

    def _TodoCounts(self, Todo):
        Kids = self._TodoChildren(Todo.get("id"))
        if not Kids:
            return (1 if Todo.get("status") == "done" else 0, 1)
        Done, Total = 0, 0
        for Kid in Kids:
            if self._TodoChildren(Kid.get("id")):
                d, t = self._TodoCounts(Kid)
            else:
                d, t = (1 if Kid.get("status") == "done" else 0, 1)
            Done, Total = Done + d, Total + t
        return Done, Total

    def TodoTree(self):
        """Unicode tree (model + logs): phases with done/total, status glyphs."""
        Glyph = {"todo": "◇", "in_progress": "◐", "blocked": "⊘", "done": "●", "cancelled": "×"}
        Lines = ["Todo"]
        Roots = [t for t in self.Todos if not t.get("parent") or not any(x.get("id") == t.get("parent") for x in self.Todos)]
        def Render(Node, Prefix, Last):
            Kids = self._TodoChildren(Node.get("id"))
            Bar = "└─ " if Last else "├─ "
            if Kids:
                Done, Total = self._TodoCounts(Node)
                Lines.append(f"{Prefix}{Bar}{Node.get('title', '')} · {Done}/{Total}")
                Ext = "   " if Last else "│  "
                for i, Kid in enumerate(Kids):
                    Render(Kid, Prefix + Ext, i == len(Kids) - 1)
            else:
                Extra = ""
                if Node.get("priority", "normal") != "normal":
                    Extra += f" [{Node.get('priority')}]"
                if Node.get("assignee", "*") not in ("*", ""):
                    Extra += f" @{Node.get('assignee')}"
                Lines.append(f"{Prefix}{Bar}{Glyph.get(Node.get('status', 'todo'), '◇')} {Node.get('title', '')}{Extra}")
        for i, Root in enumerate(Roots):
            Render(Root, "", i == len(Roots) - 1)
        return "\n".join(Lines)

    def ManageTodos(self, Action, TaskId=None, Title=None, Description=None, Status=None, Priority=None, Assignee=None, Parent=None, Plan=None, Dlg=None):
        if Action == "list":
            return self.Todos
        if Action == "add":
            if not Title or not Title.strip():
                raise ValueError("A task title is required.")
            if Status is not None and Status not in self.TodoStatuses:
                raise ValueError(f"status must be one of: {', '.join(sorted(self.TodoStatuses))}.")
            if Parent is not None and not any(t.get("id") == Parent for t in self.Todos):
                raise ValueError(f"Parent todo not found: {Parent}")
            Todo = {
                "id": uuid.uuid4().hex[:12],
                "title": Title.strip(),
                "description": (Description or "").strip(),
                "status": Status or "todo",
                "priority": Priority or "normal",
                "assignee": (Assignee or "").strip() or "*",
                "parent": Parent,
                "dlg": Dlg,
            }
            self.Todos.append(Todo)
            self._todos_count = len(self.Todos)
            self._WalAppend("todos", todos=self.Todos)
            self.Save()
            return Todo
        if Action == "plan":
            # One call builds a whole tree: [{title, status?, priority?,
            # assignee?, description?, subtasks: [...]}, ...]
            if not isinstance(Plan, list) or not Plan:
                raise ValueError("plan must be a non-empty array of {title, subtasks?} nodes.")
            if Parent is not None and not any(t.get("id") == Parent for t in self.Todos):
                raise ValueError(f"Parent todo not found: {Parent}")
            Created = []
            def Build(Nodes, ParentId):
                for Node in Nodes:
                    if not isinstance(Node, dict) or not isinstance(Node.get("title", ""), str) or not Node["title"].strip():
                        raise ValueError("Each plan node needs a non-empty title.")
                    NodeStatus = Node.get("status", "todo")
                    if NodeStatus not in self.TodoStatuses:
                        raise ValueError(f"status must be one of: {', '.join(sorted(self.TodoStatuses))}.")
                    NodePriority = Node.get("priority", "normal") or "normal"
                    if NodePriority not in {"low", "normal", "high", "urgent"}:
                        raise ValueError("priority must be low, normal, high, or urgent.")
                    Todo = {
                        "id": uuid.uuid4().hex[:12],
                        "title": Node["title"].strip(),
                        "description": str(Node.get("description", "") or "").strip(),
                        "status": NodeStatus,
                        "priority": NodePriority,
                        "assignee": str(Node.get("assignee", "") or "").strip() or "*",
                        "parent": ParentId,
                        "dlg": Dlg,
                    }
                    Created.append(Todo)
                    Subs = Node.get("subtasks", []) or []
                    if Subs:
                        if not isinstance(Subs, list):
                            raise ValueError(f"subtasks of '{Todo['title']}' must be an array.")
                        Build(Subs, Todo["id"])
            Build(Plan, Parent)
            self.Todos.extend(Created)
            self._RollupTodos()
            self._todos_count = len(self.Todos)
            self._WalAppend("todos", todos=self.Todos)
            self.Save()
            return {"created": len(Created), "tasks": Created, "tree": self.TodoTree()}

        Todo = next((Item for Item in self.Todos if Item.get("id") == TaskId), None)
        if Todo is None and TaskId:
            # Leniency: models truncate IDs (85d5b0b1 for 85d5b0b187ff) — unique prefix match.
            Cands = [Item for Item in self.Todos if str(Item.get("id", "")).startswith(str(TaskId))]
            if len(Cands) == 1:
                Todo = Cands[0]
        if Todo is None:
            raise ValueError(f"Todo not found: {TaskId}")
        def _CascadeDone(RootId):
            # Marking a phase done completes its whole subtree (explicit intent).
            for Kid in self._TodoChildren(RootId):
                Kid["status"] = "done"
                _CascadeDone(Kid.get("id"))
        if Action == "complete":
            Todo["status"] = "done"
            _CascadeDone(TaskId)
        elif Action == "update":
            if Title is not None:
                Todo["title"] = Title.strip()
            if Description is not None:
                Todo["description"] = Description.strip()
            if Status is not None:
                if Status not in self.TodoStatuses:
                    raise ValueError(f"status must be one of: {', '.join(sorted(self.TodoStatuses))}.")
                Todo["status"] = Status
                if Status == "done":
                    _CascadeDone(TaskId)
            if Priority is not None:
                Todo["priority"] = Priority
            if Assignee is not None:
                Todo["assignee"] = Assignee.strip() or "*"
        elif Action == "delete":
            # Cascade: removing a phase removes its whole subtree.
            Gone = {TaskId}
            while True:
                Kids = [t.get("id") for t in self.Todos if t.get("parent") in Gone and t.get("id") not in Gone]
                if not Kids:
                    break
                Gone.update(Kids)
            self.Todos = [t for t in self.Todos if t.get("id") not in Gone]
            self._todos_count = len(self.Todos)
            self._WalAppend("todos", todos=self.Todos)
            self.Save()
            return {"deleted": TaskId, "cascade": len(Gone) - 1}
        else:
            raise ValueError(f"Unknown todo action: {Action}")
        self._RollupTodos()
        self._WalAppend("todos", todos=self.Todos)
        self.Save()
        return Todo

    def AddInstruction(self, Text):
        self.AddLongTerm(Text, {"type": "instruction"})

    def SearchLongTerm(self, Query, Limit=5):
        QueryWords = self._Tokenize(Query)
        if not QueryWords:
            return []

        Results = []
        for Entry in self.LongTerm:
            Text = Entry.get("text", "")
            Meta = Entry.get("metadata", {})
            if Meta.get("type") == "instruction":
                continue
            TextWords = self._Tokenize(Text)
            Score = 0
            for Word in QueryWords:
                if Word in TextWords:
                    Score += 2
                if Word in Text.lower():
                    Score += 0.5
            if Score > 0:
                Results.append({"text": Text, "metadata": Meta, "score": Score})

        Results.sort(key=lambda Item: Item["score"], reverse=True)
        return Results[:Limit]

    def _Tokenize(self, Text):
        return re.findall(r"[a-zA-Z0-9_]+", Text.lower())
