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
        Message = {"role": Role, "content": Content}
        self._WalAppend("message", role=Role, content=Content)
        self.ShortTerm.append(Message)
        self._short_term_count = len(self.ShortTerm)
        if self._short_term_count >= self.SummaryThreshold:
            self.Compact()
        self.Save()

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
            Parts.append("Goals/requests: " + " | ".join(Users[:5]))
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
        # AnchorTakeover: trailing messages the caller re-sends itself
        # (BuildPrompt's "Immediate previous exchange"). Skip them here or the
        # same exchange is transmitted twice, doubling its token cost.
        Recent = list(self.ShortTerm)
        if AnchorTakeover > 0:
            Recent = Recent[:-AnchorTakeover] if len(Recent) > AnchorTakeover else []
        Parts = []

        # The digest's "Latest conclusion" is just the final assistant message,
        # which the anchor also carries — exclude the takeover window so it is
        # not stated twice.
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
        self._WalAppend("tool", name=ToolName, arguments=SafeArguments,
                        result=ResultText, is_error=IsError, timestamp=time.time())
        self.ToolEvents.append({
            "name": ToolName,
            "arguments": SafeArguments,
            "result": ResultText,
            "is_error": IsError,
            "timestamp": time.time(),
        })
        self._tool_events_count = len(self.ToolEvents)
        self.Save()

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
        return "\n".join(
            f"- {Event['name']}({json.dumps(Event['arguments'], ensure_ascii=False)}): {Event['result']}"
            for Event in Filtered
        )

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
        return len(Matches)

    def ManageTodos(self, Action, TaskId=None, Title=None, Description=None, Status=None, Priority=None, Assignee=None):
        if Action == "list":
            return self.Todos
        if Action == "add":
            if not Title or not Title.strip():
                raise ValueError("A task title is required.")
            Todo = {
                "id": uuid.uuid4().hex[:12],
                "title": Title.strip(),
                "description": (Description or "").strip(),
                "status": Status or "todo",
                "priority": Priority or "normal",
                "assignee": (Assignee or "").strip() or "*",
            }
            self.Todos.append(Todo)
            self._todos_count = len(self.Todos)
            self._WalAppend("todos", todos=self.Todos)
            self.Save()
            return Todo

        Todo = next((Item for Item in self.Todos if Item.get("id") == TaskId), None)
        if Todo is None:
            raise ValueError(f"Todo not found: {TaskId}")
        if Action == "complete":
            Todo["status"] = "done"
        elif Action == "update":
            if Title is not None:
                Todo["title"] = Title.strip()
            if Description is not None:
                Todo["description"] = Description.strip()
            if Status is not None:
                Todo["status"] = Status
            if Priority is not None:
                Todo["priority"] = Priority
            if Assignee is not None:
                Todo["assignee"] = Assignee.strip() or "*"
        elif Action == "delete":
            self.Todos.remove(Todo)
            self._todos_count = len(self.Todos)
            self._WalAppend("todos", todos=self.Todos)
            self.Save()
            return {"deleted": TaskId}
        else:
            raise ValueError(f"Unknown todo action: {Action}")
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
