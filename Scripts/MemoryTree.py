"""Memory nodes: a hierarchy of independent memory stores.

Every agent -- and every subagent an agent delegates to -- owns a private
memory. Nodes are addressed by a dotted path so the whole tree is greppable:

    1:AE
    1.1:AE-worker-1
    1.2:AE-worker-2
    2:CodeBuddy
    2.1:CodeBuddy-worker-1

A subagent's memory is its own: it reads and writes only its own store, and its
transcript is never spliced into the delegator's. What crosses the boundary is
the delegation's debrief, which the delegator receives explicitly.

Storage layout, one directory per node under <Base>/nodes/:
    nodes/1:AE/memory.json + memory-wal.jsonl
    nodes/1.1:AE-worker-1/memory.json + memory-wal.jsonl
"""

import re
import threading

try:
    from Memory import Memory
except ImportError:  # package-relative import when Scripts/ is a package
    from .Memory import Memory

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

# ':' is illegal in Windows directory names, so on-disk names use '__' where a
# label uses ':' (1.1:worker -> 1.1__worker). Labels stay colon-joined for
# display and for grepping the prompt.
_DIRSEP = "__"


def SafeNodeName(Name):
    """Agent names land in directory names, so strip anything path-ish.
    Dot runs collapse and leading dots go, so '../../etc' cannot escape."""
    Cleaned = _UNSAFE.sub("-", str(Name or "agent"))
    Cleaned = re.sub(r"\.{2,}", "-", Cleaned).strip("-.")
    return Cleaned or "agent"


def NodeId(ParentId, Index):
    """'1' + 1 -> '1.1';  '1.1' + 3 -> '1.1.3'. Depth is unbounded."""
    Base = str(ParentId or "").strip(".")
    return f"{Base}.{Index}" if Base else str(Index)


def NodeLabel(NodeId_, Name):
    """'1.1' + 'worker' -> '1.1:worker'."""
    return f"{NodeId_}:{SafeNodeName(Name)}"


def ParseLabel(Label):
    """'1.1:worker' -> ('1.1', 'worker'). Tolerates a bare id."""
    Text = str(Label or "").strip()
    for Sep in (":", _DIRSEP):
        if Sep in Text:
            Id, _, Name = Text.partition(Sep)
            return Id.strip(), SafeNodeName(Name)
    return Text, ""


def DirName(Label):
    """Label -> filesystem-safe directory name ('1.1:worker' -> '1.1__worker')."""
    Id, Name = ParseLabel(Label)
    return f"{Id}{_DIRSEP}{Name}" if Name else Id


def Depth(NodeId_):
    return len([P for P in str(NodeId_ or "").split(".") if P])


class MemoryTree:
    """Allocates node ids and builds Memory instances, one directory per node.

    Node ids are per-parent sequential (1.1, 1.2, ...) and are persisted in
    index.json so a restart continues numbering instead of recycling ids --
    recycled ids would silently inherit a dead worker's transcript.
    """

    def __init__(self, BasePath=".", RootLabel="AE"):
        from pathlib import Path
        self.BasePath = Path(BasePath)
        self.NodesPath = self.BasePath / "nodes"
        self.NodesPath.mkdir(parents=True, exist_ok=True)
        self.RootLabel = RootLabel
        self._lock = threading.RLock()
        self._counters = {}   # parent id -> last used child index
        self._nodes = {}      # label -> Memory
        self._Load()

    # ------------------------------------------------------------- persistence
    @property
    def IndexFile(self):
        return self.NodesPath / "index.json"

    def _Load(self):
        import json
        try:
            Data = json.loads(self.IndexFile.read_text(encoding="utf-8"))
            if isinstance(Data, dict):
                self._counters = {str(K): int(V) for K, V in (Data.get("counters") or {}).items()}
        except Exception:
            self._counters = {}

    def _Save(self):
        import json
        Temporary = self.IndexFile.with_suffix(".tmp")
        try:
            Temporary.write_text(json.dumps({"counters": self._counters}, indent=2), encoding="utf-8")
            Temporary.replace(self.IndexFile)
        except OSError:
            pass

    # ------------------------------------------------------------------- nodes
    def NodeDir(self, Label):
        return self.NodesPath / DirName(Label)

    def CreateRoot(self, Name, NodeNumber=1, **Kwargs):
        """Top-level node for an agent profile: label '1:AE'. Called once per
        profile so the tree has a stable addressable root."""
        Id = str(NodeNumber)
        Label = NodeLabel(Id, Name)
        if Label not in self._nodes:
            self._nodes[Label] = Memory(BasePath=str(self.NodeDir(Label)), **Kwargs)
        return Label, self._nodes[Label]

    def NextChildId(self, ParentId):
        with self._lock:
            Key = str(ParentId or "")
            Index = self._counters.get(Key, 0) + 1
            self._counters[Key] = Index
            self._Save()
            return NodeId(Key, Index)

    def Create(self, ParentId, Name, **Kwargs):
        """New child node with its own Memory. Returns (label, Memory)."""
        ChildId = self.NextChildId(ParentId)
        Label = NodeLabel(ChildId, Name)
        MemorySystem = Memory(BasePath=str(self.NodeDir(Label)), **Kwargs)
        self._nodes[Label] = MemorySystem
        return Label, MemorySystem

    def Adopt(self, Label, **Kwargs):
        """Existing node's Memory (created on first use)."""
        Label = str(Label)
        if Label not in self._nodes:
            self._nodes[Label] = Memory(BasePath=str(self.NodeDir(Label)), **Kwargs)
        return self._nodes[Label]

    def Get(self, Label):
        return self._nodes.get(str(Label))

    def Labels(self):
        """Every node directory present, whether or not it is in memory.
        Directory names come back as colon labels so callers only deal in one
        format."""
        Found = set(self._nodes)
        try:
            for Path in self.NodesPath.iterdir():
                if Path.is_dir():
                    Found.add(Path.name.replace(_DIRSEP, ":", 1))
        except OSError:
            pass
        return sorted(Found, key=lambda L: (Depth(ParseLabel(L)[0]), L))

    def Tree(self):
        """Nested view for rendering: [{'label','id','name','depth','children'}]."""
        Nodes = {}
        for Label in self.Labels():
            Id, Name = ParseLabel(Label)
            Nodes[Id] = {"label": Label, "id": Id, "name": Name, "children": [],
                         "memory": self.Get(Label) is not None}
        for Id in sorted(Nodes, key=lambda I: [int(P) if P.isdigit() else 0 for P in I.split(".")]):
            Node = Nodes[Id]
            Parent = ".".join(Id.split(".")[:-1])
            if Parent and Parent in Nodes:
                Nodes[Parent]["children"].append(Node)
            else:
                Nodes[Id]["_root"] = True
        return [N for N in Nodes.values() if N.get("_root")]

    def Descendants(self, Label):
        """Node labels under a parent, breadth-first."""
        Id = ParseLabel(Label)[0]
        Prefix = f"{Id}."
        return [L for L in self.Labels() if ParseLabel(L)[0].startswith(Prefix)]

    def SaveAll(self):
        with self._lock:
            for MemorySystem in list(self._nodes.values()):
                try:
                    MemorySystem.Save()
                except Exception:
                    continue

    def ClearSubtree(self, Label):
        """Forget every node below a parent (keeps the parent's own store).
        Used by `.clear` so wiped state does not resurface from a worker."""
        Id = ParseLabel(Label)[0]
        Prefix = f"{Id}."
        Removed = 0
        for Child in self.Descendants(Label):
            ChildId = ParseLabel(Child)[0]
            if not ChildId.startswith(Prefix):
                continue
            self._nodes.pop(Child, None)
            Directory = self.NodeDir(Child)
            if Directory.is_dir():
                import shutil
                shutil.rmtree(Directory, ignore_errors=True)
                Removed += 1
        with self._lock:
            for Key in [K for K in self._counters if K == Id or K.startswith(Prefix)]:
                self._counters.pop(Key, None)
            self._Save()
        return Removed