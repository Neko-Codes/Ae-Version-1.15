"""Taxonomy: the shared classification scheme for every AE capability.

One vocabulary for all eight capability types so tools, skills, workflows,
events, knowledge and notes can be listed, filtered and sorted the same way.

Resolution order for a file's category (first hit wins, explicit beats inferred):
    1. frontmatter `category:`
    2. folder path -- Skills/<category>/<name>/skill.ae gives `category`
    3. "uncategorized"

Tags are normalized to a sorted, de-duplicated, lowercase set that always
includes the category, so filtering by tag is total (never a dead tag).
"""

import re

# Canonical categories per type. Knowledge and data are subject-driven and
# deliberately unconstrained; everything else is a closed vocabulary so the
# prompt can list real groups instead of an unbounded pile of names.
CATEGORIES = {
    "tool": (
        "core", "web", "file", "system", "git", "terminal", "memory", "task",
        "cognitive", "communication", "delegation", "entity", "knowledge",
        "media", "vision", "data", "scheduling", "session",
    ),
    "skill": (
        "autonomous-ai-agents", "blocked-page-recovery", "creative", "delegate",
        "devops", "direct-call", "email", "media", "note-taking", "pdf-generation",
        "playwright-pdf", "productivity", "python-file-write", "reasoning",
        "reminders", "research", "self", "self-delegate", "skill-workflow", "skills",
        "social-media", "software-development", "test", "tools", "web",
        "ae-platform", "ae-script-creation", "apple", "example-category",
    ),
    "workflow": (
        "planning", "research", "analysis", "creation", "review", "debugging",
        "operations", "communication", "learning", "self-improvement",
    ),
    "knowledge": (),
    "note": ("lesson", "correction", "harness", "user", "observation"),
    "event": ("message", "interval", "file_change", "subagent_complete"),
    "ae": (),
    "data": (),
}

UNCATEGORIZED = "uncategorized"

# Folder name -> capability type, used for discovery and for deriving a
# capability's category from where it lives on disk.
TYPE_FOLDERS = {
    "Tools": "tool",
    "Skills": "skill",
    "Workflows": "workflow",
    "Executables": "ae",
    "Events": "event",
    "Knowledge": "knowledge",
    "Notes": "note",
}

# Folders that name a capability root, not a category. `core` is deliberately
# NOT here: AutomatableExecutables/Tools/core/ really does mean category=core.
STRUCTURAL = {"skill", "Active", "Inactive", "Agents"}


def _Slug(Value):
    """Lowercase kebab-case, safe for display and for comparison."""
    Text = str(Value or "").strip().lower()
    Text = re.sub(r"[\s_]+", "-", Text)
    Text = re.sub(r"[^a-z0-9.-]+", "", Text)
    Text = re.sub(r"-{2,}", "-", Text).strip("-.")
    return Text


def Normalize(Category, Type=""):
    """Canonical form of a category name. Unknown categories pass through
    (lowercased) rather than being rejected -- an agent may legitimately
    invent a subject area for knowledge."""
    Slug = _Slug(Category) or UNCATEGORIZED
    Known = CATEGORIES.get(Type)
    if Known and Slug not in Known and Slug != UNCATEGORIZED:
        return Slug
    return Slug


def NormalizeTags(Raw, Category=""):
    """Sorted unique lowercase tags, always including the category."""
    Out = set()
    if isinstance(Raw, str):
        Raw = re.split(r"[,\s]+", Raw)
    if isinstance(Raw, (list, tuple, set)):
        for Item in Raw:
            if isinstance(Item, str):
                Tag = _Slug(Item)
                if Tag:
                    Out.add(Tag)
    Cat = _Slug(Category)
    if Cat and Cat != UNCATEGORIZED:
        Out.add(Cat)
    return sorted(Out)


ROOT_FOLDERS = ("Skills", "Tools", "Workflows", "Executables", "Events",
                "Knowledge", "Notes", "AutomatableExecutables")


def FromPath(PathOrRelative, Type=""):
    """Derive (category, subcategory) from a capability's path.

    Accepts either a path relative to a capability root
    (Skills/research/deep-dive/skill.ae) or an absolute one -- everything up to
    the last capability root is discarded, so Agents/Ae/AutomatableExecutables/
    Skills/tools/x.ae infers `tools` just like the global copy does.
    """
    Parts = [P for P in str(PathOrRelative or "").replace("\\", "/").split("/") if P and P != "."]
    if not Parts:
        return UNCATEGORIZED, ""

    # Cut to the tail that matters: after the last capability root, if any.
    Cut = 0
    for Index, Part in enumerate(Parts):
        if Part in ROOT_FOLDERS:
            Cut = Index + 1
    Parts = Parts[Cut:]
    # Drop structural folders (skill.ae dirs, Active/Inactive) case-insensitively.
    Parts = [P for P in Parts if P not in STRUCTURAL and P.lower() not in STRUCTURAL]
    if not Parts:
        return UNCATEGORIZED, ""

    Stem = Parts[-1]
    if Stem.lower().endswith(".ae"):
        Stem = Stem[:-3]
    Depth = len(Parts)

    if Type == "skill":
        # Skills/<category>/<name>/skill.ae  ->  category + subcategory
        if Depth >= 3:
            return Normalize(Parts[-3], Type), _Slug(Parts[-2])
        # Skills/<category>/skill.ae -- the file is the capability, no subcategory
        if Depth == 2:
            return Normalize(Parts[-2], Type), ""
        return UNCATEGORIZED, ""

    if Depth >= 2:
        # Tools/core/x.ae -> core ; Workflows/planning/y.ae -> planning
        # Knowledge/Active/k.ae -> Active is structural, so uncategorized.
        Parent = Parts[-2]
        if Parent in STRUCTURAL or Parent.lower() in STRUCTURAL:
            return UNCATEGORIZED, ""
        # Tools/core/web_search.ae -> category only. The file IS the capability,
        # so its stem is a name, not a subcategory (that only exists for skills).
        return Normalize(Parent, Type), ""
    return UNCATEGORIZED, ""


def SortKey(Type, Category, Name):
    """Deterministic ordering: category first, then name. Stable across runs
    so prompt sections never reshuffle between turns (prefix-cache friendly)."""
    return (str(Type), str(Category or UNCATEGORIZED), str(Name or "").lower())


def PromptGroups(Entries):
    """Group listing entries by category, sorted. Entries need
    `category` and `name` keys. Returns [(category, [entries]), ...]."""
    Buckets = {}
    for Entry in Entries:
        Buckets.setdefault(Entry.get("category") or UNCATEGORIZED, []).append(Entry)
    return [
        (Category, sorted(Bucket, key=lambda E: str(E.get("name", "")).lower()))
        for Category, Bucket in sorted(Buckets.items())
    ]