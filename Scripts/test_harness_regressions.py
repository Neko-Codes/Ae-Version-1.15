import tempfile
import unittest
from pathlib import Path

from .Main import (
    _AdvanceTruncatedOutput,
    _IsArtifactFollowup,
    _LooksLikeNonAnswer,
    _NeedsConcreteAnswer,
    _ReportAnswerMeetsRequirements,
    _SelectArtifactFollowupPaths,
    _TodoAddBlocked,
)
from .Memory import Memory
from .ToolRegistery import ToolRegistry


class HarnessRegressionTests(unittest.TestCase):
    def setUp(self):
        self.TempDir = tempfile.TemporaryDirectory()
        self.Memory = Memory(BasePath=self.TempDir.name)

    def tearDown(self):
        self.TempDir.cleanup()

    def test_chat_and_artifact_memory_persist_without_failed_writes(self):
        self.Memory.AddMessage("user", "teach me variables")
        self.Memory.AddToolEvent("write_file", {"path": "lesson.md"}, "written")
        self.Memory.AddToolEvent("write_file", {"path": "missing.md"}, "denied", IsError=True)

        Reloaded = Memory(BasePath=self.TempDir.name)

        self.assertEqual(Reloaded.ShortTerm[-1]["content"], "teach me variables")
        self.assertEqual(Reloaded.GetRecentArtifacts(), ["lesson.md"])

    def test_temporary_topic_lists_are_not_long_term_memory(self):
        Registry = ToolRegistry.__new__(ToolRegistry)
        Registry.Memory = self.Memory
        Registry.CurrentUserRequest = "give me a Roblox Luau learning path"
        Registry.TurnMemoryRetainBlocked = False

        Result = Registry._RetainMemory("Beginner topics: variables, functions, tables", "user_fact")
        self.assertFalse(Result["stored"])
        self.assertEqual(self.Memory.LongTerm, [])
        self.assertFalse(Registry._RetainMemory("Beginner topics: variables", "lesson")["stored"])

        Registry.CurrentUserRequest = "Remember that I prefer short examples"
        Registry.TurnMemoryRetainBlocked = False
        self.assertTrue(Registry._RetainMemory("I prefer short examples", "preference")["stored"])

    def test_invalid_todo_plan_is_atomic(self):
        with self.assertRaises(ValueError):
            self.Memory.ManageTodos("plan", Plan=[
                {"title": "valid"},
                {"title": "invalid", "status": "unknown"},
            ])
        self.assertEqual(self.Memory.Todos, [])

    def test_completion_rejects_open_current_turn_todos(self):
        Registry = ToolRegistry.__new__(ToolRegistry)
        Registry.Memory = type("MemoryStub", (), {"Todos": [{"id": "new", "title": "Write lesson", "status": "todo"}]})()
        Registry.TurnTodoIdsAtStart = set()

        with self.assertRaisesRegex(ValueError, "current-turn todos remain open"):
            Registry._TaskComplete("Lesson complete")

    def test_claim_only_learning_path_is_non_answer(self):
        Request = "Give me a short Roblox Luau learning path from beginner to advanced."
        Reply = "The Roblox Luau learning path has been successfully created and marked as done."
        self.assertTrue(_NeedsConcreteAnswer(Request))
        self.assertTrue(_LooksLikeNonAnswer(Reply, Request))
        self.assertTrue(_LooksLikeNonAnswer("I am ready to assist you. Please provide me with the details of the task.", Request))
        self.assertFalse(_LooksLikeNonAnswer("1. Variables\n2. Modules\n3. Networking", Request))

        Registry = ToolRegistry.__new__(ToolRegistry)
        Registry.Memory = None
        Registry.TurnTodoIdsAtStart = None
        Registry.CurrentUserRequest = Request
        with self.assertRaisesRegex(ValueError, "actual outline"):
            Registry._TaskComplete(Reply)

    def test_task_complete_rejects_explanation_claim_without_example(self):
        Registry = ToolRegistry.__new__(ToolRegistry)
        Registry.Memory = None
        Registry.TurnTodoIdsAtStart = None
        Registry.CurrentUserRequest = "Explain a Luau local variable and show one example with its value."
        with self.assertRaisesRegex(ValueError, "actual example/output"):
            Registry._TaskComplete("Provided a concise explanation and an example of a Luau local variable.")
        self.assertEqual(Registry._TaskComplete("A local variable stores a value; `local score = 10` prints 10.")["status"], "complete")

    def test_plural_artifact_followup_selects_all_recent_files(self):
        Paths = ["Part2.md", "Part1.md"]
        Request = "Can you list those off and add the missing intermediate section?"
        self.assertTrue(_IsArtifactFollowup(Request, Paths))
        self.assertEqual(_SelectArtifactFollowupPaths(Request, Paths), Paths)
        self.assertEqual(_SelectArtifactFollowupPaths("Summarize Part1.md", Paths), ["Part1.md"])

    def test_research_recall_is_bounded_and_excludes_failures(self):
        self.Memory.AddToolEvent("web_search", {"query": "Luau"}, "Official docs found")
        self.Memory.AddToolEvent("web_fetch", {"url": "https://create.roblox.com/docs/luau"}, "Official Luau documentation")
        self.Memory.AddToolEvent("web_fetch", {"url": "https://broken.example"}, "blocked", IsError=True)

        Context = self.Memory.GetRecentResearch(MaxChars=200)

        self.assertIn("Official Luau documentation", Context)
        self.assertNotIn("broken.example", Context)
        self.assertLessEqual(len(Context), 200)

    def test_report_gate_and_continuation_budget(self):
        SelfContained = "Generated a detailed report based on sources."
        self.assertFalse(_ReportAnswerMeetsRequirements(SelfContained, True))
        self.assertTrue(_TodoAddBlocked("add", 2))
        self.assertFalse(_TodoAddBlocked("plan", 99))

        Parts = []
        Continue, _, Count = _AdvanceTruncatedOutput("part one", "length", Parts, 0)
        self.assertTrue(Continue)
        Continue, Final, _ = _AdvanceTruncatedOutput("part two", "stop", Parts, Count)
        self.assertFalse(Continue)
        self.assertEqual(Final, "part one\n\npart two")


if __name__ == "__main__":
    unittest.main()