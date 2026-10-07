# Automatable Executables

AE uses shared formats so agents can discover and compose capabilities:

- `Skills/<skill-name>/SKILL.md` uses the Agent Skills format: YAML frontmatter with `name` and `description`, followed by Markdown instructions. Resources can live beside the file.
- `Tools/*.py` can register local Python tools by defining `register_tools(registry)`. Treat plugins as trusted code; module import runs Python.
- `Executables/*.ae` are parameterized AEX Python automations. The agent can inspect them, but every run asks for approval.
- `Events/*.ae` are declarative triggers that send an input prompt to an agent. Events do not execute Python.

All `.ae` files use YAML frontmatter bounded by `---`. Executable scripts use `type: ae`; events use `type: event`. Tool scripts may use `type: tool` and are exposed as OpenAI-compatible function tools.

The event dispatcher checks message, interval, and file-change triggers at chat-turn boundaries. `subagent_complete` triggers can be sent through `ToolRegistry.DispatchEvent` when subagent execution is connected.

Script code runs as Python in the AE process after approval. Approval is not an operating-system sandbox; only use scripts from trusted sources.