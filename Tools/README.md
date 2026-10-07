# Core Tools

Core tools are editable `.ae` definitions under `Tools/core/`. Each uses the standard YAML frontmatter and an OpenAI function schema derived from `parameters`. Set `core: true` and `effect: read` only for trusted, non-mutating core tools; the registry will expose those without per-call approval and may batch adjacent read-only calls.

Core tool scripts compose the primitive registry tools with `call_tool("tool_name", argument=value)`. Underlying tools retain their own approvals and boundaries. Core definitions are loaded from disk for each agent request, so edits become available on the next request. Custom AEX or Python plugins are not treated as core by default.
