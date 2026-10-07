# Events

Store declarative AEX event definitions as `.ae` files. An event matches one of its triggers and supplies `input_prompt` as a new instruction for its target agent. Events contain no executable code.

```yaml
---
type: event
name: release-reminder
version: 1.0.0
description: Ask the agent to review release readiness when a release request appears.
target_agent: CodeBuddy
input_prompt: Review the current project for release readiness and report blockers.
trigger:
  on:
    - type: message
      match: "prepare a release"
---
```

Supported trigger types:

- `message`: optional `match` substring in the user's message.
- `interval`: `every` duration such as `30s`, `5m`, `2h`, or `1d`; checked when a chat turn arrives.
- `file_change`: workspace-relative `path`; checked when a chat turn arrives.
- `subagent_complete`: dispatched with completion data through `ToolRegistry.DispatchEvent`.

Set `target_agent` to an agent name or `"*"` for all agents. Message, interval, and file-change triggers are checked at turn boundaries, not by a background daemon. Event activation is recorded separately in shared memory.