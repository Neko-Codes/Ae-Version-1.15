# Executables

Store parameterized AEX automations as `.ae` files. The agent can list and read their full definitions. Calling `run_executable` always asks the user before executing Python.

```yaml
---
type: ae
name: summarize_text
version: 1.0.0
description: Summarize supplied text into a short result.
parameters:
  text:
    type: string
    required: true
    description: Text to summarize.
execute: |
  words = params["text"].split()
  return_value = {"summary": " ".join(words[:40])}
---
```

Supported parameter types: `string`, `number`, `integer`, `boolean`, `array`, `object`, and `json`. Put the result in `return_value`; it is returned to the calling agent as JSON.

An executable can call registered tools with `call_tool("tool_name", argument=value)`, load a skill with `call_skill("skill-name")`, or emit a named event with `emit("event-name", data)`. Each called tool still enforces its own approval policy.