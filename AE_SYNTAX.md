## Messaging vs Delegation

- `self_prompt(content, memory=True|False)` — own history. In `.ae`: `self_prompt("...", memory=False)`.
- `send_message(target_agent, content)` — cross-agent inbox. In `.ae`: `send_message("CodeBuddy", "...")`.
- `ask_user(question)` — ask one question and wait. Batch up to 6 in ONE call: `ask_user(questions=[...])` where each item is a string or `{question, options[2-8], default, required}`; options render as a numbered pick list. Returns `{answers: [{question, answer}], answer: <first>}`.
- `remind(text, delay_seconds, target_agent?)` — one-shot inbox reminder to yourself after a delay (within-session, not cron).
- `agent_note add/list/clear` — standing self-notes injected into your system prompt (not memory). Record loop lessons here.
- `turn_stats` — live per-turn tool counts. `update_being(text)` — rewrite your own system prompt (approval-gated). `clock` — current date/time ground truth (use BEFORE any time question; never guess the date).
- `stage_gate(milestone, demo)` — phase-boundary user decision (continue/redirect/stop). `delegate_task(..., stance)` — neutral/adversarial(reviewer) workers that debrief to you. `handoff save/load` — continuity packets across sessions.
- `todo_list plan` — one-call nested trees ([{title, subtasks}]), phases roll up, `cancelled` never blocks. `executions` / `stop_execution` — watch and cooperatively stop live .ae loops (which relay via report() + poll should_stop(); e.g. stopwatch).
- `task_complete(summary)` — signal that all work is complete. Call when all todos are done.
- `delegate_task(goal, context, target_agent, provider, model, max_tokens, toolsets, timeout_seconds, background=False)` — spawns TempAgent (volatile memory, restricted tool copy, no nesting, auto-cleanup). Returns `Subagent result:\n...` (sync) or a `dlg_` id (background).
  - Sync (default): blocks; dispatch + completion panels in the terminal.
  - Background (`background=true`): returns immediately; `check_subagent(id)` collects, `steer_subagent(id, text)` redirects mid-flight, `stop_subagent(id)` ends one without disturbing siblings; `delegations` command lists all; completions auto-print.
  - Background workers can't prompt: reads + safe tools (search, memory notes, messaging) run; writes/terminal/delegation are skipped with a note the worker sees and routes around.
  - toolsets: web|file|terminal|memory|git|skill|event|executable (delegator chooses)
  - self-delegate: omit provider/model, small max_tokens (50-100)
- `revive_subagent(id, extra_context)` — re-runs a finished (done/error/killed) delegation with its original goal/model/toolsets plus extra guidance. Returns a new `dlg_` id. Only for terminal delegations; running ones must be steered or stopped first.
- `steer_subagent(id, text)` — redirects a running background delegation mid-flight. Text lands in the worker's next round. Fails if already finished.
- `check_subagent(id)` — checks status/elapsed/result of a background delegation.
- `stop_subagent(id)` — stops a running background delegation. Siblings keep running.
- `delegations` — CLI command lists all background delegations with status/elapsed/model.
- `undo` (tool + CLI) — restores workspace files changed by recent write_file/patch_file (undo N steps back).

## Sessions & automation
- `session_usage` — live call counts, per-tool tops, token/USD budget state. Check before heavy fan-outs.
- `fetch_many(urls[<=10])` — parallel page fetch, one call. Never N serial web_fetch calls.
- `wait_for_file(path, timeout_seconds, poll_seconds)` — block until a workspace file appears/changes. No polling loops.
- `checkpoint save/restore/list/delete` — named todos + long-term-memory snapshots; save before risky refactors.
- Interval/file_change events fire on a background scheduler between turns (message events still fire on chat only); completions surface automatically.
- `scheduler [on|off]` toggles the background tick; per-event opt-out via `scheduler: false` frontmatter.
- `usage` command shows per-agent token totals for the session.
- `allow <tool>` / `allowed` / `unallow <tool>` manage the session allow-list (always-allow without prompts).
- Chat failover: a missing provider key falls over to the next keyed provider automatically (`providers` shows key status).

## What lives where (scalability contract)
- Python (`Scripts/`) holds ONLY syscalls: workspace file IO, terminal, HTTP, memory store (+locks), inbox files, event polling/dispatch, approval gate, LLM dispatch, delegation threads. These cannot be `.ae` because they ARE the trust + concurrency boundary.
- EVERYTHING composable ships as editable `.ae`: all of `create_*`, `read/update/delete_entity`, `list_all_entities`, `update_syntax_guide`, the `board` renderer (`Executables/board.ae`), and every core-tool wrapper in `Tools/core/` (thin `call_builtin` shims you can rewrite).
- Prompt text in `Main.py` (capability catalog, protocols) is configuration, not logic: it only names tools the registry reports. Add a tool and it appears automatically.
- Rule of thumb: if it touches disk/network/processes/locks/threads, it's a syscall. If it decides, composes, formats, or automates, it's `.ae`.

## Continuous Execution
The agent loops while real work is happening, and answers chat directly:
- Plain chat (hi, questions): answer with no tools, no todos.
- Real tasks: use tools, track with `todo_list` (add, update, complete).
- Loop ends when todos are all `done`, `task_complete(summary)` is called, or the model stops calling tools.
- Use `ask_user` for clarification, `self_prompt` for internal reasoning.

## Reasoning modes (per-agent `reasoning_mode`)

- `explicit`: prompt appends `## Reasoning — think step by step` before acting.
- `hidden`: engine does a private planning LLM call, injects plan as hidden context (not shown as final answer).
- `none`: no extra reasoning.

## Validation checklist (before saving)

1. YAML parses, `name`/`type`/`description` valid.
2. `execute` compiles (`python -m py_compile` mental check).
3. `params.get(...)` for optional args; required args validated.
4. Return JSON-serializable via `return_value`.
5. No `delegate_task` inside TempAgent-eligible code paths (blocked at runtime).
6. Paths stay inside workspace (`..` rejected).