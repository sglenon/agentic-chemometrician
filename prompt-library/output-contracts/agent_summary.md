# Output Contract: Agent Summary

Every agent summary presented to the user must conform to this structure.

## Required fields

| Field | Description |
|---|---|
| `step` | The workflow step being summarised (e.g. "inspect_dataset", "validate_results") |
| `status` | `ok`, `deferred`, `failed`, or `pending_approval` |
| `tool_called` | The MCP tool name invoked |
| `run_id` | The run ID if an artifact was produced; `null` if not applicable |
| `key_findings` | Bullet list of 1–5 factual findings from tool output only |
| `warnings` | List of active warnings with severity; empty list if none |
| `next_step` | The next recommended action |
| `approval_required` | `true` if a human-in-the-loop gate is triggered; `false` otherwise |

## Format

```
## [Step Name]

**Status**: [ok | deferred | failed | pending_approval]
**Tool**: [tool_name]
**Run ID**: [run_id or N/A]

### Findings
- [Finding 1 from tool output]
- [Finding 2 from tool output]

### Warnings
- [WARNING code: message] (severity)
- (none)

### Next step
[One sentence describing what happens next]

### Approval required
[Yes — reason | No]
```

## Rules

- `key_findings` must reference tool output directly. Do not add editorial conclusions.
- `warnings` must include every warning returned by the tool; none may be suppressed.
- `approval_required` must be `true` at all required human-in-the-loop gates.
- Do not write `status: ok` if the tool returned `ok=False`.
