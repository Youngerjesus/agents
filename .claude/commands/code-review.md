---
description: Runs the code-review skill, employing multiple specialized agents to evaluate correctness, testing, performance, and maintainability of the current code changes.
---

## User Input

```text
$ARGUMENTS
```

## Code Review Execution

Please invoke the code-review skill (located at `.claude/skills/code-review/SKILL.md`) to perform a comprehensive code review. 

1. Gather the context of the current changes (using `git diff`).
2. Run the `code-reviewer` agent.
3. Run the `security-reviewer`, `maintainability-reviewer`, and `performance-reviewer` agents.
4. Provide the unified output format as defined in the skill documentation.
