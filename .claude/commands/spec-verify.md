---
description: Iteratively verify implementation against specifications (spec.md, tasks.md, plan.md, design.md) using verification_agent, and then trigger code-review.
---

## User Input

```text
$ARGUMENTS
```

## Outline

You are instructed to execute the `spec-verify` skill immediately after an implement phase to enforce complete adherence to project plans and requirements.

1. **Context Gathering**: 
   - Adopt the `verification_agent` instructions from `.gemini/agents/verification_agent.md` (or `.claude/agents/verification_agent.md`).
   - Load all applicable specification artifacts: `spec.md`, `tasks.md`, `plan.md`, and `design.md`.

2. **Iterative Verification (The Loop)**:
   - Check the implementation against all loaded documents.
   - Ensure that every individual task mapped in `tasks.md` marked as completed is genuinely functional.
   - Run the relevant programmatic test suite to verify tests pass and no regressions exist.
   - **Self-Correction**: If you find any discrepancies, missing implementations, logic blind-spots, or failing tests, fix them yourself by writing the missing code or rectifying existing issues. Note: Do not just point out the problems; go ahead and fix them.
   - Iterate Verification -> Execution (Fix) -> Verification until the implementation perfectly and fully maps to the specs.

3. **Status Confirmation**: 
   - After a 100% compliance rate is reached, output a final verification success report.

4. **Handoff (Conditional)**:
   - Analyze the nature of the changes. 
   - **Frontend/UI/UX**: Stop execution. Prompt the user to verify the changes visually in their local environment. Instruct them to manually execute the `/code-review` command if satisfied.
   - **Backend/Logic**: Automatically invoke the `code-review` command or skill (e.g., `.claude/skills/code-review/SKILL.md`) to initiate the multi-agent code evaluation review process natively.
