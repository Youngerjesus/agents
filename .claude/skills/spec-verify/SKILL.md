---
name: spec-verify
description: A specialized skill to iteratively verify that the implementation adheres to all specifications (spec.md, tasks.md, plan.md, design.md) using the verification_agent, and then trigger code review.
license: Complete terms in LICENSE.txt
---

# Spec Verification Skill

This skill ensures that the newly implemented code strictly and comprehensively adheres to all defined product and technical specifications before moving into the code review phase. It leverages the specialized `verification_agent` in an iterative loop.

## Workflow 

When requested to verify the implementation after a `speckit.implement` phase, follow this sequence of actions:

1. **Context & Persona Loading**: 
   - Load the `verification_agent` persona and guidelines from `.gemini/agents/verification_agent.md` (or `.claude/agents/verification_agent.md`).
   - Read the core specifications: `spec.md` (Product rules / logic), `tasks.md` (Implementation list), `plan.md` (Technical approach), and `design.md` (System architecture). 

2. **Iterative Verification (Execution)**:
   - Identify discrepancies between the specifications and the actual source code (`src/` or related directories).
   - Assert that every task marked as `[x]` or `[X]` in `tasks.md` is genuinely functional in the codebase.
   - Run the testing suite defined in the workspace (using relevant bash commands based on `plan.md` or `package.json`).
   - Specifically check if UI/UX flow, data contracts, and required test coverages match the implementation.
   - Perform a regression check and behavioral properties check as documented in the `verification_agent`.

3. **Feedbacks and Fixes**:
   - If there are unfulfilled specifications, incomplete tasks, or failing tests, output a detailed report outlining the gaps.
   - **Fix the gaps**: Implement the missing features, address the test failures, and fix logic bugs.
   - **REPEAT the Iterative Verification (Step 2)** until all issues are resolved and 100% compliance is reached. Do not proceed until everything perfectly aligns with the spec.

4. **Verification Handoff Analysis & Execution**:
   - Once the technical verification is successfully completed (0 gaps, all tests passing, strict spec compliance), analyze the nature of the implemented features.
   - **Frontend / UI / UX / Human-Centric Tasks**: If the implementation involves visual changes, CSS, interactive frontend components, or anything that requires human visual confirmation:
     - Output a final verification success report.
     - **STOP execution and prompt the user**: *"Technical spec verification is complete. Since this update includes frontend/UI changes, please verify the functionality in your local environment. If it looks good, please execute the `/code-review` command to proceed."*
   - **Backend / Logic / API / Purely Automated Tasks**: If the changes are purely backend, algorithms, APIs, or easily testable background logic with no UI changes:
     - Output a final verification success report.
     - **Automatically trigger the `code-review` skill** (e.g., executing the `/code-review` command or invoking `.claude/skills/code-review/SKILL.md`) to evaluate correctness, performance, maintainability, and security.

## Requirements

Ensure that the AI model acts repeatedly, reflecting and fixing issues, until the technical verification milestone is genuinely achieved, and carefully judges whether to pause for human confirmation or auto-proceed.
