---
name: sdd-autopilot
description: Fully automates the Spec-Driven Development (SDD) pipeline. Use this skill when the user wants to automatically advance a feature specification from its initial draft (`spec.md`) through clarification, auditing, planning, designing, task generation, consistency analysis, and final readiness checks, yielding a perfectly prepared implementation phase with zero manual intervention.
---

# SDD Autopilot Workflow

This skill automates the entire Spec-Driven Development (SDD) lifecycle. The user has already run `/speckit.specify` (or manually created a `spec.md`). Your goal is to act as an elite Staff Software Engineer / Product Manager and drive the specification through every rigorous validation and design step until it is perfectly ready for implementation.

## Core Principle: Autonomy & Best Practices
You must execute the following commands in sequence.
During interactive commands (`clarify`, `spec-audit`, `analyze`), **DO NOT ask the user for clarification or options**. Instead, you must autonomously select the best option or provide the best answer using industry standard best practices, focusing on:
- High performance, scalability, and security.
- Bulletproof edge-case handling.
- Clear, measurable acceptance criteria.
- Maintaining the user's original product vision while hardening the engineering strategy.

## Execution Sequence

Execute the following steps one by one. Read the output of each command and perform the necessary follow-up actions before moving to the next step.

### Step 1: Clarification Loop (`/speckit.clarify`)
1. Run the `/speckit.clarify` command.
2. Read the multiple-choice or short-answer questions presented by the command.
3. **Autonomously answer** the questions by selecting the most robust, scalable, and secure options. Provide short answers where requested.
4. Apply the answers to the spec as instructed by the `clarify` command output.
5. Repeat `/speckit.clarify` until the command reports "No critical ambiguities detected" or that all categories are Clear.

### Step 2: Checklist Generation (`/speckit.checklist`)
1. Run the `/speckit.checklist` command.
2. If it asks for focus areas, autonomously select the most critical ones (e.g., Security, UX, Edge Cases).
3. Ensure the unit tests for the requirements are generated.

### Step 3: First Strategic Audit Loop (`/spec-audit`)
1. Run the `/spec-audit` command.
2. Review the Markdown report detailing 🔴 Critical Flaws and 🟡 Strategic Refinements.
3. If flaws are found:
   - **Autonomously rewrite/update** `spec.md` to resolve all critical flaws and incorporate the strategic refinements.
   - Run `/spec-audit` again.
4. Repeat this loop until `/spec-audit` yields a clean bill of health (no critical flaws).

### Step 4: Planning (`/speckit.plan`)
1. Run the `/speckit.plan` command to generate the technical implementation plan (`plan.md`).
2. Verify `plan.md` is successfully generated.

### Step 5: Technical Design (`/spec-design`)
1. Run the `/spec-design` command to generate the detailed technical design document (`design.md` or similar).
2. Verify the design document is successfully generated.

### Step 6: Task Breakdown (`/speckit.tasks`)
1. Run the `/speckit.tasks` command to generate the parallelized execution tasks (`tasks.md`).
2. Verify `tasks.md` is successfully generated.

### Step 7: Consistency Analysis (`/speckit.analyze`)
1. Run the `/speckit.analyze` command to perform a cross-artifact scan.
2. Review the resulting Analysis Report.
3. If inconsistencies, missing coverage, or constitution violations are found:
   - **Autonomously edit** the offending artifacts (`spec.md`, `plan.md`, `design.md`, or `tasks.md`) to apply the necessary remediations.
   - Run `/speckit.analyze` again to confirm all artifacts are perfectly aligned.

### Step 8: Final Strategic Audit Loop (`/spec-audit`)
1. Run `/spec-audit` one final time to ensure the sweeping architectural decisions and task breakdowns haven't introduced new logical gaps.
2. Automatically fix any issues found in `spec.md`, `plan.md`, or `design.md`.

### Step 9: Readiness Gate (`/speckit.readiness`)
1. Run the `/speckit.readiness` command.
2. Review the Readiness Report.
3. If the verdict is NO-GO due to critical blockers (e.g., TBD items, missing environment configs):
   - **Autonomously resolve** the blockers in the respective files by filling in safe, standard defaults or documenting exact steps for the implementation phase.
   - Run `/speckit.readiness` again until the verdict is **GO**.

## Completion
Once `/speckit.readiness` returns a **GO** verdict, notify the user that the SDD Autopilot sequence is complete. Present a brief summary of the major architectural decisions made and confirm that the project is completely ready for `/speckit.implement`.
