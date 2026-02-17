# CLAUDE.md

## Critical Rules

### SDD (Spec Driven Development) Protocol
- **No Spec, No Code**: Do not write any code without a finalized Spec. The Spec (`spec.md`, `plan.md`, `tasks.md`) is the Single Source of Truth (SSOT).
- **Control over Speed**: Focus on *What* (Properties/Requirements) and *Why*, not just *How* (Implementation details).
- **EARS Syntax**: Requirements must follow `When -> If -> Shall` structure to remove ambiguity.
- **Logical Sparring**: All specs must undergo rigorous critique to identify edge cases before implementation.
- **Property-Based Verification**: Verification is strictly based on whether the code satisfies the properties defined in the Spec.

### Code Organization & Artifacts
- **Directory Structure**:
  - `specs/`: Numbered folders for specs (e.g., `specs/001-auth/`) containing:
    - `spec.md`: Business requirements & Properties (EARS).
    - `plan.md`: Architecture, data separation, pseudo-code.
    - `tasks.md`: Granular milestones for agents.
  - `.claude/tasks/`: Persistent task tracking (`tasks.json`).
  - `src/`: Implementation logic.
  - `tests/`: TDD and Property-based tests.
- **Context Injection**: Use AST Compiler/File concatenation for full context before Spec generation.

### Development Workflow (4 Stages)

#### Stage 1. Requirement Injection (The "What")
- **Trigger**: User provides raw requirements.
- **Process**: Define the core "What" and "Why" without implementation details.

#### Stage 2. Spec Generation & Sparring (The "Plan")
- **Spec Creation**: Generate `spec.md` (Properties), `plan.md` (Design), `tasks.md` (Plan).
- **Logical Sparring**: Critically review specs for edge cases, logic holes, and side effects.
- **Output**: Finalized Spec documents.

#### Stage 3. TDD Implementation (The "How")
- **Process**:
  1.  **Context Loading**: Load relevant project state (AST/Files).
  2.  **TDD Cycle**:
      - Create Skeleton.
      - Write Tests (based on Spec Properties).
      - Implement Logic.
      - Refactor.
  3.  **Task Updates**: Update `docs/specs/[spec-name]/tasks.md` continuously.

#### Stage 4. Verification (The "Check")
- **Independent Review**: Verify implementation against `spec.md` properties using fresh context and verification agents(./claude/agents/verification_agent.md).
- **Regression**: Ensure no side effects using property-based testing.

## Toolkit & Commands

### Testing Standard
- **TDD Mandatory**: Tests must exist before implementation logic.
- **Property-Based Testing (PBT)**: Prefer verifying invariants over simple example-based tests.
- **Verification**: Code is only "Done" when it passes strict verification against the Spec.
