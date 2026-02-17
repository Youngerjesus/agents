# CLAUDE.md

## Critical Rules

### Code Organization
- **Directory Structure**:
  - `src/`: Production code.
  - `tests/`: Test code (mirrors `src` structure).
  - `scripts/`: CLI/Maintenance scripts.
  - `docs/`: Architecture, ADRs, Specs.
- **Layered Architecture**: `domain` ← `application` ← `interface`.
  - **Dependency Rule**: Outer layers depend on inner layers. Inner layers (Domain) are pure.
  - **src/<module>/domain/**: Pure business logic, entities, exceptions. No external dependencies.
  - **src/<module>/infrastructure/**: Adapters for DB, LLM, External APIs. Implements domain interfaces.
  - **src/<module>/application/**: Use cases, pipelines, orchestration.
  - **src/<module>/interface/**: Entry points (API, CLI, Consumers). Handles DTOs and ACL.

### Code Style
- **Sizing**: Files ≤ 300 lines. Functions ≤ 20 lines.
- **Design**: SRP (Single Responsibility), OCP (Open-Closed).
- **Naming**: Full descriptive names. No abbreviations.
- **Complexity**: Cyclomatic complexity ≤ 10. Nesting ≤ 3 levels.
- **Exposure**: Use `__init__.py` to define public API boundaries.

### Testing
- **TDD Mandatory**: Write tests before implementation.
- **Pytest Markers**: `unit`, `integration`, `api`, `slow`.
- **Workflow**: Run `pytest -m "not slow and not api" -q` before & after every change.
- **Unit Tests (80%)**: Test behavior/contracts, not methods. Mock external IO.
- **Integration Tests (15%)**: Test core use cases. Use real managed dependencies (e.g. Dockerized DB).
- **Format**: `GIVEN-WHEN-THEN` structure.

### Security
- **Secrets**: Never commit secrets. Use `src/settings.py` loading from `.env`.
- **Logging**: Mask PII. No secrets in logs.
- **LLM Safety**: Sanitize inputs preventing prompt injection.

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
    - `checklists`: checklists for each spec. for verification.
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
