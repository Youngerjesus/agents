# Rule: Documentation Creation Standards

This document outlines the strict rules for creating documentation within the project, specifically for Context Detail Documents and Algorithm Specifications. These rules are designed to ensure clarity, consistency, and correctness for AI-assisted development (Claude Code).

## 1. Context Detail Document Structure

All context detail documents MUST follow a structured Markdown format.

### Required Sections:

1.  **Overview**: Brief description of the module or feature.
2.  **Input/Output Specifications**:
    *   Explicitly define input data and their types.
    *   Explicitly define output data and their types.
3.  **Pseudo-code**:
    *   Define behavior using pseudo-code to capture logical flow without being tied to specific syntax.
4.  **Constraints**:
    *   **Tech Stack**: Explicitly state relevant technologies (e.g., Python 3.10+, Asyncio, Web3.py).
    *   **Core Goals**: What must be achieved?
    *   **Philosophy**: Design principles (e.g., "Fail fast," "Minimal state").
    *   **Requirements**: Functional and non-functional requirements.

### Diagramming (Mermaid):

Use Mermaid diagrams to visualize scenarios. Do not just diagram the structure; diagram the *behavior*.

*   **Scenarios Layout**:
    *   **Happy Path**: The ideal flow of execution.
    *   **Edge Cases**: Flows for boundary conditions.
    *   **Exception Strategies**: How errors are handled and where the flow is redirected.

## 2. Algorithm Design Specifications

When documenting algorithms, you MUST include a "Blueprint" section that includes mathematical and logical rigor.

### Mathematical Modeling & Invariants
*   **Mathematical Modeling**: Represent complex business logic with mathematical formulas where possible (e.g., `Slippage = (Expected - Actual) / Expected`).
*   **Invariants (Absolute Rules)**:
    *   Define **"Never-changing rules"**.
    *   Define **State Invariants**: Conditions that must be true at specific checkpoints (e.g., loop invariants).
    *   *Example*: "During the sorting process, the sub-array `A[0..i]` is always sorted."
    *   *Goal*: Use mathematical constraints (e.g., `A + B < C`) to reduce ambiguity and AI logic errors.

### Atomic Steps & Data Flow
*   **Atomic Steps**: Break down the algorithm into the smallest logical units. Number them sequentially.
*   **Data Flow Visualization**: Use Mermaid Sequence or Flowchart diagrams to show how data transforms through these steps.

## 3. Pre-Implementation Verification

Before writing any code based on these documents, you MUST perform a "Logic Check".

**Mandatory Prompt for AI**:
> "Based on the above specification, before implementing the algorithm, please point out any logical contradictions or expected performance bottlenecks. Also, suggest at least 3 edge cases that I might have missed."

---

## Example Template

```markdown
# [Module Name] Specification

## 1. Context Detail
### Inputs
- `user_id` (UUID): Unique identifier.
- `amount` (Decimal): Transaction amount.

### Outputs
- `transaction_hash` (String): On-chain tx hash.

### Constraints
- **Tech Stack**: Python, SQLAlchemy.
- **Invariant**: Account balance must never be negative. $Balance_{after} = Balance_{before} - Amount \ge 0$.

## 2. Algorithm: Balance Transfer
### Invariants
1. Total system supply remains constant: $\sum Balance_i = Constant$.

### Atomic Steps
1. Lock source and destination rows.
2. Check $Balance_{source} \ge Amount$.
3. Decrement source, Increment destination.
4. Commit transaction.

## 3. Verification
(Include the mandatory verification prompt here)
```
