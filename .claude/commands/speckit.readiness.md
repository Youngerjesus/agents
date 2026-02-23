---
description: Verify if the project is fully ready for implementation by checking for missing external service details, unresolved discussions, and ambiguities across all spec files.
---

## User Input

```text
$ARGUMENTS
```

## Goal

Determine if the project is **"Ready to Code"** (Implementation Phase).
This command scans all project specification artifacts (`spec.md`, `plan.md`, `design.md`, `contracts/*`, `research.md`, `tasks.md`) to identify **blockers** that must be resolved before writing code.
It specifically looks for:
1.  **External Service Readiness**: Verified credentials (API Keys issued), official documentation links, and defined error handling.
2.  **Environment Setup**: Database schema defined, Docker configuration ready (if applicable), and local dev environment specified.
3.  **Data Pipeline Completeness**: **Input** (Request/Event), **Process** (Internal Models/State), and **Output** (Response/Artifact) models are fully defined. *Software is data transformation.*
4.  **Unresolved Decisions**: "TBD", "TODO", "???", "Decision Required", "Open Question".
5.  **Ambiguous Logic**: Flows with missing error states or undefined "happy paths".
6.  **Data Model Completeness**: Types defined for all fields to prevent implementation ambiguity (e.g., `status` as int vs string).

## Execution Steps

### 1. Initialize Context

Run `.specify/scripts/bash/check-prerequisites.sh --json --require-spec --require-plan --require-design` (if available) or assume standard paths.
Load the following files (if they exist):

- `FEATURE_DIR/spec.md`
- `FEATURE_DIR/plan.md`
- `FEATURE_DIR/design.md`
- `FEATURE_DIR/contracts/*.md` (API contracts)
- `FEATURE_DIR/research.md`
- `FEATURE_DIR/tasks.md` (if available)
- `.claude/CLAUDE.md` (Project Rules)

### 2. Readiness Checks (Deep Scan)

Perform the following specific checks. If any **BLOCKER** is found, the project is **NOT READY**.

#### A. External Service Readiness (BLOCKER)
For every external service mentioned (e.g., Toss, Kakao, Gemini, Ablecity, AWS, Firebase, etc.):
- **Are API Credentials Issued?** (Is the API Key available or is there a clear path/link to get it?)
- **Is the API Documentation Linked?** (Direct link to official guides/references)
- **Is the Authentication Method defined?** (API Key, OAuth, JWT, etc.)
- **Are critical Endpoints listed?** (URLs or SDK methods)
- **Are Error/Failure modes defined?** (Rate limits, timeouts, 4xx/5xx handling)

#### B. Environment & Infrastructure (BLOCKER)
- **Database**: Is the schema defined (e.g., `data-model.md` or SQL file)?
- **Docker**: Is `docker-compose.yml` or container strategy defined (if applicable)?
- **Config**: Are all required environment variables listed (e.g., `.env.example` content defined)?
- **Local Dev**: Is the local development setup (e.g., `npm run dev`, `python main.py`) documented?

#### C. Unresolved "TBD" Items (BLOCKER)
Scan all files for:
- `TODO`
- `TBD` (To Be Determined)
- `???`
- `<placeholder>`
- `Decision needed`
- `Confirm with user`
- `FIXME`

#### D. Ambiguity & Logic Gaps (WARNING / BLOCKER)
- **Data Model Types**: Are all fields in `data-model.md` explicitly typed? (e.g., `status` as `Enum(pending, paid)` vs `string`). *Undefined types lead to implementation ambiguity.*
- **Flows**: Do sequence diagrams in `design.md` match the API endpoints in `contracts/api.md`?

#### D. Review Status (BLOCKER)
- Have `spec.md`, `plan.md`, and `design.md` been explicitly "Approved" or marked as "Final"? (Check for metadata or generic status).

### 3. Generate Readiness Report

Output a structured Markdown report.

## Spec Implementation Readiness Report

### 🚦 Verdict: [ GO / NO-GO ]
*(Calculated based on Critical Blockers count. 0 = GO, >0 = NO-GO)*

### 1. Critical Blockers (Must Fix)
| ID | File | Issue | Action Required |
|----|------|-------|-----------------|
| B1 | `plan.md` | API Key for 'Toss' not documented | Add auth method to `plan.md` or `design.md` |
| B2 | `design.md` | "TODO: Define error state" at L150 | Define UI behavior for 500 errors |

### 2. Warnings (Risks)
| ID | File | Issue | Recommendation |
|----|------|-------|----------------|
| W1 | `spec.md` | Performance goal "Fast" is vague | Define latency in ms (e.g., <200ms) |

### 3. External Service Checklist
| Service | Creds Issued? | Docs Linked? | Auth Method | Endpoints | Status |
|---------|---------------|--------------|-------------|-----------|--------|
| Toss | ✅ | ✅ | ✅ | ✅ | **Ready** |
| Kakao | ❌ (Issue #10) | ✅ | ✅ | ❌ | **Not Ready** |

### 4. Implementation Plan Review
- **Tasks Generated?**: [Yes/No]
- **Tech Stack Finalized?**: [Yes/No]
- **Database Schema Frozen?**: [Yes/No]
- **Environment Configured?**: [Yes/No]

---

### Recommended Next Steps

- **If NO-GO**:
  1. Resolve all Critical Blockers.
  2. Run `/spec-impl-ready` again.

- **If GO**:
  1. Proceed to `/speckit.implement` (or `speckit.tasks` if tasks are missing).
