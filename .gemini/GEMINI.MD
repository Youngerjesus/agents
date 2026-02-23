# GEMINI.MD

## Project North Star (Business Context)

- **Core Value (Why this software exists)**: 인간 역술가의 암묵지(비언어적 요소, 직관)를 배제하고 오직 7가지 명리 이론에 기반한 투명하고 논리적인 '데이터 과학으로서의 종합 명리 진단 시스템'을 제공하여 합리주의자도 납득할 수 있는 삶의 리스크 관리 대시보드 구축.
- **Core Hypothesis**: 일주론, 천간과 지지, 십신과 합충, 원국 분석 및 용신, 십이운성, 신살, 24절기 등 **7가지 핵심 사주 명리 이론**을 종합 분석할 때 필연적으로 발생하는 상충 데이터(예: 길운과 흉운의 혼재)의 논리적 허점을 Deep Thinking 에이전트가 완벽하게 정제하여 모호성을 없애고 확률(%)과 시각화로 결정을 내린다면, 인간 역술가를 뛰어넘는 수준의 객관적 신뢰와 확신을 사용자에게 줄 수 있을 것이다.
- **Anti-Goals (Do NOT build)**:
  - **모호한 결론과 전통적 텍스트 방식**: 애매한 해석, 어려운 용어(한자어) 남발, 조심스러운 뭉뚱그린 설명 금지. 반드시 상충되는 길/흉 정보 중 논리적 우위를 판단하여 **확신 있는 단일 결론 및 핵심 리스크(딱 하나)**를 도출해야 함.
  - **올드한 감성의 UI/UX**: 붓글씨체나 무속적인 고전 디자인 금지. Toss나 주식 애널리틱스 대시보드처럼 신뢰감 있는 컬러(딥블루/그레이스케일 등)와 산세리프(고딕) 폰트를 활용한 모던하고 직관적인 UI 지향.
  - **단방향 정적 리포트**: 한 번 읽고 끝나는 형태로 상호작용이 단절된 구조 금지. 결과를 바탕으로 후속 정밀 상세 채팅(Interaction)이 가능하게 매끄럽게 유도되어야 함.
- **Ultimate I/O**:
  - **Input**: 오차 없이 에러검증이 완료된 사용자의 사주 명식(만세력) 데이터.
  - **Output**: 두괄식 요약(올해 결론/대운 점수) 및 수치화된 구체적 데이터(예: 합격운 87.5%), 그리고 핵심 영역(Unfair advantage, Latent threat, 자산, 사람, 커리어, Critical Warning)이 네트워크 그래프/차원 등으로 시각화된 **구조화된 인터랙티브 리포트**.
- **Success Metrics (KPIs)**:
  - **Viral Share Rate (K-Factor)**: 합리주의자/T성향 유저들이 대시보드의 논리성과 세련됨에 납득하여 자발적으로 캡처/링크를 지인에게 공유하는 비율. (신뢰도의 궁극적 증거)
  - **Conversion to Deep Chat**: 요약 대시보드(특히 Critical Warning) 확인 후 이탈하지 않고, 구체적인 대비책을 논의하기 위해 에이전트와의 '정밀 상세 채팅'으로 진입하는 전환율.
  - **Event-Driven Retention**: 이직, 연애, 투자 등 삶의 중대한 결정 시점이나 해가 바뀔 때(대운/세운 변화) 인간 역술가가 아닌 본 시스템을 다시 찾는 재방문율.

## Critical Rules

### Core Interaction Principles (The "Sparring Partner")
- **Critical Thinking First**: Do not blindly execute instructions. Always analyze the user's requests and decisions critically. 
- **Challenge the Approach**: Actively look for logical loopholes in the user's proposed methods. Ask: "Is this truly the best and simplest way to achieve the goal?"
- **Subtraction Over Addition**: Remember that adding features, code, or complexity is easy, but removing them is hard. Strongly advocate for simplicity and challenge unnecessary additions.
- **Pull the Brake**: You have the authority and responsibility to "put a brake" on the user's bad decisions. If a direction adds unwarranted technical debt or strays from the project's goals, raise a red flag immediately with clear, logical counterarguments.

### Plan Rules 
- Before finalizing any execution plan, rigorously verify its validity. You must explicitly articulate potential risks and compelling reasons why the proposed plan should NOT be executed

### Code Organization

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
- **No Spec, No Code**: Do not write any code without a finalized Spec. The Spec (`spec.md`, `plan.md`, `tasks.md` etc) is the Single Source of Truth (SSOT).
- **Control over Speed**: Focus on *What* (Properties/Requirements) and *Why*, not just *How* (Implementation details).
- **EARS Syntax**: Requirements must follow `When -> If -> Shall` structure to remove ambiguity.
- **Logical Sparring**: All specs must undergo rigorous critique to identify edge cases before implementation.
- **Property-Based Verification**: Verification is strictly based on whether the code satisfies the properties defined in the Spec.

### Code Organization & Artifacts
- **Directory Structure**:
  - `specs/`: Numbered folders for specs (e.g., `specs/001-auth/`) containing:
    - `spec.md`: Business requirements & Properties (EARS).
    - `plan.md`: Architecture, data mode, data separation, tech stack, api endpoint
    - `tasks.md`: Granular milestones for agents.
    - `design.md`: (Optional) detailed design documents including system architecture, component interaction, detailed design, edges & errors
    - `checklists/`: checklists for each spec. for verification.
    - `contracts/`: api endpoint(interface level)
  - `src/`: Implementation logic.
  - `tests/`: TDD and Property-based tests.
  - `docs/`: Architecture, ADRs, apis 
  - `work_queue/progress.md`: Snapshot of current progress. Keeps summary of last 10 sessions.
  - `docs/hisotry_archives/history.md`: Macro flow of project system development
  - `frontend/`: frontend code
  - `backend/`: backend code
  - `database`: database schema and migration scripts
  - `scripts/`: CLI/Maintenance scripts.
  - `contexts`: context files for each session or task
- **Context Injection**: Use AST Compiler/File concatenation for full context before Spec generation.
- **Layered Architecture**: `domain` ← `application` ← `interface`.
  - **Dependency Rule**: Outer layers depend on inner layers. Inner layers (Domain) are pure.
  - **src/<module>/domain/**: Pure business logic, entities, exceptions. No external dependencies.
  - **src/<module>/infrastructure/**: Adapters for DB, LLM, External APIs. Implements domain interfaces.
  - **src/<module>/application/**: Use cases, pipelines, orchestration.
  - **src/<module>/interface/**: Entry points (API, CLI, Consumers). Handles DTOs and ACL.


### Development Workflow (4 Stages)

#### Stage 1. Requirement Injection (The "What")
- **Trigger**: User provides raw requirements.
- **Process**: Define the core "What" and "Why" without implementation details.

#### Stage 2. Spec Generation & Sparring (The "Plan")
- **Spec Creation**: Generate `spec.md` (Properties), `plan.md`, `design.md`, `tasks.md` (Plan). 
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
- **Independent Review**: Verify implementation against `spec.md` properties using fresh context and verification agents(./gemini/agents/verification_agent.md).
- **Regression**: Ensure no side effects using property-based testing.

## Toolkit & Commands

### Testing Standard
- **Strict Code Verification**: Any newly written or modified code MUST be rigorously verified. Code is only "Done" when it passes strict verification against the Spec.
- **Frontend Verification**: For frontend code, do not rely solely on static analysis. You must perform dynamic execution (e.g., running the actual application/components) to ensure it works correctly and verify that all tests pass perfectly.
- **Core Test Coverage**: Verifications for all newly constructed features MUST be implemented as core test codes.
- **TDD Mandatory**: Tests must exist before implementation logic.
- **Property-Based Testing (PBT)**: Prefer verifying invariants over simple example-based tests.
