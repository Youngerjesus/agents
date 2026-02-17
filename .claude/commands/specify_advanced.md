---
description: Create or update a feature specification with advanced strategic verification.
handoffs: 
  - label: Build Technical Plan
    agent: speckit.plan
    prompt: Create a plan for the spec. I am building with...
  - label: Clarify Spec Requirements
    agent: speckit.clarify
    prompt: Clarify specification requirements
    send: true
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

The text the user typed after `/specify_advanced` in the triggering message **is** the feature description. Assume you always have it available in this conversation even if `$ARGUMENTS` appears literally below. Do not ask the user to repeat it unless they provided an empty command.

Given that feature description, do this:

1. **Generate a concise short name** (2-4 words) for the branch:
   - Analyze the feature description and extract the most meaningful keywords
   - Create a 2-4 word short name that captures the essence of the feature
   - Use action-noun format when possible (e.g., "add-user-auth", "fix-payment-bug")
   - Preserve technical terms and acronyms (OAuth2, API, JWT, etc.)
   - Keep it concise but descriptive enough to understand the feature at a glance
   - Examples:
     - "I want to add user authentication" → "user-auth"
     - "Implement OAuth2 integration for the API" → "oauth2-api-integration"
     - "Create a dashboard for analytics" → "analytics-dashboard"
     - "Fix payment processing timeout bug" → "fix-payment-timeout"

2. **Check for existing branches before creating new one**:

   a. First, fetch all remote branches to ensure we have the latest information:

      ```bash
      git fetch --all --prune
      ```

   b. Find the highest feature number across all sources for the short-name:
      - Remote branches: `git ls-remote --heads origin | grep -E 'refs/heads/[0-9]+-<short-name>$'`
      - Local branches: `git branch | grep -E '^[* ]*[0-9]+-<short-name>$'`
      - Specs directories: Check for directories matching `specs/[0-9]+-<short-name>`

   c. Determine the next available number:
      - Extract all numbers from all three sources
      - Find the highest number N
      - Use N+1 for the new branch number

   d. Run the script `.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS"` with the calculated number and short-name:
      - Pass `--number N+1` and `--short-name "your-short-name"` along with the feature description
      - Bash example: `.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS" --json --number 5 --short-name "user-auth" "Add user authentication"`
      - PowerShell example: `.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS" -Json -Number 5 -ShortName "user-auth" "Add user authentication"`

   **IMPORTANT**:
   - Check all three sources (remote branches, local branches, specs directories) to find the highest number
   - Only match branches/directories with the exact short-name pattern
   - If no existing branches/directories found with this short-name, start with number 1
   - You must only ever run this script once per feature
   - The JSON is provided in the terminal as output - always refer to it to get the actual content you're looking for
   - The JSON output will contain BRANCH_NAME and SPEC_FILE paths
   - For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot")

3. Load `.specify/templates/spec-template.md` to understand required sections.

4. Follow this execution flow regarding **Advanced Strategic Verification** during spec generation:

    1. Parse user description from Input
    2. Extract key concepts
    3. Generate Functional Requirements & Success Criteria
    4. **[NEW] Perform Pre-Spec Strategy Check**:
       Ask yourself these critical questions *before* finalizing the content:
       - **Logical Assessment**: Are there any holes in the logic?
       - **Pre-Mortem**: If this project fails, what would be the cause? (Identify risks).
       - **Optimality**: Is this the best way? (Consider if AI research can suggest a better solution to overcome human bias).
       - **Readiness**: Are all ingredients ready? (External services, API keys, data sources).
       - **Artifact Check**: Does the expected final artifact look valid and valuable?
    5. **Incorporate Findings**:
       - If significant risks or better alternatives are found, include a **"Strategic Considerations & Risks"** section in the spec.
       - If specific prerequisites are missing, list them as **Blocking Dependencies**.
    6. Return: SUCCESS (spec ready for writing)

5. Write the specification to SPEC_FILE using the template structure, replacing placeholders with concrete details derived from the feature description and your strategic analysis.

6. **Specification Quality Validation**: After writing the initial spec, validate it against quality criteria:

   a. **Create Spec Quality Checklist**: Generate a checklist file at `FEATURE_DIR/checklists/requirements.md` using the checklist template structure with these validation items:

      ```markdown
      # Specification Quality Checklist: [FEATURE NAME]
      
      **Purpose**: Validate specification completeness, quality, and STRATEGIC SOUNDNESS before proceeding.
      **Created**: [DATE]
      **Feature**: [Link to spec.md]
      
      ## Content Quality
      
      - [ ] No implementation details (languages, frameworks, APIs)
      - [ ] Focused on user value and business needs
      - [ ] Written for non-technical stakeholders
      - [ ] All mandatory sections completed
      
      ## Advanced Strategic Verification (CRITICAL)
      
      - [ ] **Logical Integrity**: No logical loopholes or inconsistencies found.
      - [ ] **Risk Assessment (Pre-Mortem)**: Potential failure modes identified and mitigated.
      - [ ] **Optimality Check**: Confirmed this is the best approach (checking against known alternatives).
      - [ ] **Readiness**: All external dependencies/services are confirmed available/ready.
      - [ ] **Artifact Validity**: The expected output seems valid and achievable.
      
      ## Requirement Completeness
      
      - [ ] No [NEEDS CLARIFICATION] markers remain
      - [ ] Requirements are testable and unambiguous
      - [ ] Success criteria are measurable
      - [ ] Success criteria are technology-agnostic (no implementation details)
      - [ ] All acceptance scenarios are defined
      - [ ] Scope is clearly bounded
      - [ ] Dependencies and assumptions identified
      
      ## Feature Readiness
      
      - [ ] All functional requirements have clear acceptance criteria
      - [ ] User scenarios cover primary flows
      - [ ] Feature meets measurable outcomes defined in Success Criteria
      - [ ] No implementation details leak into specification
      
      ## Notes
      
      - Items marked incomplete require spec updates.
      ```

   b. **Run Validation Check**: Review the spec against each checklist item.
   c. **Handle Validation Results**: (Same logic as standard specify - iterate until passing or clarify).

7. Report completion with branch name, spec file path, checklist results, and readiness for the next phase.

**NOTE:** The script creates and checks out the new branch and initializes the spec file before writing.

## General Guidelines

- **Focus on STRATEGY and VALUE**.
- **Challenge the premise**: If the requested feature seems flawed, use the "Strategic Considerations" section to politely suggest better alternatives based on your analysis.
- **Verify Assumptions**: Don't just assume an API exists—verify it mentally or check if it's a known common service.
