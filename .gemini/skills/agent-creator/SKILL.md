---
name: agent-creator
description: Guide for creating effective custom subagents in Claude Code. This skill should be used when the user wants to create a new specialized agent (subagent) to handle specific tasks, roles, or domains (e.g., "create a python expert agent", "make a code reviewer agent").
---

# Agent Creator

This skill provides guidance for creating effective custom subagents in Claude Code.

## About Subagents

Subagents are specialized AI assistants that handle specific types of tasks. Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions.

### When to Create a Subagent

Create a subagent when:
1.  **Specialized Role**: The user needs an expert in a specific field (e.g., "security auditor", "frontend designer").
2.  **Specific Workflow**: The task involves a repeatable, definitive process (e.g., "triaging bugs", "writing documentation").
3.  **Tool Constraints**: You want to limit the agent's capabilities for safety or focus (e.g., "read-only researcher").
4.  **Context Management**: You want to keep the main conversation clean and offload verbose tasks.

## Core Principles

1.  **Focused Scope**: A subagent should do one thing well. Avoid "kitchen sink" agents.
2.  **Clear Description**: The `description` field is CRITICAL. Claude uses it to decide when to route tasks to this agent.
3.  **Appropriate Tools**: Give the agent only the tools it needs.
4.  **Distinct Persona**: Give the agent a clear system prompt defining its role, style, and constraints.

## Agent Creation Process

To create a new agent, you will generate a Markdown file with YAML frontmatter in the `.claude/agents/` directory (for project-specific agents) or `~/.claude/agents/` (for global user agents).

### Step 1: Define the Agent

Ask the user (or infer from context) the following:
-   **Name**: A unique, hyphenated name (e.g., `python-expert`, `security-auditor`).
-   **Description**: A clear summary of what the agent does and *when* it should be used.
-   **Role/Persona**: What kind of expert is this agent? (e.g., "Senior Python Backend Engineer").
-   **Tools**: What tools should it have access to? (Default: `inherit` from parent, or specific list like `Read, Bash, Write`).

### Step 2: Create the Agent File

Create a file at `.claude/agents/<agent-name>.md` with the following structure:

```markdown
---
name: <agent-name>
description: <description-for-router>
tools: <tool-list-or-inherit>
model: <model-alias-or-inherit>
---

<System Prompt / Agent Persona>
```

#### Frontmatter Fields

| Field | Description | Example |
| :--- | :--- | :--- |
| `name` | Unique identifier (lowercase, hyphens) | `code-reviewer` |
| `description` | **Crucial**. Tells Claude *when* to use this agent. | "Expert code reviewer. Use for reviewing PRs and checking best practices." |
| `tools` | List of allowed tools. | `Read, Grep, Glob, Bash` |
| `model` | `sonnet`, `opus`, `haiku`, or `inherit`. | `sonnet` |

#### System Prompt Guidelines

The body of the file is the System Prompt. It should include:
1.  **Role Definition**: "You are a [Role]..."
2.  **Responsibility**: What is the agent responsible for?
3.  **Workflow**: Steps the agent should typically follow.
4.  **Constraints**: What the agent should NOT do.
5.  **Output Style**: How the agent should communicate.

### Example: Python Expert

File: `.claude/agents/python-expert.md`

```markdown
---
name: python-expert
description: A specialized agent for writing, refactoring, and debugging Python code. Use for complex Python tasks requiring adherence to PEP 8 and modern practices.
tools: inherit
model: sonnet
---

You are a Senior Python Engineer and Architect.

**Your Goal**: Write high-quality, efficient, and maintainable Python code.

**Guidelines**:
-   Always follow PEP 8 style guidelines.
-   Use type hinting (mypy) for all function signatures.
-   Prefer modern Python features (3.10+).
-   Write clear docstrings (Google style).
-   When debugging, analyze the root cause before suggesting fixes.
-   Prioritize code readability and performance.

**Workflow**:
1.  Analyze the request.
2.  If modifying existing code, read the file first.
3.  Plan your changes.
4.  Implement the changes using best practices.
5.  Verify the code (if possible/requested).
```

### Step 3: Verify

After creating the file:
1.  Inform the user the agent has been created.
2.  Tell the user they can invoke it by asking Claude: "Use the [agent-name] agent to..." or simply by assigning a task that matches the description.
