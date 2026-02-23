---
name: code-review
description: Comprehensive code review process utilizing specialized agents for correctness, performance, maintainability, security, and testing. Use this skill or command when the user asks for a code review before merging or committing changes.
license: Complete terms in LICENSE.txt
---

# Code Review Skill

This skill orchestrates a comprehensive code review by chaining the execution of specialized reviewer agents.

## Workflow 

When requested to review code, you should follow this sequence of actions:

1. **Understand Scope**: Look at the current `.git` diff (`git diff` and `git diff --staged`) or ask the user which specific files to review.
2. **Execute Primary Reviewer**: Use the `code-reviewer` agent to check for business logic Correctness, Test coverage, and Regressions.
3. **Execute Specialized Reviewers**: 
   - Use the `performance-reviewer` agent to check for N+1 queries, rendering bottlenecks, and unnecessary computations.
   - Use the `maintainability-reviewer` agent to evaluate code smells (e.g., mysterious names, long functions, duplicate code).
   - Use the `security-reviewer` agent to examine authentication, authorization, injection vulnerabilities, and other critical security aspects. (Always run this alongside the others).
4. **Compile the Report**: Gather the findings from each reviewer and present them in a unified format, organized by severity and category (Correctness, Performance, Maintainability, Security).

## Agents to utilize

Read and follow the guidelines defined for each of the following agents in the `.claude/agents` directory:
- `.claude/agents/code-reviewer.md`
- `.claude/agents/performance-reviewer.md`
- `.claude/agents/maintainability-reviewer.md`
- `.claude/agents/security-reviewer.md` 

## Unified Output Format

```markdown
# Comprehensive Code Review Report

## 1. Correctness & Testing (code-reviewer)
- [CRITICAL] ...
- [HIGH] ...

## 2. Security (security-reviewer)
- [CRITICAL] ...
- [HIGH] ...

## 3. Maintainability (maintainability-reviewer)
- [READABILITY] ...
- [CONSISTENCY] ...

## 4. Performance (performance-reviewer)
- [HIGH] ...

## Review Summary 
| Category | Pass/Fail | Critical Issues |
|---|---|---|
| Correctness | Fail | 1 |
| Security | Pass | 0 |
| Maintainability | Warn | 0 |
| Performance | Pass | 0 |

**Verdict:** WARNING - Please address the critical and high issues before merging.
```
