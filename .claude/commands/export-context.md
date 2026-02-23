---
description: Extracts the current workspace context into a high-density file to allow seamless continuation of work in a different session.
---

# Task: Export High-Density Context for Session Handoff

Based on the operations performed in the current session, generate a highly compressed context file so that the user can seamlessly continue their work in a new chat session.

## Requirements
1. **Remove Unnecessary Content**: Strictly exclude all greetings, generalized advice, and non-essential text. Only record technical facts and the current state.
2. **Maximize Information Density**: Clearly state the overarching goals, completed implementations, issues encountered, and precise next steps.
3. **Naming Convention**: Generate the file sequentially in the `contexts/` directory (create the directory if it doesn't exist) with the format `contexts/001_task_name.md`, `contexts/002_task_name.md`, etc., incorporating the sequence number and a brief description of the task. Check existing files in the `contexts` folder to determine the correct next sequence number.

## Contents to Include
- **Current Objective**: The ultimate goal and background context the current session was aiming to achieve.
- **Done**: Code changes made, files created/modified, and issues resolved up to this point.
- **Key Decisions**: Crucial technical decisions such as architectural changes, selected libraries, and reasons for specific implementations that a new session must know as constraints.
- **WIP & Next Steps**: Immediate code to write next, unresolved bugs, and remaining To-Dos.

Synthesize the above information and write the context file. Once generation is complete, instruct the user to start a new session by attaching the newly created context file (e.g., `@contexts/001_feature_implementation.md`) and then close the current session.

## Reference Information

Before generating the context, gather the following information:

1. **Existing Context Files**: Check `contexts/` directory for sequencing
2. **Current Git Status**: Run `git status` to see current changes
3. **Summary of Recent Changes**: Run `git diff --stat HEAD` to see modified files
4. **Current Branch**: Run `git branch --show-current`
