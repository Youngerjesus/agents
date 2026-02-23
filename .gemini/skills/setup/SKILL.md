---
name: setup
description: Initialize a new project with standard directory structure and Poetry configuration. Use this skill when starting a new project to ensure consistency and setup the workspace according to best practices.
---

# Setup Project Skill

This skill automates the setup of a new project by creating a standard directory structure, initializing essential files, and setting up a Poetry environment.

## Usage

To initialize the project, run the setup script:

```bash
python3 .claude/skills/setup/scripts/setup_project.py
```

This script will:

1.  **Create Directories**:
    -   `docs/apis`, `docs/adrs`, `specs`
    -   `work_queue`
    -   `contexts`
    -   `src`
    -   `tests`
    -   `temp`

2.  **Create Files**:
    -   `work_queue/progress.md`
    -   `work_queue/worklog_list.json`
    -   `temp/todo.md`
    -   `temp/temp.md`
    -   `.env`
    -   `.gitignore`
    -   `README.md`

3.  **Initialize Poetry**:
    -   Runs `poetry init -n` to create a `pyproject.toml` file.
