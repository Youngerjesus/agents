# Rule: Worklog List Creation

When creating or updating the `worklog_list.json` file in the `work_queue` directory, you MUST adhere to the following schema and formatting rules. This ensures consistency across all task tracking.

## Location
- Directory: `work_queue/`
- Filename: `worklog_list.json`

## JSON Structure & Schema
The file must be a JSON array of task objects. Each task object MUST follow this exact structure:

```json
{
    "task_id": "TASK-EXAMPLE-001",
    "task_number": 1,
    "title": "Example Task: Implement Login Feature",
    "description": "Implement the basic login functionality using username and password. This includes validating the input and checking against the database.",
    "status": "completed",
    "_status_comment": "pending(시작전), in_progress(진행중), completed(완료)",
    "priority": "high",
    "context": {
        "target_files": [
            "src/auth/domain/service.py"
        ],
        "reference_files": [
            "src/auth/domain/models.py",
            "src/shared/utils.py"
        ],
        "documentation": [
            "docs/impl_specs/auth_spec.md"
        ]
    },
    "verification": {
        "test_specs": [
            "Test Scenario 1: Verify that login succeeds with valid credentials",
            "Test Scenario 2: Ensure error is raised when password does not match hash"
        ],
        "checklist": [
            "Ensure password is hashed before comparison",
            "Return appropriate error for invalid credentials"
        ]
    },
    "dependencies": [],
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
}
```

## Field Descriptions
- **task_id**: Unique identifier string (e.g., "TASK-{PROJECT}-{NUMBER}").
- **task_number**: Integer serial number of the task.
- **title**: Concise summary of the task.
- **description**: Detailed description of what needs to be done.
- **status**: One of `pending`, `in_progress`, `completed`.
- **_status_comment**: Helper comment for status values (always "pending(시작전), in_progress(진행중), completed(완료)").
- **priority**: Importance level (e.g., `high`, `medium`, `low`).
- **context**:
    - **target_files**: List of files to be modified.
    - **reference_files**: List of files to be read for context.
    - **documentation**: Related documentation files.
- **verification**:
    - **test_specs**: High-level test scenarios.
    - **checklist**: Specific items to verify during implementation.
- **dependencies**: List of `task_id`s that this task depends on.
- **created_at**: ISO 8601 timestamp of creation.
- **updated_at**: ISO 8601 timestamp of last update.

## Guidelines
1. Always maintain valid JSON syntax.
2. Ensure dates are in UTC ISO 8601 format.
3. When adding a new task, increment the `task_number` and generate a unique `task_id`.
