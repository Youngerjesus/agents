import os
import subprocess
import json
from pathlib import Path

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_file(path, content=""):
    """Creates a file with the given content if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        with open(path, "w") as f:
            f.write(content)
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")

def initialize_poetry():
    """Initializes a poetry project."""
    if not Path("pyproject.toml").exists():
        print("Initializing Poetry project...")
        try:
            subprocess.run(["poetry", "init", "-n"], check=True)
            print("Poetry initialized successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error initializing Poetry: {e}")
        except FileNotFoundError:
            print("Poetry not found. Please install Poetry first.")
    else:
        print("pyproject.toml already exists. Skipping Poetry initialization.")

def main():
    # Define directories to create
    directories = [
        "docs/apis",
        "docs/adrs",
        "specs",
        "work_queue",
        "contexts",
        "src",
        "tests",
        "temp"
    ]

    # Define files to create with their content
    files = {
        "work_queue/progress.md": "# Project Progress\n\n- [ ] Initial Setup\n",
        "work_queue/worklog_list.json": json.dumps([], indent=2),
        "temp/todo.md": "# Todo List\n",
        "temp/temp.md": "# Temporary Notes\n",
        ".env": "",
        ".gitignore": """
# Python
__pycache__/
*.py[cod]
*$py.class

# Poetry
.venv/
poetry.lock

# Environment variables
.env

# IDEs
.vscode/
.idea/

# Temp
temp/
""",
        "README.md": "# Project Name\n\nDescription of the project.\n"
    }

    print("Starting project setup...")

    # Create directories
    for directory in directories:
        create_directory(directory)

    # Create files
    for file_path, content in files.items():
        create_file(file_path, content)

    # Initialize Poetry
    initialize_poetry()

    print("Project setup complete.")

if __name__ == "__main__":
    main()
