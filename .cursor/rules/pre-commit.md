# Pre-commit Checks

After making any code changes, always run pre-commit checks to ensure code quality and catch issues early.

## Rule

Use `source .venv/bin/activate && pre-commit run --all-files` to check for issues after making changes.

## When to Run

- After editing any Python files
- After making changes to configuration files
- Before committing code
- After completing a feature or fix

## What It Checks

The pre-commit configuration includes:
- **ruff**: Linting and code quality checks
- **ruff-format**: Code formatting checks

## Fixing Issues

If pre-commit finds issues:
1. Review the errors reported
2. Fix the issues in the code
3. Re-run pre-commit to verify fixes
4. Ensure all checks pass before considering the task complete
