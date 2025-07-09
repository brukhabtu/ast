# AST CLI Usage Guide

The AST library provides a command-line interface for navigating and analyzing Python codebases.

## Installation

```bash
pip install -e .
```

## Commands

### find-function

Search for function definitions in Python files.

```bash
ast find-function <name> [options]
```

#### Arguments

- `name`: Function name or pattern to search for. Supports wildcards:
  - `*` - matches any sequence of characters
  - `?` - matches any single character

#### Options

- `-p, --path PATH`: Directory to search in (default: current directory)
- `-f, --format {plain,json,markdown}`: Output format (default: plain)
- `--ignore PATTERN`: Patterns to ignore (can be specified multiple times)
- `--no-progress`: Disable progress bar
- `--case-sensitive`: Use case-sensitive matching

## Examples

### Basic Usage

Find all functions named "main":
```bash
ast find-function main
```

Output:
```
/path/to/file.py:10:0 - main
  Main entry point for the application.
```

### Pattern Matching

Find all test functions:
```bash
ast find-function "test_*"
```

Find all functions containing "parse":
```bash
ast find-function "*parse*"
```

### Search in Specific Directory

Search only in the src directory:
```bash
ast find-function "*" --path ./src
```

### Output Formats

#### JSON Format
```bash
ast find-function main --format json
```

Output:
```json
[
  {
    "name": "main",
    "file": "/path/to/file.py",
    "line": 10,
    "column": 0,
    "end_line": 15,
    "end_column": 0,
    "docstring": "Main entry point for the application."
  }
]
```

#### Markdown Format
```bash
ast find-function main --format markdown
```

Output:
```markdown
# Search Results

## `main` in /path/to/file.py
- **Location**: Line 10, Column 0
- **Docstring**: Main entry point for the application.
```

### Ignoring Patterns

Ignore specific directories or files:
```bash
ast find-function "*" --ignore "tests/*" --ignore "build/*" --ignore "*_test.py"
```

### Case-Sensitive Search

By default, searches are case-insensitive. For case-sensitive matching:
```bash
ast find-function "TestCase" --case-sensitive
```

### Large Repositories

For large repositories, the progress bar shows scanning progress:
```bash
ast find-function "process_*"
```

Disable the progress bar with:
```bash
ast find-function "process_*" --no-progress
```

## .gitignore Support

The CLI automatically respects `.gitignore` patterns and common Python ignore patterns:
- `__pycache__/`
- `*.pyc`, `*.pyo`, `*.pyd`
- `venv/`, `.venv/`, `env/`
- `build/`, `dist/`
- `.coverage`, `htmlcov/`
- And more...

## Performance Tips

1. **Use specific paths**: Narrow down the search to relevant directories
   ```bash
   ast find-function "api_*" --path ./src/api
   ```

2. **Use ignore patterns**: Exclude known irrelevant directories
   ```bash
   ast find-function "*" --ignore "node_modules/*" --ignore "vendor/*"
   ```

3. **Disable progress for scripts**: When using in scripts or CI
   ```bash
   ast find-function "test_*" --format json --no-progress
   ```

## Integration Examples

### Finding all test functions in a project
```bash
ast find-function "test_*" --path ./tests --format json | jq '.[].name'
```

### Generating a function index
```bash
ast find-function "*" --format markdown > FUNCTION_INDEX.md
```

### Checking for specific patterns in CI
```bash
# Find all TODO functions
ast find-function "todo_*" || echo "No TODO functions found"
```

### Creating a function map
```bash
# Export all functions to JSON and process with other tools
ast find-function "*" --format json > functions.json
```

## Error Handling

The CLI handles various error conditions gracefully:

- **File not found**: Silently skips missing files
- **Syntax errors**: Skips files with syntax errors and continues
- **Permission errors**: Skips files without read permissions
- **Invalid patterns**: Reports clear error messages

## Exit Codes

- `0`: Success
- `1`: Error (invalid path, invalid arguments)
- `2`: Invalid command line arguments