# AST Library Test Suite

This directory contains the comprehensive test suite for the AST library. The tests are organized following modern Python testing best practices with clear separation of concerns and multiple testing layers.

## Quick Start

```bash
# Run all tests
uv run pytest

# Run only fast unit tests
uv run pytest -m unit

# Run with coverage report
uv run pytest --cov=ast_lib --cov-report=html
```

## Directory Structure

- **`unit/`** - Fast, isolated unit tests for individual components
- **`integration/`** - Tests for component interactions
- **`e2e/`** - End-to-end workflow tests
- **`fixtures/`** - Shared test data and fixtures
- **`helpers/`** - Test utilities and helper functions
- **`test_code/`** - Example Python code for testing
- **`conftest.py`** - Pytest configuration and fixtures
- **`factories.py`** - Test data factories for AST nodes
- **`test_infrastructure.py`** - Meta-tests for the test framework itself

## Key Features

### 1. **Test Data Factories**
Flexible factories for creating AST nodes and Python code:
```python
from tests.factories import ASTFactory, CodeFactory

func = ASTFactory.create_function("my_func", ["x", "y"])
code = CodeFactory.create_simple_class("MyClass")
```

### 2. **Property-Based Testing**
Hypothesis strategies for generating test data:
```python
from tests.helpers.hypothesis_strategies import valid_python_code

@given(valid_python_code())
def test_parse_any_valid_code(code):
    tree = ast.parse(code)
    assert isinstance(tree, ast.Module)
```

### 3. **Performance Testing**
Built-in performance measurement utilities:
```python
from tests.helpers import measure_performance

result = measure_performance(parser.parse, large_file, iterations=100)
assert result.mean < 0.1  # 100ms requirement
```

### 4. **Comprehensive Fixtures**
Session and function-scoped fixtures for common needs:
- `sample_ast_tree` - Pre-built AST for testing
- `create_test_project` - Multi-file project creator
- `mock_cache` - Functional mock cache
- `large_python_file` - Performance testing data

### 5. **CI/CD Integration**
Multi-stage GitHub Actions workflow:
1. Fast feedback (unit tests)
2. Integration tests
3. Full suite with coverage
4. E2E tests (main branch only)
5. Performance benchmarks

## Writing Tests

See [TESTING_GUIDE.md](../TESTING_GUIDE.md) for comprehensive documentation on:
- Writing effective tests
- Using test utilities
- Performance testing
- Property-based testing
- CI/CD integration
- Best practices

## Running Specific Test Types

```bash
# Unit tests only
uv run pytest -m unit

# Integration tests
uv run pytest -m integration

# End-to-end tests
uv run pytest -m e2e

# Performance benchmarks
uv run pytest -m performance --performance

# Exclude slow tests
uv run pytest -m "not slow"
```

## Test Coverage

We maintain a minimum of 80% code coverage. View coverage reports:
```bash
# Generate HTML coverage report
uv run pytest --cov=ast_lib --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Contributing

When adding new tests:
1. Place them in the appropriate layer (unit/integration/e2e)
2. Use descriptive test names that explain the behavior
3. Leverage existing factories and helpers
4. Add appropriate markers (@pytest.mark.unit, etc.)
5. Ensure tests are fast and isolated
6. Document any new test utilities