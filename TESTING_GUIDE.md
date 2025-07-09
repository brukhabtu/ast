# AST Library Testing Guide

This guide provides comprehensive documentation for using the AST library's testing infrastructure. It covers test organization, writing effective tests, using test utilities, and maintaining test quality.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Organization](#test-organization)
3. [Writing Tests](#writing-tests)
4. [Test Utilities](#test-utilities)
5. [Property-Based Testing](#property-based-testing)
6. [Performance Testing](#performance-testing)
7. [CI/CD Integration](#cicd-integration)
8. [Best Practices](#best-practices)

## Quick Start

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run only unit tests (fast feedback)
uv run pytest tests/unit -m unit

# Run with coverage
uv run pytest --cov=ast_lib --cov-report=html

# Run tests in parallel
uv run pytest -n auto

# Run specific test file
uv run pytest tests/unit/test_parser.py

# Run tests matching pattern
uv run pytest -k "test_parse"
```

### Test Markers

```bash
# Run by marker
uv run pytest -m unit          # Fast unit tests only
uv run pytest -m integration   # Integration tests
uv run pytest -m e2e          # End-to-end tests
uv run pytest -m "not slow"   # Exclude slow tests
uv run pytest -m performance --performance  # Performance benchmarks
```

## Test Organization

### Directory Structure

```
tests/
├── unit/                 # Fast, isolated unit tests
│   ├── test_parser.py
│   ├── test_symbols.py
│   └── test_cache.py
├── integration/          # Component interaction tests
│   ├── test_parser_symbols.py
│   └── test_import_resolution.py
├── e2e/                  # Full workflow tests
│   └── test_complete_analysis.py
├── fixtures/             # Shared test data
├── helpers/              # Test utilities
│   ├── __init__.py
│   ├── ast_helpers.py
│   ├── code_helpers.py
│   ├── performance_helpers.py
│   └── hypothesis_strategies.py
├── test_code/            # Example Python files
│   ├── edge_cases/
│   ├── common_patterns/
│   └── large_files/
├── conftest.py           # Pytest configuration
├── factories.py          # Test data factories
└── test_infrastructure.py # Meta-tests
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ComponentName>`
- Test functions: `test_<behavior>_<condition>_<expected_result>`

Examples:
```python
# Good test names
def test_parse_function_with_decorators_returns_ast()
def test_cache_get_nonexistent_key_returns_none()
def test_symbol_extraction_nested_classes_includes_all()

# Poor test names
def test_parse()  # Too vague
def test_1()      # Non-descriptive
def test_it_works()  # Unclear what "it" is
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from ast_lib.parser import Parser
from tests.factories import CodeFactory
from tests.helpers import assert_ast_structure

class TestParser:
    """Test the Parser component."""
    
    def test_parse_simple_function(self):
        """Test parsing a simple function definition."""
        # Arrange
        code = CodeFactory.create_simple_function("my_func", ["x", "y"])
        parser = Parser()
        
        # Act
        result = parser.parse(code)
        
        # Assert
        assert result is not None
        func_node = assert_ast_structure(
            result.body[0], 
            ast.FunctionDef,
            name="my_func"
        )
        assert len(func_node.args.args) == 2
```

### Using Fixtures

```python
def test_parse_large_file(large_python_file):
    """Test parsing performance on large files."""
    parser = Parser()
    result = parser.parse(large_python_file)
    assert result is not None

def test_multi_file_project(create_test_project, temp_project_dir):
    """Test analyzing a multi-file project."""
    # Create project structure
    project = create_test_project({
        "main.py": "from lib import helper",
        "lib.py": "def helper(): pass"
    }, temp_project_dir)
    
    # Test cross-file analysis
    analyzer = ProjectAnalyzer(project)
    imports = analyzer.analyze_imports()
    assert "lib" in imports["main.py"]
```

### Testing Error Cases

```python
def test_parse_syntax_error_raises_exception(syntax_error_codes):
    """Test that syntax errors are handled properly."""
    parser = Parser()
    
    for code in syntax_error_codes:
        with pytest.raises(SyntaxError):
            parser.parse(code)

def test_cache_concurrent_access_thread_safe(mock_cache):
    """Test cache handles concurrent access safely."""
    import threading
    
    def access_cache():
        for i in range(100):
            mock_cache.set(f"key_{i}", ast.parse(f"x = {i}"))
            mock_cache.get(f"key_{i}")
    
    threads = [threading.Thread(target=access_cache) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should complete without errors
```

## Test Utilities

### AST Helpers

```python
from tests.helpers import (
    assert_ast_equal,
    assert_ast_structure,
    find_nodes,
    count_nodes,
)

# Compare AST structures
tree1 = ast.parse("x = 1")
tree2 = ast.parse("x = 1")
assert_ast_equal(tree1, tree2)

# Find specific nodes
functions = find_nodes(tree, ast.FunctionDef)
classes = find_nodes(tree, ast.ClassDef)

# Count nodes
total_nodes = count_nodes(tree)
binops = count_nodes(tree, ast.BinOp)

# Assert structure with field checking
func = assert_ast_structure(
    node,
    ast.FunctionDef,
    name="my_func",
    returns=None  # No return annotation
)
```

### Code Generation Factories

```python
from tests.factories import ASTFactory, CodeFactory, EdgeCaseFactory

# Create AST nodes
func = ASTFactory.create_function("test", ["x", "y"])
cls = ASTFactory.create_class("TestClass", bases=[ast.Name("BaseClass")])
module = ASTFactory.create_module([func, cls])

# Generate code strings
simple_func = CodeFactory.create_simple_function()
async_code = CodeFactory.create_async_patterns()
edge_case = EdgeCaseFactory.create_deeply_nested_code(depth=5)

# Generate test suites
test_suite = TestDataGenerator.generate_test_suite()
for name, pattern in test_suite.items():
    tree = ast.parse(pattern.code)
    # Test with various patterns
```

### Performance Helpers

```python
from tests.helpers import (
    measure_performance,
    assert_performance,
    benchmark_context,
)

# Measure function performance
result = measure_performance(
    parser.parse,
    large_code,
    iterations=100,
    warmup=10
)
print(f"Mean time: {result.mean:.6f}s")

# Assert performance requirements
assert_performance(
    parser.parse,
    simple_code,
    max_duration=0.001,  # 1ms max
    iterations=100
)

# Benchmark code blocks
with benchmark_context("Complex parsing") as times:
    for code in test_codes:
        with times:
            result = parser.parse(code)
```

## Property-Based Testing

### Using Hypothesis

```python
from hypothesis import given, strategies as st
from tests.helpers.hypothesis_strategies import (
    valid_identifier,
    ast_function_def,
    valid_python_code,
)

class TestPropertyBased:
    
    @given(valid_identifier())
    def test_parse_any_valid_identifier(self, identifier):
        """Test parsing any valid Python identifier."""
        code = f"{identifier} = 42"
        tree = ast.parse(code)
        assert tree.body[0].targets[0].id == identifier
    
    @given(ast_function_def())
    def test_extract_symbols_from_any_function(self, func_def):
        """Test symbol extraction works for any function."""
        module = ast.Module(body=[func_def])
        symbols = extract_symbols(module)
        assert func_def.name in symbols
    
    @given(valid_python_code())
    def test_parse_does_not_crash(self, code):
        """Test parser handles any valid Python code."""
        try:
            tree = ast.parse(code)
            assert isinstance(tree, ast.Module)
        except SyntaxError:
            # Generated code should be valid
            pytest.fail(f"Failed to parse: {code}")
```

### Custom Strategies

```python
from hypothesis import strategies as st

# Strategy for generating import statements
@st.composite
def import_statement(draw):
    module = draw(st.from_regex(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*"))
    return f"import {module}"

# Strategy for class inheritance
@st.composite  
def class_with_inheritance(draw):
    name = draw(valid_identifier("Class"))
    bases = draw(st.lists(valid_identifier(), min_size=0, max_size=3))
    base_str = f"({', '.join(bases)})" if bases else ""
    return f"class {name}{base_str}: pass"
```

## Performance Testing

### Writing Performance Tests

```python
import pytest

@pytest.mark.performance
def test_parser_performance(benchmark, large_python_file):
    """Benchmark parser performance."""
    parser = Parser()
    
    result = benchmark(parser.parse, large_python_file)
    
    # Assert performance requirements
    assert benchmark.stats['mean'] < 0.1  # 100ms mean
    assert benchmark.stats['max'] < 0.2   # 200ms max

@pytest.mark.performance
def test_cache_performance():
    """Test cache provides performance benefit."""
    from tests.helpers import compare_performance
    
    def without_cache(code):
        return ast.parse(code)
    
    def with_cache(code):
        cache = ASTCache()
        if cached := cache.get(code):
            return cached
        result = ast.parse(code)
        cache.set(code, result)
        return result
    
    results = compare_performance(
        {"without_cache": without_cache, "with_cache": with_cache},
        "x = 1 + 2 * 3",
        iterations=1000
    )
    
    # Cache should be faster on repeated calls
    assert results["with_cache"].mean < results["without_cache"].mean
```

### Performance Tracking

```python
from tests.helpers import PerformanceTracker

def test_complete_workflow_performance():
    """Track performance across workflow steps."""
    tracker = PerformanceTracker("AST Analysis Workflow")
    
    # Measure each step
    code = load_large_test_file()
    
    tree = tracker.measure(parse_file, code, name="Parse")
    symbols = tracker.measure(extract_symbols, tree, name="Extract Symbols")
    imports = tracker.measure(analyze_imports, tree, name="Analyze Imports")
    
    # Generate report
    print(tracker.report())
    
    # Assert overall performance
    tracker.assert_all_under(0.1)  # Each step < 100ms
```

## CI/CD Integration

### GitHub Actions Workflow

The test suite runs in multiple stages:

1. **Fast Feedback** - Unit tests only (< 30s)
2. **Integration Tests** - Component interactions (< 2min)
3. **Full Suite** - All tests with coverage (< 5min)
4. **E2E Tests** - Complete workflows (< 10min)
5. **Performance** - Benchmarks on main branch

### Running Tests Locally Like CI

```bash
# Simulate CI environment
export CI=true

# Run fast feedback stage
uv run pytest tests/unit -m unit --maxfail=5

# Run full suite with coverage
uv run pytest tests/ -m "not e2e" --cov --cov-fail-under=80

# Run code quality checks
uv run black --check .
uv run ruff check .
uv run mypy ast_lib tests
```

## Best Practices

### 1. Test Value Over Coverage

```python
# ❌ Bad: Testing for coverage
def test_init():
    obj = MyClass()
    assert obj is not None  # Meaningless

# ✅ Good: Testing behavior
def test_cache_init_creates_empty_cache():
    cache = Cache(max_size=100)
    assert cache.size == 0
    assert cache.max_size == 100
```

### 2. Clear Test Names

```python
# ❌ Bad: Vague names
def test_parser()
def test_error()
def test_works()

# ✅ Good: Descriptive names
def test_parser_handles_unicode_identifiers()
def test_cache_evicts_lru_when_full()
def test_symbol_extraction_includes_nested_classes()
```

### 3. Isolated Tests

```python
# ❌ Bad: Tests depend on each other
class TestCache:
    cache = Cache()  # Shared state!
    
    def test_add_item():
        TestCache.cache.set("key", "value")
    
    def test_get_item():
        # Depends on test_add_item
        assert TestCache.cache.get("key") == "value"

# ✅ Good: Independent tests
class TestCache:
    def test_set_and_get(self):
        cache = Cache()
        cache.set("key", "value")
        assert cache.get("key") == "value"
```

### 4. Meaningful Assertions

```python
# ❌ Bad: Weak assertions
def test_parse_module():
    result = parse("import os")
    assert result  # Just checks truthy

# ✅ Good: Specific assertions
def test_parse_module():
    result = parse("import os")
    assert isinstance(result, ast.Module)
    assert len(result.body) == 1
    assert isinstance(result.body[0], ast.Import)
    assert result.body[0].names[0].name == "os"
```

### 5. Test Edge Cases

```python
def test_parser_edge_cases():
    """Test parser handles edge cases correctly."""
    edge_cases = [
        ("", "empty file"),
        ("# only comments", "comment-only file"),
        ("\\n" * 1000, "many blank lines"),
        ("x = 'unclosed string", "syntax error"),
        ("def f(): pass" * 1000, "many functions"),
    ]
    
    for code, description in edge_cases:
        # Test each edge case
        ...
```

### 6. Performance Awareness

```python
@pytest.mark.unit
def test_fast_operation():
    """Unit tests should be milliseconds."""
    start = time.time()
    result = fast_operation()
    duration = time.time() - start
    
    assert duration < 0.1  # Fail if too slow
    assert result == expected

@pytest.mark.integration
@pytest.mark.timeout(5)  # Fail if takes > 5 seconds
def test_integration_workflow():
    """Integration tests have reasonable timeouts."""
    ...
```

## Dogfooding Notes

As we build the testing infrastructure, we're discovering patterns:

1. **Factory Pattern Works Well** - Creating test data through factories is more flexible than fixtures
2. **Type Hints Help** - Type-annotated test utilities catch errors early
3. **Performance Matters** - Even test utilities need to be fast
4. **Meta-Testing is Valuable** - Testing our test infrastructure ensures reliability

These insights will be incorporated into the main library design.