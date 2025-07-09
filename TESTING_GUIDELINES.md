# AST Library Testing Guidelines

## Core Principles

1. **Test Behavior, Not Implementation** - Test what the code does, not how it does it
2. **Value Over Coverage** - 80% meaningful coverage > 100% trivial coverage
3. **Fast Feedback** - Unit tests must run in milliseconds
4. **Real-World Validation** - Test with actual Python code patterns

## Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── e2e/           # Full workflow tests
├── fixtures/      # Shared test data
├── helpers/       # Test utilities
└── conftest.py    # Pytest configuration
```

## Required Test Types per Component

### Parser Module
- **Unit**: Parse individual Python constructs (functions, classes, etc.)
- **Edge Cases**: Syntax errors, incomplete code, Python 3.8-3.12 features
- **Property**: Random valid Python code generation with Hypothesis
- **Integration**: Parse complete files from test_repos/

### Symbol Extraction
- **Unit**: Extract symbols from AST nodes
- **Edge Cases**: Nested definitions, decorators, metaclasses
- **Integration**: Symbol table consistency across files
- **Performance**: Symbol lookup under 10ms

### Import Analysis  
- **Unit**: Parse different import styles
- **Edge Cases**: Circular imports, missing modules, dynamic imports
- **Integration**: Resolve imports in real package structures
- **Graph Tests**: Validate dependency relationships

### Caching Layer
- **Unit**: Cache operations (get, set, invalidate)
- **Concurrency**: Thread-safe operations
- **Performance**: Demonstrate 10x speedup
- **Memory**: LRU eviction under memory pressure

## Test Data Management

### Use Factories, Not Fixtures
```python
# Good - Flexible factory
def create_ast_function(name="test_func", args=None, body=None):
    return ast.FunctionDef(
        name=name,
        args=args or ast.arguments(...),
        body=body or [ast.Pass()],
        ...
    )

# Bad - Rigid fixture
@pytest.fixture
def sample_function():
    return ast.FunctionDef(name="fixed_name", ...)
```

### Real Code Examples
Keep a `test_code/` directory with real Python patterns:
- `edge_cases/` - Unusual but valid Python
- `common_patterns/` - Typical code structures  
- `large_files/` - Performance test cases

## Performance Testing

Every module must have performance benchmarks:
```python
def test_parser_performance(benchmark):
    large_file = read_test_file("large_module.py")
    result = benchmark(parse_file, large_file)
    assert benchmark.stats['mean'] < 0.1  # 100ms max
```

## Anti-Patterns to Avoid

### ❌ Testing Private Methods
```python
# Bad
def test_parser_internal_state():
    parser = Parser()
    assert parser._internal_cache == {}  # Don't test internals
```

### ❌ Meaningless Coverage
```python
# Bad - Tests nothing useful
def test_init():
    parser = Parser()
    assert parser is not None  # Worthless test
```

### ❌ Overly Coupled Tests
```python
# Bad - Too many dependencies
def test_everything():
    # Tests parser + symbols + cache + API all together
    # If it fails, unclear what broke
```

### ✅ Focused, Valuable Tests
```python
# Good - Clear purpose, isolated
def test_parse_async_function():
    code = "async def fetch(): await api.get()"
    tree = parse_file(code)
    funcs = extract_functions(tree)
    assert funcs[0].name == "fetch"
    assert funcs[0].is_async is True
```

## Integration Test Strategy

Integration tests should test real workflows:
```python
def test_find_definition_across_files():
    # Setup a multi-file project
    project = create_test_project({
        "main.py": "from utils import helper\nhelper()",
        "utils.py": "def helper(): pass"
    })
    
    # Test cross-file lookup
    ast_lib = ASTLib(project.path)
    result = ast_lib.find_definition("helper", "main.py")
    
    assert result.file == "utils.py"
    assert result.line == 1
```

## Continuous Integration

### Test Stages
1. **Fast Feedback** (< 30s): Unit tests only
2. **Full Validation** (< 2min): Unit + Integration  
3. **Release Gate** (< 10min): All tests + large repo validation

### Parallel Execution
```bash
# Run tests in parallel by type
pytest tests/unit -n auto
pytest tests/integration -n 4
pytest tests/e2e  # Run serially
```

## Test Review Checklist

Before merging, ensure tests:
- [ ] Run in < 100ms (unit) or < 5s (integration)
- [ ] Test actual behavior, not implementation
- [ ] Include edge cases and error conditions
- [ ] Have clear, descriptive names
- [ ] Provide value beyond coverage metrics
- [ ] Work in isolation (no order dependencies)
- [ ] Use property-based testing where applicable