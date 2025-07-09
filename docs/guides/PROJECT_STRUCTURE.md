# AST Library Project Structure

```
ast/                              # Root directory
│
├── astlib/                       # Main library package
│   ├── __init__.py              # Package initialization
│   ├── api.py                   # Unified API (ASTLib class)
│   ├── benchmark.py             # Performance benchmarking utilities
│   ├── cache.py                 # Caching layer for performance
│   ├── call_graph.py            # Call graph analysis
│   ├── cli.py                   # Command-line interface
│   ├── finder.py                # Function finding utilities
│   ├── import_graph.py          # Import dependency graph
│   ├── imports.py               # Import analysis
│   ├── indexer.py               # Project-wide symbol indexing
│   ├── parser.py                # AST parsing with error recovery
│   ├── references.py            # Reference finding
│   ├── symbol_table.py          # Symbol storage and lookup
│   ├── symbols.py               # Symbol extraction
│   ├── types.py                 # Type definitions
│   ├── visitor.py               # AST visitor utilities
│   └── walker.py                # Directory walking
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration and fixtures
│   ├── factories.py             # Test data factories
│   ├── test_infrastructure.py   # Testing utilities
│   │
│   ├── unit/                    # Unit tests
│   │   ├── test_api.py
│   │   ├── test_benchmark.py
│   │   ├── test_cache.py
│   │   ├── test_call_graph.py
│   │   ├── test_cli.py
│   │   ├── test_edge_cases.py
│   │   ├── test_example.py
│   │   ├── test_finder.py
│   │   ├── test_imports.py
│   │   ├── test_indexer.py
│   │   ├── test_parser.py
│   │   ├── test_references.py
│   │   ├── test_symbols.py
│   │   ├── test_symbol_table.py
│   │   ├── test_visitor.py
│   │   └── test_walker.py
│   │
│   ├── integration/             # Integration tests
│   │   ├── test_benchmark_with_repos.py
│   │   ├── test_cache_performance.py
│   │   ├── test_cross_file_navigation.py
│   │   ├── test_import_resolution.py
│   │   ├── test_indexer_integration.py
│   │   ├── test_parser_symbols.py
│   │   └── test_performance.py
│   │
│   ├── e2e/                     # End-to-end tests
│   │   └── test_cli_commands.py
│   │
│   ├── fixtures/                # Test fixtures and data
│   │   ├── mock_repos.py
│   │   ├── generator.py
│   │   └── test_projects/       # Sample project structures
│   │       ├── broken_imports/
│   │       ├── circular_imports/
│   │       ├── django_like/
│   │       └── large_project/
│   │
│   ├── helpers/                 # Test helper utilities
│   │   ├── ast_helpers.py
│   │   ├── code_helpers.py
│   │   ├── hypothesis_strategies.py
│   │   └── performance_helpers.py
│   │
│   └── test_code/               # Sample code for testing
│       ├── common_patterns/
│       └── edge_cases/
│
├── validation/                  # Validation and analysis tools
│   ├── analyze_results.py       # Test result analysis
│   ├── dogfood_tracker.py       # Dogfooding tracking
│   ├── generate_regression_tests.py
│   ├── test_harness.py          # Test execution harness
│   └── validate.py              # Validation framework
│
├── examples/                    # Example usage
│   └── demo_symbol_extraction.py
│
├── .github/                     # GitHub configuration
│   └── workflows/               # CI/CD workflows
│
├── demo_output/                 # Demo output files
├── cache_analysis/              # Cache analysis results
├── cache_benchmarks/            # Benchmark results
│
├── Configuration Files:
├── .gitignore                   # Git ignore patterns
├── .pylintrc                    # Pylint configuration
├── pyproject.toml              # Project configuration
├── pytest.ini                   # Pytest configuration
├── README.md                    # Project documentation
├── CLAUDE.md                    # Claude-specific instructions
├── ITERATIVE_PLAN.md           # Development plan
├── DOGFOODING_STRATEGY.md      # Dogfooding approach
├── ITERATION_0_SUMMARY.md      # Iteration 0 results
├── INVENTORY.md                # Tools inventory
├── ARCHITECTURE.md             # Architecture documentation
├── SKIP_TEST_ANALYSIS.md       # Test analysis
├── TEST_FIXES_SUMMARY.md       # Test fixes documentation
├── PROJECT_STRUCTURE.md        # This file
│
└── Scripts/Demos:
    ├── analyze_test_quality.py  # Test quality analysis
    ├── create_baseline.py       # Performance baseline
    ├── demo_benchmark.py        # Benchmarking demo
    ├── demo_test_value.py       # Test value demo
    ├── dogfood_demo.py          # Dogfooding demo
    ├── download_test_repos.py   # Repository downloader
    ├── example.py               # Basic usage example
    ├── find_skipped_tests.py    # Skip test finder
    ├── run_benchmarks.py        # Benchmark runner
    ├── test_call_graph_demo.py  # Call graph demo
    ├── test_dogfood.py          # Dogfooding test
    ├── test_reference_finding.py # Reference finding test
    └── test_unified_api.py      # API test
```

## Key Directories

### 1. **astlib/** - Core Library
The main package containing all the AST analysis functionality:
- **Core modules**: parser, symbols, visitor, walker
- **Analysis tools**: imports, references, call_graph
- **Infrastructure**: cache, indexer, symbol_table
- **Interface**: api, cli

### 2. **tests/** - Comprehensive Test Suite
Organized by test level:
- **unit/**: Tests for individual components
- **integration/**: Tests for component interactions
- **e2e/**: End-to-end CLI tests
- **fixtures/**: Test data and mock projects
- **helpers/**: Testing utilities

### 3. **validation/** - Quality Assurance
Tools for validating and analyzing the library:
- Performance analysis
- Test quality metrics
- Dogfooding tracking
- Regression testing

## Architecture Layers

1. **Parse Layer**: `parser.py`
2. **Extraction Layer**: `symbols.py`, `imports.py`, `references.py`
3. **Index Layer**: `symbol_table.py`, `indexer.py`, `import_graph.py`
4. **Analysis Layer**: `call_graph.py`, `finder.py`
5. **Interface Layer**: `api.py`, `cli.py`

## Test Organization

- **340+ tests** across unit, integration, and E2E
- **78% unit tests**, 17% integration, 5% E2E
- Comprehensive fixtures for different project types
- Property-based testing with Hypothesis
- Performance benchmarking framework