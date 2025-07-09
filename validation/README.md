# AST Library Validation System

This directory contains the comprehensive validation system for testing the AST library against real-world Python codebases.

## Overview

The validation system tests the AST library on popular Python repositories to:
- Identify compatibility issues across Python versions
- Discover edge cases and failure patterns
- Measure performance at scale
- Generate regression tests from real failures
- Track library effectiveness in production code

## Quick Start

### Run Basic Validation

```bash
# Quick validation on 3 repositories
python validate.py --quick

# Full validation on all repositories
python validate.py

# Test specific repositories
python validate.py --repos https://github.com/psf/requests.git https://github.com/django/django.git
```

### Analyze Results

```bash
# Generate insights and update reports
python analyze_results.py

# Generate regression tests from failures
python generate_regression_tests.py
```

### Run Regression Tests

```bash
# Run all generated regression tests
python regression_tests/run_regression_tests.py
```

## Repository Test Set

The validation suite tests against these popular Python projects:

### Core Python Ecosystem
- **requests** - HTTP library (51k+ stars)
- **django** - Web framework (77k+ stars)
- **flask** - Micro web framework (66k+ stars)
- **numpy** - Numerical computing (26k+ stars)
- **pandas** - Data analysis (42k+ stars)

### Modern Python Projects
- **fastapi** - Modern web API framework (72k+ stars)
- **black** - Code formatter (37k+ stars)
- **poetry** - Dependency management (29k+ stars)
- **httpx** - Async HTTP client (12k+ stars)
- **pydantic** - Data validation (19k+ stars)

### Large-Scale Projects
- **home-assistant** - Home automation (69k+ stars)
- **scikit-learn** - Machine learning (58k+ stars)
- **transformers** - NLP models (125k+ stars)

## Output Structure

```
validation/
├── validate.py                    # Main validation script
├── analyze_results.py            # Results analysis tool
├── generate_regression_tests.py  # Regression test generator
├── README.md                     # This file
│
├── validation_results.json       # Raw validation results
├── compatibility_matrix.json     # Python version compatibility
├── statistics_report.json        # Performance statistics
├── validation_report.md          # Human-readable report
├── validation_insights.json      # Analysis insights
├── stats_report.md              # Detailed statistics
│
├── failure_cases/               # Detailed failure examples
│   ├── requests/
│   │   ├── failures_summary.json
│   │   ├── example_1.md
│   │   └── ...
│   └── django/
│       └── ...
│
└── regression_tests/            # Generated regression tests
    ├── test_regression_walrus_operator.py
    ├── test_regression_pattern_matching.py
    ├── test_edge_cases.py
    ├── test_performance_regression.py
    └── run_regression_tests.py
```

## Validation Process

### 1. Repository Cloning
- Shallow clone (depth=1) for speed
- Temporary directory management
- Automatic cleanup after validation

### 2. File Discovery
- Finds all Python files (excluding tests initially)
- Tracks file sizes and counts
- Detects Python version requirements

### 3. AST Parsing
- Attempts to parse each file
- Records success/failure
- Captures error details and code snippets

### 4. Performance Measurement
- Times parsing operations
- Calculates throughput (files/second)
- Monitors memory usage

### 5. Failure Analysis
- Categorizes errors by type
- Saves failure examples
- Extracts problematic code snippets

### 6. Report Generation
- Updates compatibility matrix
- Generates statistics report
- Creates regression tests
- Produces insights and recommendations

## Metrics Collected

### Success Metrics
- Parse success rate per repository
- Overall success rate across all files
- Success rate by Python version
- Success rate by file size

### Performance Metrics
- Average parse time per file
- Parse time percentiles (median, 95th, 99th)
- Files per second throughput
- Memory usage statistics

### Error Analysis
- Error type distribution
- Most common failure patterns
- Python version specific issues
- Repository-specific challenges

### Feature Usage
- Language features encountered
- Modern Python feature adoption
- Syntax patterns distribution

## Failure Case Format

Each failure is documented with:
```json
{
  "repo_name": "requests",
  "file_path": "/path/to/file.py",
  "python_version": "3.11",
  "error_type": "SyntaxError",
  "error_message": "invalid syntax",
  "code_snippet": ">>> 42: if (x := expensive_func()):",
  "line_number": 42,
  "traceback": "full traceback..."
}
```

## Regression Test Generation

The system automatically generates regression tests from failures:

1. **Pattern-based tests** - Groups similar failures
2. **Edge case tests** - Covers discovered edge cases
3. **Performance tests** - Ensures parsing speed
4. **Version-specific tests** - Python version compatibility

## Customization

### Adding New Repositories

Edit `TEST_REPOSITORIES` in `validate.py`:
```python
TEST_REPOSITORIES = [
    {"name": "myproject", "url": "https://github.com/user/myproject.git", "stars": 1000},
    # ... existing repos
]
```

### Adjusting Validation Parameters

- Timeout settings
- Memory limits
- File size limits
- Exclusion patterns

## Integration with CI/CD

The validation system can be integrated into CI pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run AST Validation
  run: |
    python validation/validate.py --quick
    python validation/analyze_results.py
    
- name: Check Success Rate
  run: |
    python -c "
    import json
    with open('validation/statistics_report.json') as f:
        stats = json.load(f)
    success_rate = stats.get('overall_success_rate', 0)
    if success_rate < 0.95:
        exit(1)
    "
```

## Troubleshooting

### Common Issues

1. **Repository clone failures**
   - Check network connectivity
   - Verify git is installed
   - Check disk space

2. **Memory errors**
   - Reduce parallel processing
   - Skip very large files
   - Increase system memory

3. **Import errors**
   - Ensure AST library is in PYTHONPATH
   - Check Python version compatibility

### Debug Mode

Run with verbose logging:
```bash
python validate.py --verbose --repos https://github.com/psf/requests.git
```

## Contributing

When adding new validation features:

1. Update test repository list if needed
2. Add new metrics to collection
3. Update report generation
4. Document new failure patterns
5. Generate appropriate regression tests

## Future Enhancements

- [ ] Multi-Python version testing with tox
- [ ] Docker-based isolation for repositories
- [ ] Parallel repository processing
- [ ] Real-time progress dashboard
- [ ] Comparative analysis between versions
- [ ] Integration with code coverage tools