# Project Manifest

## Directory Structure

```
ast/                        # Project root
├── astlib/                 # Main library package
├── tests/                  # Test suite
├── docs/                   # Documentation
│   ├── guides/            # User guides and technical docs
│   ├── planning/          # Project planning documents
│   └── api/               # API documentation (future)
├── scripts/               # Utility scripts
├── benchmarks/            # Performance benchmarks
├── examples/              # Example usage
├── validation/            # Validation scripts
└── .github/               # GitHub Actions workflows
```

## Key Files

- `README.md` - Project overview and quick start
- `pyproject.toml` - Project configuration and dependencies
- `pytest.ini` - Test configuration
- `Dockerfile` - Container configuration
- `.gitignore` - Git ignore rules
- `.dockerignore` - Docker ignore rules

## Development Directories

- `test_repos/` - Test repositories (git-ignored)
- `cache_analysis/` - Cache analysis results (git-ignored)
- `cache_benchmarks/` - Benchmark results (git-ignored)
- `.hypothesis/` - Property-based testing data
- `.pytest_cache/` - Pytest cache
- `ast_cli.egg-info/` - Package metadata

## CI/CD

All GitHub Actions workflows are in `.github/workflows/`:
- `01-build-docker.yml` - Docker image building
- `02-test-docker.yml` - Test execution in Docker
- `claude-code-agent.yml` - AI assistant integration
- `claude-code-review.yml` - AI code review