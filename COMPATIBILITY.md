# AST Library Compatibility Matrix

This document tracks the compatibility of the AST library across different Python versions, repositories, and features.

## Python Version Support

| Python Version | Status | Test Coverage | Known Issues |
|----------------|--------|---------------|--------------|
| 3.8 | 🟢 Supported | 0% | None |
| 3.9 | 🟢 Supported | 0% | None |
| 3.10 | 🟢 Supported | 0% | None |
| 3.11 | 🟢 Supported | 0% | None |
| 3.12 | 🟢 Supported | 0% | None |
| 3.13 | 🔴 Not Tested | 0% | Not tested |

## Feature Compatibility

### Core Features

| Feature | 3.8 | 3.9 | 3.10 | 3.11 | 3.12 | Notes |
|---------|-----|-----|------|------|------|-------|
| Basic AST Parsing | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending implementation |
| Symbol Extraction | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending implementation |
| Import Analysis | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending implementation |
| Type Hints | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending implementation |
| Async/Await | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending implementation |
| Pattern Matching | ❌ | ❌ | ⏳ | ⏳ | ⏳ | Python 3.10+ feature |
| Exception Groups | ❌ | ❌ | ❌ | ⏳ | ⏳ | Python 3.11+ feature |

### Advanced Features

| Feature | Status | Python Versions | Notes |
|---------|--------|-----------------|-------|
| Walrus Operator (:=) | ⏳ | 3.8+ | Pending implementation |
| Positional-only params | ⏳ | 3.8+ | Pending implementation |
| TypedDict | ⏳ | 3.8+ | Pending implementation |
| Literal Types | ⏳ | 3.8+ | Pending implementation |
| TypeAlias | ⏳ | 3.10+ | Pending implementation |
| Match Statements | ⏳ | 3.10+ | Pending implementation |
| Exception Notes | ⏳ | 3.11+ | Pending implementation |
| Type Parameter Syntax | ⏳ | 3.12+ | Pending implementation |

## Repository Test Results

### Validation Summary

| Repository | Files | Success Rate | Parse Time | Last Tested |
|------------|-------|--------------|------------|-------------|
| requests | - | - | - | Not tested |
| django | - | - | - | Not tested |
| flask | - | - | - | Not tested |
| numpy | - | - | - | Not tested |
| pandas | - | - | - | Not tested |
| fastapi | - | - | - | Not tested |
| black | - | - | - | Not tested |
| poetry | - | - | - | Not tested |
| httpx | - | - | - | Not tested |
| pydantic | - | - | - | Not tested |
| home-assistant | - | - | - | Not tested |
| scikit-learn | - | - | - | Not tested |
| transformers | - | - | - | Not tested |

## Known Limitations

### Syntax Support
- **Pattern Matching**: Requires Python 3.10+
- **Exception Groups**: Requires Python 3.11+
- **Type Parameters**: Requires Python 3.12+
- **f-string improvements**: Version-specific features

### Performance Characteristics
- **Small Codebases** (< 100 files): Target < 1s parsing
- **Medium Codebases** (100-1000 files): Target < 5s parsing
- **Large Codebases** (1000+ files): Target < 30s parsing

### Edge Cases
- Very long lines (> 10,000 characters)
- Deeply nested structures (> 100 levels)
- Circular imports
- Dynamic code generation
- Non-UTF8 encoded files

## Testing Methodology

1. **Repository Selection**: Popular Python projects with diverse coding styles
2. **Version Testing**: Each repository tested against supported Python versions
3. **Feature Coverage**: Specific language features tested in isolation
4. **Performance Benchmarking**: Timed parsing of entire repositories
5. **Error Tracking**: All failures logged with reproducible examples

## Updating This Document

This document is automatically updated by the validation suite. Manual updates should only add notes or clarifications. Run `python validation/validate.py` to refresh compatibility data.

Legend:
- 🟢 Fully Supported
- 🟡 Partial Support
- 🔴 Not Supported
- ⏳ Pending Implementation
- ❌ Not Applicable