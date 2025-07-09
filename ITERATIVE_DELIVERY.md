# Iterative Delivery Plan - AST Library

## Core Principles
1. **Deliver working vertical slices** - Each iteration provides immediate value
2. **Front-load technical risks** - Tackle hardest problems first
3. **Defer polish** - Let patterns emerge naturally
4. **Quick feedback loops** - Test with real LLM use cases early

## Risk Analysis

### High Risk Areas (Address First)
1. **Performance at Scale** - Can we parse 1000+ files quickly?
2. **Memory Management** - Will large codebases blow up memory?
3. **Error Recovery** - Can we handle malformed/partial code?
4. **Cross-file References** - Tracking symbols across modules

### Medium Risk Areas
1. **Incremental Updates** - Efficient reparsing
2. **Type Inference** - Accuracy without type hints
3. **API Design** - Getting the interface right for LLMs

### Low Risk Areas
1. **Basic AST Operations** - Well-understood problem
2. **Export Formats** - Can iterate later
3. **Documentation** - Can improve over time

## Iteration Plan

### Iteration 0: Spike & Validate (Week 1)
**Goal**: Prove core concept works at scale

**Deliverables**:
- Minimal AST parser that handles 1000+ files
- Basic symbol extraction (functions only)
- Simple CLI: `ast find-function <name>`
- Performance benchmark script

**Success Criteria**:
- Parse large repo (e.g., Django) in <10 seconds
- Memory usage stays under 500MB
- Handles syntax errors without crashing

**Risk Mitigation**:
- Test with real-world messy code early
- Profile memory usage from day 1
- Try multiple AST libraries (ast, tree-sitter)

### Iteration 1: First Useful Tool (Week 2)
**Goal**: LLM can find definitions quickly

**Deliverables**:
- Function & class definition finder
- Cross-file symbol index
- API: `ast.find_definition("symbol_name")`
- Basic caching (in-memory only)

**Success Criteria**:
- LLM can navigate to any definition
- <50ms lookup time
- Works with imports

**Example Use Case**:
```python
# LLM asks: "Where is RequestHandler defined?"
result = ast.find_definition("RequestHandler")
# Returns: {"file": "server.py", "line": 45, "type": "class"}
```

### Iteration 2: Usage Tracking (Week 3)
**Goal**: Find where symbols are used

**Deliverables**:
- Reference finder
- Basic call graph
- API: `ast.find_references("symbol")`
- Distinguish def vs use

**Success Criteria**:
- Find all usages across files
- Track function calls
- Handle method calls

**Risk Mitigation**:
- Start with simple cases (no dynamic dispatch)
- Add complexity incrementally

### Iteration 3: Smart Navigation (Week 4)
**Goal**: Understand code relationships

**Deliverables**:
- Import dependency graph
- Function caller/callee analysis
- API: `ast.get_dependencies("module")`
- Simple pattern matching

**Success Criteria**:
- Trace import chains
- Find unused imports
- Basic dead code detection

### Iteration 4: Performance Critical (Week 5)
**Goal**: Make it production-ready

**Deliverables**:
- Incremental parsing
- Persistent cache (SQLite)
- File watcher integration
- Parallel processing

**Success Criteria**:
- <1s incremental updates
- Cache survives restarts
- 10x speedup with parallelism

**Risk Mitigation**:
- benchmark before optimizing
- Simple cache invalidation first

### Iteration 5: Context Understanding (Week 6)
**Goal**: Extract semantic information

**Deliverables**:
- Docstring extraction
- Type annotation parsing
- Function signatures
- Scope analysis

**Success Criteria**:
- Extract all documentation
- Understand type hints
- Resolve variable scopes

### Iteration 6: Code Intelligence (Week 7)
**Goal**: Provide code insights

**Deliverables**:
- Complexity metrics
- Pattern detection
- Similar code finding
- Basic refactoring suggestions

**Success Criteria**:
- Identify complex functions
- Find duplicate patterns
- Suggest extractions

### Iteration 7: Polish & Package (Week 8)
**Goal**: Production-ready library

**Deliverables**:
- Stable API
- Error handling
- Documentation
- PyPI package

## Anti-Patterns to Avoid

1. **Over-engineering early** - No complex abstractions until patterns emerge
2. **Perfect parsing** - 80% accuracy is fine initially
3. **Feature creep** - Stick to LLM navigation needs
4. **Premature optimization** - Measure first, optimize later

## Validation Strategy

Each iteration includes:
1. **Dogfooding** - Use it to analyze the AST library itself
2. **Real-world testing** - Run against popular Python repos
3. **LLM integration test** - Actually use with an LLM
4. **Performance regression test** - Ensure we don't slow down

## Pivot Points

Be ready to change direction if:
- Performance is fundamentally too slow
- Memory usage is unmanageable
- LLMs need different information than expected
- A better approach emerges (e.g., tree-sitter)

## Success Metrics Per Iteration

| Iteration | Key Metric | Target |
|-----------|------------|--------|
| 0 | Parse time for 1000 files | <10s |
| 1 | Symbol lookup time | <50ms |
| 2 | Reference finding accuracy | >90% |
| 3 | Dependency graph build time | <5s |
| 4 | Incremental update time | <1s |
| 5 | Docstring extraction coverage | >95% |
| 6 | Pattern matching speed | <100ms |
| 7 | API stability | 0 breaking changes |

## Code Structure Evolution

**Iteration 0-2**: Everything in one file, focus on algorithms
**Iteration 3-4**: Extract modules as patterns emerge
**Iteration 5-6**: Introduce interfaces/protocols
**Iteration 7**: Final refactoring and API cleanup

This approach ensures we:
- Always have something working
- Learn from real usage
- Don't over-architect
- Can pivot based on discoveries