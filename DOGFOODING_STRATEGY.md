# Dogfooding Strategy - AST Tools Building AST Tools

## Core Principle
Each iteration's tools must be immediately used by agents in subsequent iterations to navigate and develop the AST library itself. If a tool isn't helpful for building the AST library, it won't be helpful for other codebases.

## Progressive Tool Adoption

### Iteration 0 Output → Used by Iteration 1
**Tools Available:**
- `ast find-function <name>` - Find function definitions
- Basic symbol extraction
- Performance benchmarks

**Dogfooding in Iteration 1:**
```bash
# Agent 1 uses to understand existing code structure
ast find-function parse_file
ast find-function extract_symbols

# Agent 2 uses to find caching opportunities
ast find-function --list-all | grep -E "(parse|extract)" 

# Agent 3 uses to analyze imports in AST codebase
ast find-function analyze_imports
```

### Iteration 1 Output → Used by Iteration 2
**New Tools Available:**
- `ast.find_definition("symbol")` - Cross-file definition lookup
- Import resolution
- In-memory caching

**Dogfooding in Iteration 2:**
```python
# Agents use API to navigate AST codebase
import ast_lib

# Find where Parser class is defined
parser_def = ast_lib.find_definition("Parser")
print(f"Parser defined at {parser_def.file}:{parser_def.line}")

# Find all imports of parser module
parser_imports = ast_lib.find_imports("parser")
```

### Iteration 2 Output → Used by Iteration 3
**New Tools Available:**
- `ast.find_references("symbol")` - Find all usages
- Call graph analysis
- Scope resolution

**Dogfooding in Iteration 3:**
```python
# Analyze AST library's own architecture
ast_analyzer = ast_lib.ASTLib(".")

# Find all calls to parse_file
parse_calls = ast_analyzer.find_references("parse_file")
print(f"parse_file called {len(parse_calls)} times")

# Analyze call relationships
call_graph = ast_analyzer.build_call_graph()
critical_functions = call_graph.most_called_functions(10)
```

### Iteration 3 Output → Used by Iteration 4
**New Tools Available:**
- Dependency graphs
- Dead code detection
- Pattern matching

**Dogfooding in Iteration 4:**
```python
# Find optimization opportunities in AST library
optimizer = ast_lib.CodeOptimizer(".")

# Find unused code in AST library
dead_code = optimizer.find_dead_code()
print(f"Found {len(dead_code)} unused functions")

# Find repeated patterns that could be refactored
patterns = optimizer.find_patterns("for _ in ast.walk(node)")
```

## Dogfooding Requirements per Agent

### Agent Task Template
```
Agent X: [Component Name]
Primary Task: [Build component]

Dogfooding Requirements:
1. Must use previous iteration's tools to understand relevant code
2. Document which AST tools were helpful/not helpful
3. Suggest improvements based on actual usage
4. Create test cases from real usage patterns

Example Usage:
- Use 'ast find-function' to locate similar implementations
- Use find_references() to understand usage patterns
- Use dependency graph to avoid circular imports
```

## Validation Metrics

### Tool Usefulness Score
Track how often each tool is actually used by agents:
- High use = valuable tool, keep and enhance
- Low use = questionable value, consider removing
- No use = remove from library

### Productivity Metrics
- Time to find relevant code: before vs after tool
- Number of files manually read: before vs after
- Accuracy of code understanding: with vs without tools

## Real Usage Examples

### Building Parser Enhancement
```python
# Without AST tools: Agent reads 10+ files to understand parser
# With AST tools:
ast_lib = ASTLib(".")

# Instantly find all parser-related code
parser_refs = ast_lib.find_references("Parser")
visitor_impls = ast_lib.find_pattern("def visit_*")

# Understand parser architecture in seconds
parser_deps = ast_lib.get_dependencies("parser.py")
```

### Adding New Feature
```python
# Agent adding new symbol type needs to:
# 1. Find where symbols are defined
symbol_def = ast_lib.find_definition("Symbol")

# 2. Find all symbol extraction points
extract_refs = ast_lib.find_references("extract_symbols")

# 3. Find test patterns
symbol_tests = ast_lib.find_pattern("test_*symbol*")
```

## Feedback Loop Process

1. **Build** - Create tool in iteration N
2. **Use** - Apply tool in iteration N+1
3. **Measure** - Track actual usage and time saved
4. **Improve** - Enhance based on real pain points
5. **Repeat** - Continue cycle

## Anti-Patterns to Avoid

### ❌ Building Without Using
- Creating tools speculatively
- Not validating with real usage
- Assuming what would be helpful

### ❌ Forced Usage
- Using tools just to check a box
- Not reporting when tools are unhelpful
- Continuing to build unused features

### ✅ Honest Feedback
- Report when manual exploration is faster
- Suggest what would actually help
- Remove features that don't provide value

## Success Criteria

1. **Iteration 1 agents** report 50%+ time savings using Iteration 0 tools
2. **Each tool** is used at least 5 times by subsequent agents
3. **Agents prefer** AST tools over manual file reading for navigation
4. **New features** are suggested based on actual pain points
5. **Final library** only includes battle-tested, proven-useful tools

## Dogfooding Reports

Each agent must include in their output:
```markdown
## Dogfooding Report
- Tools used: [list of AST tools used]
- Time saved: [estimated hours saved]
- Pain points: [what was still hard]
- Missing features: [what tools would have helped]
- Effectiveness rating: [1-10 for each tool used]
```

This ensures we build tools that LLMs actually need and use, not what we imagine they might need.