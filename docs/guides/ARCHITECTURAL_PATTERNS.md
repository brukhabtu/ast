# Architectural Patterns - AST Library

## Emergent Design Patterns

Through iterative development and dogfooding, several architectural patterns have naturally emerged:

## 1. **Result Object Pattern** 🎯
Instead of throwing exceptions, operations return rich result objects:

```python
# Pattern seen in:
# - ParseResult (parser.py)
# - LookupResult (symbol_table.py)
# - ImportAnalysisResult (imports.py)

@dataclass
class ParseResult:
    tree: Optional[ast.AST] = None
    errors: List[ParseError] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.tree is not None and not self.errors
    
    @property
    def partial_success(self) -> bool:
        return self.tree is not None and bool(self.errors)
```

**Benefits**: Error recovery, partial results, rich error context

## 2. **Visitor Pattern with Context** 🌳
Enhanced visitor that tracks traversal context:

```python
# Pattern in visitor.py
class NodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parent_stack = []
        self.depth = 0
        self.path = []
    
    def visit(self, node):
        # Auto-tracks context during traversal
        self._enter_node(node)
        result = super().visit(node)
        self._exit_node()
        return result
```

**Benefits**: Parent access, depth tracking, path awareness

## 3. **Multi-Index Storage Pattern** 🗂️
Single source of truth with multiple indexes for fast lookup:

```python
# Pattern in symbol_table.py
class SymbolTable:
    def __init__(self):
        # Primary storage
        self._symbols: List[Symbol] = []
        
        # Multiple indexes
        self._by_name: Dict[str, List[Symbol]] = defaultdict(list)
        self._by_type: Dict[SymbolType, List[Symbol]] = defaultdict(list)
        self._by_file: Dict[str, List[Symbol]] = defaultdict(list)
        self._by_qualified_name: Dict[str, Symbol] = {}
```

**Benefits**: O(1) lookups, flexible queries, memory efficient

## 4. **Transparent Caching Layer** 💾
Caching that requires zero code changes:

```python
# Pattern in cache.py integration
# Original function untouched:
def parse_file(filepath: str) -> ParseResult:
    # parsing logic...

# Cache wraps transparently in __init__.py:
from .cache import CacheManager
_cache_manager = CacheManager.get_instance()
parse_file = _cache_manager.cache_parse_file(parse_file)
```

**Benefits**: No API changes, automatic invalidation, configurable

## 5. **Factory Pattern for Test Data** 🏭
Emerged in testing infrastructure:

```python
# Pattern in tests/factories.py
class ASTFactory:
    @staticmethod
    def create_function(name="test_func", **kwargs):
        defaults = {
            "args": ast.arguments(...),
            "body": [ast.Pass()],
            "decorator_list": []
        }
        return ast.FunctionDef(name=name, **{**defaults, **kwargs})
```

**Benefits**: Flexible test data, reduces boilerplate, maintainable

## 6. **Progressive Error Recovery** 🔧
Multiple fallback strategies for parsing errors:

```python
# Pattern in parser.py
class ErrorRecoveringParser:
    def parse(self):
        strategies = [
            self._try_full_parse,
            self._try_statement_by_statement,
            self._try_with_fixes,
            self._extract_partial_ast
        ]
        
        for strategy in strategies:
            result = strategy()
            if result.tree:
                return result
```

**Benefits**: Maximum AST extraction, graceful degradation

## 7. **Dataclass-First Design** 📊
Heavy use of dataclasses for data structures:

```python
@dataclass
class Symbol:
    name: str
    type: SymbolType
    position: Position
    # ... rich metadata
    
@dataclass
class ImportInfo:
    module: str
    names: List[str]
    import_type: ImportType
```

**Benefits**: Type safety, immutability, automatic methods

## 8. **Command Pattern in CLI** 🎮
Extensible command structure:

```python
# Implied pattern in cli.py
commands = {
    'find-function': FindFunctionCommand,
    'find-class': FindClassCommand,  # Easy to add
    'find-imports': FindImportsCommand,
}

def main():
    command = commands[args.command]
    command.execute(args)
```

**Benefits**: Easy to extend, single responsibility

## 9. **Graph as First-Class Concept** 🕸️
Import relationships modeled as proper graph:

```python
# Pattern in import_graph.py
class ImportGraph:
    def __init__(self):
        self.nodes: Dict[str, ImportNode] = {}
        self.edges: List[Tuple[str, str]] = []
    
    def find_circular_imports(self) -> List[CircularImport]:
        # Graph algorithms on code structure
```

**Benefits**: Powerful analysis, standard algorithms apply

## 10. **Performance Measurement Built-In** ⏱️
Not bolted on, but integrated:

```python
# Pattern throughout
with Timer() as t:
    result = expensive_operation()
    
# Automatic in LookupResult
@dataclass
class LookupResult:
    symbol: Symbol
    lookup_time_ms: float  # Always tracked
```

**Benefits**: Performance awareness, easy optimization

## Anti-Patterns Avoided

### ❌ **God Object**
No single class that does everything. Clear separation:
- Parser only parses
- SymbolTable only stores/retrieves
- ImportGraph only handles dependencies

### ❌ **Premature Abstraction**
Started simple, patterns emerged from use:
- No complex class hierarchies
- No unnecessary interfaces
- Concrete implementations first

### ❌ **Tight Coupling**
Modules work independently:
- Parser doesn't know about symbols
- Symbols don't know about caching
- Cache doesn't know about specific operations

## Architectural Principles

1. **Fail Gracefully** - Always return something useful
2. **Performance Matters** - Measure everything
3. **Type Safety** - Dataclasses and type hints everywhere
4. **Composability** - Small tools that work together
5. **Developer Experience** - Easy to use correctly

## Evolution Through Dogfooding

The architecture evolved naturally:
- **Iteration 0**: Basic parsing → Result objects emerged
- **Iteration 1**: Cross-file needs → Caching layer appeared
- **Testing**: Repetitive setup → Factory pattern developed
- **Performance**: Slow lookups → Multi-index pattern

These patterns weren't planned upfront but emerged from real usage and pain points, resulting in a clean, performant architecture.