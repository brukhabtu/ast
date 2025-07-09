#!/usr/bin/env python3
"""
Demonstration of symbol extraction capabilities.

This script shows how to use the AST library to extract and query symbols
from Python source code.
"""

import ast
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ast.symbols import extract_symbols, SymbolType
from ast.symbol_table import SymbolTable, SymbolQuery


def demonstrate_basic_extraction():
    """Demonstrate basic symbol extraction from code."""
    print("=== Basic Symbol Extraction ===")
    
    code = '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self, initial_value: float = 0):
        self.value = initial_value
    
    def add(self, x: float) -> float:
        """Add a value."""
        self.value += x
        return self.value
    
    @property
    def double(self) -> float:
        """Get double the current value."""
        return self.value * 2

def calculate_sum(numbers: list[float]) -> float:
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total

async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    # Simulated async operation
    return {"url": url, "data": "example"}

CONSTANT_PI = 3.14159
_private_var = "secret"
'''
    
    # Parse and extract symbols
    tree = ast.parse(code)
    symbols = extract_symbols(tree)
    
    print(f"Found {len(symbols)} symbols:")
    for symbol in symbols:
        indent = "  " * (1 if symbol.parent else 0)
        print(f"{indent}- {symbol.name} ({symbol.type.value})")
        if symbol.signature:
            print(f"{indent}  Signature: {symbol.signature}")
        if symbol.decorators:
            print(f"{indent}  Decorators: {', '.join(symbol.decorators)}")
    
    return symbols


def demonstrate_symbol_table():
    """Demonstrate symbol table queries."""
    print("\n=== Symbol Table Queries ===")
    
    # Create and populate symbol table
    table = SymbolTable()
    
    # Add symbols from multiple "files"
    for i, module in enumerate(["module1.py", "module2.py", "utils.py"]):
        code = f'''
class Service{i}:
    def process(self, data):
        return data
    
    @staticmethod
    def helper():
        pass

def function{i}():
    pass

async def async_func{i}():
    pass

_private_func{i} = lambda: None
'''
        tree = ast.parse(code)
        symbols = extract_symbols(tree, module)
        table.add_symbols(symbols)
    
    # Demonstrate various queries
    print(f"\nTotal symbols: {len(table)}")
    
    # Find by type
    print("\nClasses:")
    for result in table.find_by_type(SymbolType.CLASS):
        print(f"  - {result.symbol.name} (lookup: {result.lookup_time_ms:.2f}ms)")
    
    # Find by file
    print("\nSymbols in module1.py:")
    for result in table.find_by_file("module1.py"):
        print(f"  - {result.symbol.name}")
    
    # Find async functions
    print("\nAsync functions:")
    for result in table.find_async_symbols():
        print(f"  - {result.symbol.name}")
    
    # Complex query
    print("\nNon-private functions in utils.py:")
    query = SymbolQuery(
        type=SymbolType.FUNCTION,
        file_path="utils.py",
        is_private=False
    )
    for result in table.query(query):
        print(f"  - {result.symbol.name}")
    
    # Show stats
    print("\nSymbol Table Stats:")
    stats = table.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return table


def demonstrate_real_file_analysis():
    """Analyze a real Python file."""
    print("\n=== Real File Analysis ===")
    
    # Analyze this file itself
    this_file = Path(__file__)
    
    with open(this_file) as f:
        code = f.read()
    
    tree = ast.parse(code)
    symbols = extract_symbols(tree, str(this_file))
    
    table = SymbolTable()
    table.add_symbols(symbols)
    
    print(f"Analyzing: {this_file.name}")
    print(f"Total symbols: {len(symbols)}")
    
    # Group by type
    by_type = {}
    for symbol in symbols:
        by_type.setdefault(symbol.type, []).append(symbol)
    
    for sym_type, syms in by_type.items():
        print(f"\n{sym_type.value.title()}s ({len(syms)}):")
        for sym in syms:
            print(f"  - {sym.name}")
            if sym.docstring:
                first_line = sym.docstring.split('\n')[0]
                print(f"    Doc: {first_line}")


def measure_performance():
    """Measure performance of symbol extraction."""
    print("\n=== Performance Measurement ===")
    
    import time
    
    # Generate a large code sample
    lines = []
    for i in range(100):
        lines.extend([
            f"class Class{i}:",
            f"    def method{i}(self, x: int) -> int:",
            f"        return x * {i}",
            "",
            f"def function{i}(arg: str) -> bool:",
            f"    return len(arg) > {i}",
            "",
        ])
    
    large_code = "\n".join(lines)
    
    # Measure parsing time
    start = time.perf_counter()
    tree = ast.parse(large_code)
    parse_time = (time.perf_counter() - start) * 1000
    
    # Measure extraction time
    start = time.perf_counter()
    symbols = extract_symbols(tree)
    extract_time = (time.perf_counter() - start) * 1000
    
    # Measure table creation
    table = SymbolTable()
    start = time.perf_counter()
    table.add_symbols(symbols)
    table_time = (time.perf_counter() - start) * 1000
    
    # Measure lookups
    start = time.perf_counter()
    result = table.find_by_name("Class50", exact=True)
    lookup_time = result[0].lookup_time_ms if result else 0
    
    print(f"Code size: {len(large_code)} characters")
    print(f"Symbols extracted: {len(symbols)}")
    print(f"Parse time: {parse_time:.2f}ms")
    print(f"Extract time: {extract_time:.2f}ms")
    print(f"Table build time: {table_time:.2f}ms")
    print(f"Lookup time: {lookup_time:.2f}ms")
    
    # Verify performance requirements
    print(f"\nPerformance check: {'✓' if lookup_time < 10 else '✗'} Lookup < 10ms")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_basic_extraction()
    demonstrate_symbol_table()
    demonstrate_real_file_analysis()
    measure_performance()