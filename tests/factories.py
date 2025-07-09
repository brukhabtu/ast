"""
Test data factories for creating AST nodes and Python code patterns.

This module provides flexible factories for generating test data including:
- AST node builders with sensible defaults
- Python code snippet generators
- Complex code pattern builders
- Edge case generators
"""

import ast
import random
import string
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from uuid import uuid4


# --- AST Node Factories ---

class ASTFactory:
    """Factory for creating AST nodes with sensible defaults."""
    
    @staticmethod
    def create_name(id: str = "var", ctx: type[ast.AST] = ast.Load) -> ast.Name:
        """Create an ast.Name node."""
        return ast.Name(id=id, ctx=ctx())
    
    @staticmethod
    def create_constant(value: Any = 42) -> ast.Constant:
        """Create an ast.Constant node."""
        return ast.Constant(value=value)
    
    @staticmethod
    def create_function(
        name: str = "test_func",
        args: Optional[list[str]] = None,
        body: Optional[list[ast.AST]] = None,
        decorators: Optional[list[ast.AST]] = None,
        returns: Optional[ast.AST] = None,
        is_async: bool = False,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Create a function definition node."""
        if args is None:
            args = []
        
        arguments = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=arg, annotation=None) for arg in args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        
        if body is None:
            body = [ast.Pass()]
        
        if decorators is None:
            decorators = []
        
        cls = ast.AsyncFunctionDef if is_async else ast.FunctionDef
        return cls(
            name=name,
            args=arguments,
            body=body,
            decorator_list=decorators,
            returns=returns,
        )
    
    @staticmethod
    def create_class(
        name: str = "TestClass",
        bases: Optional[list[ast.AST]] = None,
        body: Optional[list[ast.AST]] = None,
        decorators: Optional[list[ast.AST]] = None,
    ) -> ast.ClassDef:
        """Create a class definition node."""
        if bases is None:
            bases = []
        
        if body is None:
            body = [ast.Pass()]
        
        if decorators is None:
            decorators = []
        
        return ast.ClassDef(
            name=name,
            bases=bases,
            keywords=[],
            body=body,
            decorator_list=decorators,
        )
    
    @staticmethod
    def create_module(body: Optional[list[ast.AST]] = None) -> ast.Module:
        """Create a module node."""
        if body is None:
            body = []
        return ast.Module(body=body, type_ignores=[])
    
    @staticmethod
    def create_import(
        names: Optional[list[tuple[str, Optional[str]]]] = None
    ) -> ast.Import:
        """Create an import statement."""
        if names is None:
            names = [("os", None)]
        
        return ast.Import(
            names=[ast.alias(name=name, asname=asname) for name, asname in names]
        )
    
    @staticmethod
    def create_import_from(
        module: str = "typing",
        names: Optional[list[tuple[str, Optional[str]]]] = None,
        level: int = 0,
    ) -> ast.ImportFrom:
        """Create a from...import statement."""
        if names is None:
            names = [("Any", None)]
        
        return ast.ImportFrom(
            module=module,
            names=[ast.alias(name=name, asname=asname) for name, asname in names],
            level=level,
        )
    
    @staticmethod
    def create_assign(
        target: str = "x",
        value: Optional[ast.AST] = None,
    ) -> ast.Assign:
        """Create an assignment statement."""
        if value is None:
            value = ASTFactory.create_constant(42)
        
        return ast.Assign(
            targets=[ASTFactory.create_name(target, ast.Store)],
            value=value,
        )
    
    @staticmethod
    def create_if(
        test: Optional[ast.AST] = None,
        body: Optional[list[ast.AST]] = None,
        orelse: Optional[list[ast.AST]] = None,
    ) -> ast.If:
        """Create an if statement."""
        if test is None:
            test = ASTFactory.create_name("condition")
        
        if body is None:
            body = [ast.Pass()]
        
        if orelse is None:
            orelse = []
        
        return ast.If(test=test, body=body, orelse=orelse)


# --- Code Pattern Factories ---

@dataclass
class CodePattern:
    """Container for code patterns with metadata."""
    code: str
    description: str
    python_version: Optional[tuple[int, int]] = None
    tags: list[str] = field(default_factory=list)


class CodeFactory:
    """Factory for generating Python code patterns."""
    
    @staticmethod
    def create_simple_function(
        name: str = "func",
        params: Optional[list[str]] = None,
        return_type: Optional[str] = None,
    ) -> str:
        """Create a simple function definition."""
        if params is None:
            params = ["x", "y"]
        
        param_str = ", ".join(params)
        return_annotation = f" -> {return_type}" if return_type else ""
        
        return f"""
def {name}({param_str}){return_annotation}:
    '''A simple function for testing.'''
    result = {' + '.join(params) if params else '0'}
    return result
""".strip()
    
    @staticmethod
    def create_simple_class(
        name: str = "TestClass",
        base: Optional[str] = None,
        methods: Optional[list[str]] = None,
    ) -> str:
        """Create a simple class definition."""
        if methods is None:
            methods = ["__init__", "method"]
        
        base_str = f"({base})" if base else ""
        
        method_defs = []
        for method in methods:
            if method == "__init__":
                method_defs.append("""
    def __init__(self, value=None):
        self.value = value""")
            else:
                method_defs.append(f"""
    def {method}(self):
        return self.value""")
        
        return f"""
class {name}{base_str}:
    '''A simple class for testing.'''
{''.join(method_defs)}
""".strip()
    
    @staticmethod
    def create_module_with_imports() -> str:
        """Create a module with various import patterns."""
        return """
#!/usr/bin/env python3
'''Test module with various import patterns.'''

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
try:
    import numpy as np
except ImportError:
    np = None

# Relative imports
from . import utils
from ..core import base
from ...lib import helpers

# Constants
VERSION = "1.0.0"
DEBUG = False

# Module-level code
def main():
    print("Module loaded")

if __name__ == "__main__":
    main()
""".strip()
    
    @staticmethod
    def create_async_patterns() -> str:
        """Create code with async/await patterns."""
        return """
import asyncio
from typing import AsyncIterator, Coroutine

async def fetch_data(url: str) -> dict:
    '''Async function example.'''
    await asyncio.sleep(0.1)
    return {"url": url, "data": "example"}

async def process_items(items: list[str]) -> AsyncIterator[str]:
    '''Async generator example.'''
    for item in items:
        await asyncio.sleep(0.01)
        yield f"Processed: {item}"

class AsyncProcessor:
    '''Class with async methods.'''
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def setup(self):
        self.connected = True
    
    async def cleanup(self):
        self.connected = False
    
    async def process(self, data: Any) -> Any:
        async with self:
            return await self._process_internal(data)
    
    async def _process_internal(self, data: Any) -> Any:
        return data

# Async comprehensions
async def example():
    results = [x async for x in process_items(["a", "b", "c"])]
    return results
""".strip()


# --- Edge Case Factories ---

class EdgeCaseFactory:
    """Factory for generating edge case code patterns."""
    
    @staticmethod
    def create_deeply_nested_code(depth: int = 5) -> str:
        """Create deeply nested code structure."""
        indent = "    "
        code_lines = ["def outer():"]
        
        for i in range(depth):
            level_indent = indent * (i + 1)
            if i < depth - 1:
                code_lines.append(f"{level_indent}def level_{i}():")
            else:
                code_lines.append(f"{level_indent}return {i}")
        
        # Add calls back up
        for i in range(depth - 2, -1, -1):
            level_indent = indent * (i + 2)
            code_lines.append(f"{level_indent}return level_{i}()")
        
        code_lines.append(f"{indent}return level_0()")
        return "\n".join(code_lines)
    
    @staticmethod
    def create_unicode_identifiers() -> str:
        """Create code with unicode identifiers (Python 3+)."""
        return """
# Unicode identifiers
def 你好(名字: str) -> str:
    return f"你好, {名字}!"

λ = lambda x: x ** 2
π = 3.14159
Δ = 0.001

class Café:
    def __init__(self, название: str):
        self.название = название
    
    def приветствие(self):
        return f"Welcome to {self.название}"
""".strip()
    
    @staticmethod
    def create_complex_decorators() -> str:
        """Create complex decorator patterns."""
        return """
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def decorator_factory(prefix: str) -> Callable:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            print(f"{prefix}: Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@decorator_factory("DEBUG")
@staticmethod
@property
def complex_method():
    '''Method with multiple decorators.'''
    return "result"

# Stacked decorators with arguments
@decorator_factory("TRACE")
@decorator_factory("LOG")
@decorator_factory("MONITOR")
def heavily_decorated(x: int) -> int:
    return x * 2
""".strip()
    
    @staticmethod
    def create_type_annotation_edge_cases() -> str:
        """Create edge cases for type annotations."""
        return """
from typing import (
    Union, Optional, Literal, TypedDict, Protocol,
    Callable, Generic, TypeVar, ParamSpec, Concatenate,
    overload, final, Final
)
from collections.abc import Sequence, Mapping
import sys

# Complex union types
ComplexType = Union[int, str, list[dict[str, Union[int, str]]]]

# Literal types
Mode = Literal["r", "w", "rb", "wb"]

# TypedDict
class UserDict(TypedDict, total=False):
    name: str
    age: int
    email: Optional[str]

# Protocol
class Drawable(Protocol):
    def draw(self) -> None: ...

# Generic with constraints
T = TypeVar('T', bound=Drawable)
P = ParamSpec('P')

class Container(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item

# Complex callable types
Handler = Callable[Concatenate[str, P], Union[int, None]]

# Conditional types
if sys.version_info >= (3, 10):
    Value = int | str | None
else:
    Value = Union[int, str, None]

# Forward references
def process(data: 'ComplexType') -> 'UserDict':
    return UserDict(name=str(data))

# Overloaded functions
@overload
def process_value(x: int) -> str: ...

@overload
def process_value(x: str) -> int: ...

def process_value(x: Union[int, str]) -> Union[str, int]:
    if isinstance(x, int):
        return str(x)
    return len(x)
""".strip()


# --- Test Data Generator ---

class TestDataGenerator:
    """Generate collections of test data for comprehensive testing."""
    
    @staticmethod
    def generate_random_identifier(prefix: str = "var") -> str:
        """Generate a random valid Python identifier."""
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{prefix}_{suffix}"
    
    @staticmethod
    def generate_test_suite() -> dict[str, CodePattern]:
        """Generate a comprehensive test suite of code patterns."""
        return {
            "simple_function": CodePattern(
                code=CodeFactory.create_simple_function(),
                description="Basic function definition",
                tags=["function", "basic"],
            ),
            "simple_class": CodePattern(
                code=CodeFactory.create_simple_class(),
                description="Basic class definition",
                tags=["class", "basic"],
            ),
            "module_imports": CodePattern(
                code=CodeFactory.create_module_with_imports(),
                description="Module with various import patterns",
                tags=["import", "module"],
            ),
            "async_patterns": CodePattern(
                code=CodeFactory.create_async_patterns(),
                description="Async/await patterns",
                python_version=(3, 5),
                tags=["async", "coroutine"],
            ),
            "nested_code": CodePattern(
                code=EdgeCaseFactory.create_deeply_nested_code(),
                description="Deeply nested functions",
                tags=["edge_case", "nested"],
            ),
            "unicode_identifiers": CodePattern(
                code=EdgeCaseFactory.create_unicode_identifiers(),
                description="Unicode identifiers",
                python_version=(3, 0),
                tags=["edge_case", "unicode"],
            ),
            "complex_decorators": CodePattern(
                code=EdgeCaseFactory.create_complex_decorators(),
                description="Complex decorator patterns",
                tags=["decorator", "advanced"],
            ),
            "type_annotations": CodePattern(
                code=EdgeCaseFactory.create_type_annotation_edge_cases(),
                description="Type annotation edge cases",
                python_version=(3, 8),
                tags=["typing", "edge_case"],
            ),
        }
    
    @staticmethod
    def generate_syntax_errors() -> list[tuple[str, str]]:
        """Generate code with syntax errors for error handling tests."""
        return [
            ("def func(:", "Incomplete function definition"),
            ("class Test\n    pass", "Missing colon in class"),
            ("import from module", "Invalid import syntax"),
            ("x = 1 2 3", "Invalid assignment"),
            ("if True\n    pass", "Missing colon in if"),
            ("for in range(10):", "Missing variable in for loop"),
            ("x = (1, 2, 3]", "Mismatched brackets"),
            ("'''unclosed string", "Unclosed string literal"),
            ("def func():\nreturn", "Invalid indentation"),
            ("lambda x, x: x", "Duplicate parameter"),
        ]