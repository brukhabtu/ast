"""
Code generation and file handling test helpers.
"""

import ast
import tempfile
from pathlib import Path
from typing import Optional, Union


def create_temp_python_file(
    content: str,
    prefix: str = "test_",
    suffix: str = ".py",
    dir: Optional[Path] = None
) -> Path:
    """
    Create a temporary Python file with the given content.
    
    Args:
        content: Python code content
        prefix: File name prefix
        suffix: File name suffix
        dir: Directory to create file in (None for system temp)
    
    Returns:
        Path to the created file
    """
    with tempfile.NamedTemporaryFile(
        mode='w',
        prefix=prefix,
        suffix=suffix,
        dir=dir,
        delete=False
    ) as f:
        f.write(content)
        return Path(f.name)


def generate_test_module(
    name: str = "test_module",
    imports: Optional[list[str]] = None,
    functions: Optional[list[str]] = None,
    classes: Optional[list[str]] = None,
) -> str:
    """
    Generate a complete Python module for testing.
    
    Args:
        name: Module name (used in docstring)
        imports: List of import statements
        functions: List of function definitions
        classes: List of class definitions
    
    Returns:
        Complete module source code
    """
    parts = [f'"""Test module: {name}"""']
    
    if imports:
        parts.append("")
        parts.extend(imports)
    
    if functions:
        parts.append("")
        parts.extend(functions)
    
    if classes:
        parts.append("")
        parts.extend(classes)
    
    # Add a main block
    parts.extend([
        "",
        "if __name__ == '__main__':",
        "    print(f'Running {__name__}')",
    ])
    
    return "\n".join(parts)


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python syntax without executing.
    
    Args:
        code: Python code to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def load_test_case(path: Union[str, Path]) -> str:
    """
    Load a test case from a file.
    
    Args:
        path: Path to the test case file
    
    Returns:
        File contents
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test case not found: {path}")
    
    return path.read_text()


def create_test_package(
    root_dir: Path,
    package_name: str,
    modules: dict[str, str],
    create_init: bool = True
) -> Path:
    """
    Create a test package structure.
    
    Args:
        root_dir: Root directory to create package in
        package_name: Name of the package
        modules: Dict of module_name -> module_content
        create_init: Whether to create __init__.py files
    
    Returns:
        Path to the package directory
    """
    package_dir = root_dir / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    
    if create_init:
        (package_dir / "__init__.py").write_text(
            f'"""Test package: {package_name}"""'
        )
    
    for module_name, content in modules.items():
        module_path = package_dir / f"{module_name}.py"
        module_path.write_text(content)
    
    return package_dir


def generate_import_test_cases() -> dict[str, str]:
    """
    Generate various import test cases.
    
    Returns:
        Dict of test_name -> code
    """
    return {
        "simple_import": "import os",
        "aliased_import": "import os as operating_system",
        "from_import": "from os import path",
        "from_import_aliased": "from os import path as p",
        "multiple_imports": "import os, sys, json",
        "multiple_from_imports": "from os import path, environ, getcwd",
        "relative_import": "from . import module",
        "relative_parent_import": "from .. import parent_module",
        "wildcard_import": "from os import *",
        "nested_import": "from os.path import join, exists",
        "try_except_import": """
try:
    import optional_module
except ImportError:
    optional_module = None
""",
        "conditional_import": """
import sys
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated
""",
    }


def generate_function_test_cases() -> dict[str, str]:
    """
    Generate various function definition test cases.
    
    Returns:
        Dict of test_name -> code
    """
    return {
        "simple_function": """
def simple_func():
    pass
""",
        "function_with_args": """
def func_with_args(a, b, c):
    return a + b + c
""",
        "function_with_defaults": """
def func_with_defaults(a, b=10, c=None):
    return a + (b or 0)
""",
        "function_with_annotations": """
def annotated_func(a: int, b: str = "default") -> str:
    return f"{a}: {b}"
""",
        "function_with_varargs": """
def varargs_func(*args, **kwargs):
    return args, kwargs
""",
        "async_function": """
async def async_func(url: str) -> dict:
    return {"url": url}
""",
        "decorated_function": """
@decorator
@another_decorator(arg="value")
def decorated_func():
    pass
""",
        "nested_function": """
def outer_func(x):
    def inner_func(y):
        return x + y
    return inner_func
""",
        "generator_function": """
def generator_func(n):
    for i in range(n):
        yield i * 2
""",
        "lambda_assignment": """
square = lambda x: x ** 2
""",
    }


def generate_class_test_cases() -> dict[str, str]:
    """
    Generate various class definition test cases.
    
    Returns:
        Dict of test_name -> code
    """
    return {
        "simple_class": """
class SimpleClass:
    pass
""",
        "class_with_init": """
class ClassWithInit:
    def __init__(self, value):
        self.value = value
""",
        "class_with_inheritance": """
class ChildClass(ParentClass):
    def __init__(self):
        super().__init__()
""",
        "class_with_multiple_inheritance": """
class MultipleInheritance(Base1, Base2, Base3):
    pass
""",
        "class_with_class_vars": """
class ClassWithVars:
    class_var = 42
    another_var: int = 100
    
    def __init__(self):
        self.instance_var = 0
""",
        "class_with_methods": """
class ClassWithMethods:
    def method1(self):
        return "method1"
    
    @classmethod
    def class_method(cls):
        return cls.__name__
    
    @staticmethod
    def static_method():
        return "static"
    
    @property
    def prop(self):
        return self._prop
""",
        "decorated_class": """
@dataclass
@decorator(arg="value")
class DecoratedClass:
    field1: int
    field2: str = "default"
""",
        "class_with_metaclass": """
class MetaClass(type):
    pass

class ClassWithMeta(metaclass=MetaClass):
    pass
""",
        "abstract_class": """
from abc import ABC, abstractmethod

class AbstractClass(ABC):
    @abstractmethod
    def abstract_method(self):
        pass
""",
    }