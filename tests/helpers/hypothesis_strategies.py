"""
Hypothesis strategies for property-based testing of AST operations.

This module provides strategies for generating:
- Valid Python code
- AST nodes
- Complex code patterns
- Edge cases
"""

import ast
import string
from typing import Any, Optional

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy


# --- Basic Strategies ---

@st.composite
def valid_identifier(draw: st.DrawFn, prefix: str = "") -> str:
    """Generate valid Python identifiers."""
    if prefix:
        base = prefix
    else:
        # First character must be letter or underscore
        base = draw(st.sampled_from(string.ascii_letters + "_"))
    
    # Subsequent characters can include digits
    suffix_chars = string.ascii_letters + string.digits + "_"
    suffix = draw(st.text(suffix_chars, min_size=0, max_size=10))
    
    identifier = base + suffix
    
    # Avoid Python keywords
    keywords = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
        'while', 'with', 'yield'
    }
    
    if identifier in keywords:
        identifier = f"{identifier}_"
    
    return identifier


@st.composite
def import_name(draw: st.DrawFn) -> str:
    """Generate valid module/package names."""
    parts = draw(st.lists(valid_identifier(), min_size=1, max_size=4))
    return ".".join(parts)


# --- AST Node Strategies ---

@st.composite
def ast_constant(draw: st.DrawFn) -> ast.Constant:
    """Generate ast.Constant nodes with various types."""
    value = draw(st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1000, max_value=1000),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(string.printable, min_size=0, max_size=20),
        st.binary(min_size=0, max_size=20),
    ))
    return ast.Constant(value=value)


@st.composite
def ast_name(draw: st.DrawFn, ctx: Optional[type[ast.AST]] = None) -> ast.Name:
    """Generate ast.Name nodes."""
    id = draw(valid_identifier())
    if ctx is None:
        ctx = draw(st.sampled_from([ast.Load, ast.Store, ast.Del]))
    return ast.Name(id=id, ctx=ctx())


@st.composite
def ast_arg(draw: st.DrawFn) -> ast.arg:
    """Generate ast.arg nodes for function parameters."""
    arg_name = draw(valid_identifier("param"))
    # Optionally add type annotation
    annotation = draw(st.one_of(
        st.none(),
        ast_name(ctx=ast.Load),
    ))
    return ast.arg(arg=arg_name, annotation=annotation)


@st.composite
def ast_arguments(draw: st.DrawFn) -> ast.arguments:
    """Generate ast.arguments nodes."""
    # Generate positional arguments
    args = draw(st.lists(ast_arg(), min_size=0, max_size=5))
    
    # For simplicity, we'll keep other argument types minimal
    return ast.arguments(
        posonlyargs=[],
        args=args,
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )


@st.composite
def simple_statement(draw: st.DrawFn) -> ast.AST:
    """Generate simple statement nodes."""
    return draw(st.one_of(
        st.just(ast.Pass()),
        st.just(ast.Continue()),
        st.just(ast.Break()),
        st.builds(
            ast.Return,
            value=st.one_of(st.none(), ast_constant(), ast_name()),
        ),
        st.builds(
            ast.Assign,
            targets=st.lists(ast_name(ctx=ast.Store), min_size=1, max_size=1),
            value=st.one_of(ast_constant(), ast_name()),
        ),
    ))


@st.composite
def ast_function_def(draw: st.DrawFn) -> ast.FunctionDef:
    """Generate ast.FunctionDef nodes."""
    name = draw(valid_identifier("func"))
    args = draw(ast_arguments())
    body = draw(st.lists(simple_statement(), min_size=1, max_size=3))
    
    # Optionally add decorators
    decorator_list = draw(st.lists(ast_name(), min_size=0, max_size=2))
    
    # Optionally add return annotation
    returns = draw(st.one_of(st.none(), ast_name()))
    
    return ast.FunctionDef(
        name=name,
        args=args,
        body=body,
        decorator_list=decorator_list,
        returns=returns,
    )


@st.composite
def ast_class_def(draw: st.DrawFn) -> ast.ClassDef:
    """Generate ast.ClassDef nodes."""
    name = draw(valid_identifier("Class"))
    
    # Optionally add base classes
    bases = draw(st.lists(ast_name(), min_size=0, max_size=2))
    
    # Class body with mix of methods and assignments
    body = draw(st.lists(
        st.one_of(simple_statement(), ast_function_def()),
        min_size=1,
        max_size=5
    ))
    
    return ast.ClassDef(
        name=name,
        bases=bases,
        keywords=[],
        body=body,
        decorator_list=[],
    )


@st.composite
def ast_import(draw: st.DrawFn) -> ast.Import:
    """Generate ast.Import nodes."""
    num_imports = draw(st.integers(min_value=1, max_value=3))
    aliases = []
    
    for _ in range(num_imports):
        name = draw(import_name())
        asname = draw(st.one_of(st.none(), valid_identifier()))
        aliases.append(ast.alias(name=name, asname=asname))
    
    return ast.Import(names=aliases)


@st.composite
def ast_import_from(draw: st.DrawFn) -> ast.ImportFrom:
    """Generate ast.ImportFrom nodes."""
    module = draw(st.one_of(st.none(), import_name()))
    level = draw(st.integers(min_value=0, max_value=3))
    
    num_imports = draw(st.integers(min_value=1, max_value=3))
    names = []
    
    for _ in range(num_imports):
        name = draw(valid_identifier())
        asname = draw(st.one_of(st.none(), valid_identifier()))
        names.append(ast.alias(name=name, asname=asname))
    
    return ast.ImportFrom(module=module, names=names, level=level)


@st.composite
def ast_module(draw: st.DrawFn) -> ast.Module:
    """Generate complete ast.Module nodes."""
    # Mix of imports, functions, classes, and simple statements
    body = draw(st.lists(
        st.one_of(
            ast_import(),
            ast_import_from(),
            ast_function_def(),
            ast_class_def(),
            simple_statement(),
        ),
        min_size=1,
        max_size=10
    ))
    
    return ast.Module(body=body, type_ignores=[])


# --- Code Generation Strategies ---

@st.composite
def simple_expression_code(draw: st.DrawFn) -> str:
    """Generate simple Python expressions as code strings."""
    expr_type = draw(st.sampled_from([
        "literal",
        "binary_op",
        "unary_op",
        "comparison",
        "function_call",
    ]))
    
    if expr_type == "literal":
        value = draw(st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(string.printable, min_size=0, max_size=20).map(repr),
            st.booleans().map(str),
        ))
        return str(value)
    
    elif expr_type == "binary_op":
        left = draw(st.integers(min_value=1, max_value=100))
        right = draw(st.integers(min_value=1, max_value=100))
        op = draw(st.sampled_from(["+", "-", "*", "//", "%", "**"]))
        return f"{left} {op} {right}"
    
    elif expr_type == "unary_op":
        operand = draw(st.integers(min_value=-100, max_value=100))
        op = draw(st.sampled_from(["-", "+", "~"]))
        return f"{op}{operand}"
    
    elif expr_type == "comparison":
        left = draw(st.integers(min_value=1, max_value=100))
        right = draw(st.integers(min_value=1, max_value=100))
        op = draw(st.sampled_from(["<", ">", "<=", ">=", "==", "!="]))
        return f"{left} {op} {right}"
    
    else:  # function_call
        func = draw(valid_identifier())
        args = draw(st.lists(st.integers(), min_size=0, max_size=3))
        args_str = ", ".join(map(str, args))
        return f"{func}({args_str})"


@st.composite
def function_code(draw: st.DrawFn) -> str:
    """Generate valid function definitions as code strings."""
    name = draw(valid_identifier("func"))
    params = draw(st.lists(valid_identifier("param"), min_size=0, max_size=4))
    params_str = ", ".join(params)
    
    # Simple function body
    body_type = draw(st.sampled_from(["pass", "return_constant", "return_expr"]))
    
    if body_type == "pass":
        body = "    pass"
    elif body_type == "return_constant":
        value = draw(st.integers(min_value=-100, max_value=100))
        body = f"    return {value}"
    else:  # return_expr
        if params:
            body = f"    return {' + '.join(params)}"
        else:
            body = "    return 0"
    
    return f"def {name}({params_str}):\n{body}"


@st.composite
def class_code(draw: st.DrawFn) -> str:
    """Generate valid class definitions as code strings."""
    name = draw(valid_identifier("Class"))
    
    # Optionally inherit from object or another class
    base = draw(st.one_of(
        st.none(),
        st.just("object"),
        valid_identifier("Base"),
    ))
    
    inheritance = f"({base})" if base else ""
    
    # Simple class body
    has_init = draw(st.booleans())
    
    if has_init:
        body = """    def __init__(self):
        self.value = 42"""
    else:
        body = "    pass"
    
    return f"class {name}{inheritance}:\n{body}"


@st.composite
def valid_python_code(draw: st.DrawFn) -> str:
    """Generate valid Python code snippets."""
    code_type = draw(st.sampled_from([
        "expression",
        "assignment",
        "function",
        "class",
        "import",
        "mixed",
    ]))
    
    if code_type == "expression":
        return draw(simple_expression_code())
    
    elif code_type == "assignment":
        var = draw(valid_identifier())
        value = draw(simple_expression_code())
        return f"{var} = {value}"
    
    elif code_type == "function":
        return draw(function_code())
    
    elif code_type == "class":
        return draw(class_code())
    
    elif code_type == "import":
        import_type = draw(st.booleans())
        if import_type:
            module = draw(import_name())
            return f"import {module}"
        else:
            module = draw(import_name())
            name = draw(valid_identifier())
            return f"from {module} import {name}"
    
    else:  # mixed
        parts = draw(st.lists(
            st.one_of(
                function_code(),
                class_code(),
                st.builds(lambda v, e: f"{v} = {e}", valid_identifier(), simple_expression_code()),
            ),
            min_size=2,
            max_size=5
        ))
        return "\n\n".join(parts)


# --- Edge Case Strategies ---

@st.composite
def edge_case_code(draw: st.DrawFn) -> str:
    """Generate Python code with edge cases."""
    case_type = draw(st.sampled_from([
        "empty",
        "unicode",
        "deeply_nested",
        "many_arguments",
        "long_lines",
    ]))
    
    if case_type == "empty":
        # Various empty constructs
        return draw(st.sampled_from([
            "def f(): pass",
            "class C: pass",
            "if True: pass",
            "while False: pass",
            "for _ in []: pass",
            "try: pass\nexcept: pass",
        ]))
    
    elif case_type == "unicode":
        # Unicode identifiers (Python 3+)
        return draw(st.sampled_from([
            "def 函数(): return '中文'",
            "λ = lambda x: x ** 2",
            "π = 3.14159",
            "class Café: pass",
        ]))
    
    elif case_type == "deeply_nested":
        depth = draw(st.integers(min_value=3, max_value=6))
        code = "def f():\n"
        for i in range(depth):
            indent = "    " * (i + 1)
            code += f"{indent}if True:\n"
        code += "    " * (depth + 1) + "pass"
        return code
    
    elif case_type == "many_arguments":
        num_args = draw(st.integers(min_value=10, max_value=20))
        args = [f"arg{i}" for i in range(num_args)]
        return f"def f({', '.join(args)}): pass"
    
    else:  # long_lines
        # Generate a long expression
        nums = draw(st.lists(st.integers(), min_size=20, max_size=30))
        return f"result = {' + '.join(map(str, nums))}"


# --- Composite Strategies ---

def ast_node() -> SearchStrategy[ast.AST]:
    """Strategy for generating any AST node."""
    return st.one_of(
        ast_constant(),
        ast_name(),
        ast_function_def(),
        ast_class_def(),
        ast_import(),
        ast_import_from(),
        simple_statement(),
    )


def python_code() -> SearchStrategy[str]:
    """Strategy for generating Python code."""
    return st.one_of(
        valid_python_code(),
        edge_case_code(),
    )


def parseable_code() -> SearchStrategy[tuple[str, ast.AST]]:
    """Generate code that can be parsed along with its AST."""
    return valid_python_code().map(lambda code: (code, ast.parse(code)))