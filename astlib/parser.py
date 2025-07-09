"""
Core AST parser with error recovery capabilities.

This module provides robust parsing of Python source files with graceful
handling of syntax errors, returning partial ASTs when possible.
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple
import tokenize
import io


@dataclass
class ParseError:
    """Represents a parsing error with location information."""
    message: str
    line: int
    column: int
    line_text: Optional[str] = None
    error_type: str = "SyntaxError"


@dataclass
class ParseResult:
    """Result of parsing operation, containing AST and any errors."""
    tree: Optional[ast.AST] = None
    errors: List[ParseError] = field(default_factory=list)
    source: Optional[str] = None
    filepath: Optional[Path] = None
    
    @property
    def success(self) -> bool:
        """Check if parsing was completely successful."""
        return self.tree is not None and not self.errors
    
    @property
    def partial_success(self) -> bool:
        """Check if we got a partial AST despite errors."""
        return self.tree is not None and bool(self.errors)


class ErrorRecoveringParser:
    """Parser that attempts to recover from syntax errors."""
    
    def __init__(self, source: str, filepath: Optional[Path] = None):
        self.source = source
        self.filepath = filepath
        self.lines = source.splitlines(keepends=True)
        self.errors: List[ParseError] = []
    
    def parse(self) -> ParseResult:
        """
        Parse source code with error recovery.
        
        Returns ParseResult with AST (possibly partial) and any errors encountered.
        """
        # First, try standard parsing
        try:
            tree = ast.parse(self.source, filename=str(self.filepath) if self.filepath else "<string>")
            return ParseResult(tree=tree, source=self.source, filepath=self.filepath)
        except SyntaxError as e:
            # Record the error
            self._record_syntax_error(e)
            
            # Try error recovery strategies
            tree = self._try_recovery_strategies()
            
            return ParseResult(
                tree=tree,
                errors=self.errors,
                source=self.source,
                filepath=self.filepath
            )
    
    def _record_syntax_error(self, error: SyntaxError) -> None:
        """Record a syntax error with context."""
        line_text = None
        if error.lineno and 0 < error.lineno <= len(self.lines):
            line_text = self.lines[error.lineno - 1].rstrip()
        
        self.errors.append(ParseError(
            message=str(error.msg),
            line=error.lineno or 0,
            column=error.offset or 0,
            line_text=line_text,
            error_type=type(error).__name__
        ))
    
    def _try_recovery_strategies(self) -> Optional[ast.AST]:
        """Try various strategies to recover a partial AST."""
        # Strategy 1: Try parsing up to the error line
        tree = self._parse_until_error()
        if tree:
            return tree
        
        # Strategy 2: Try parsing individual top-level statements
        tree = self._parse_by_statements()
        if tree:
            return tree
        
        # Strategy 3: Try parsing with common fixes
        tree = self._parse_with_fixes()
        if tree:
            return tree
        
        return None
    
    def _parse_until_error(self) -> Optional[ast.AST]:
        """Try parsing source up to the first error line."""
        if not self.errors:
            return None
        
        first_error = self.errors[0]
        if first_error.line <= 1:
            return None
        
        # Try parsing up to the error line
        truncated_source = ''.join(self.lines[:first_error.line - 1])
        
        try:
            return ast.parse(truncated_source, filename=str(self.filepath) if self.filepath else "<string>")
        except SyntaxError:
            return None
    
    def _parse_by_statements(self) -> Optional[ast.AST]:
        """Parse source statement by statement, collecting valid ones."""
        valid_statements = []
        current_statement_lines = []
        
        # Use tokenizer to find statement boundaries
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(self.source).readline))
        except tokenize.TokenError:
            # Even tokenization failed, try line-by-line
            return self._parse_line_by_line()
        
        # Group tokens by statements (simplified - looks for NEWLINE at indent 0)
        current_indent = 0
        statement_start = 0
        
        for i, token in enumerate(tokens):
            if token.type == tokenize.INDENT:
                current_indent += 1
            elif token.type == tokenize.DEDENT:
                current_indent -= 1
            elif token.type == tokenize.NEWLINE and current_indent == 0:
                # Try to parse this statement
                statement_end = token.end[0]
                statement_source = ''.join(self.lines[statement_start:statement_end])
                
                try:
                    stmt_ast = ast.parse(statement_source)
                    valid_statements.extend(stmt_ast.body)
                except SyntaxError:
                    # Record error but continue
                    pass
                
                statement_start = statement_end
        
        if valid_statements:
            # Create a module with the valid statements
            module = ast.Module(body=valid_statements, type_ignores=[])
            ast.fix_missing_locations(module)
            return module
        
        return None
    
    def _parse_line_by_line(self) -> Optional[ast.AST]:
        """Fallback: try parsing line by line."""
        valid_statements = []
        
        for i, line in enumerate(self.lines):
            if line.strip():
                try:
                    stmt_ast = ast.parse(line)
                    valid_statements.extend(stmt_ast.body)
                except SyntaxError:
                    pass
        
        if valid_statements:
            module = ast.Module(body=valid_statements, type_ignores=[])
            ast.fix_missing_locations(module)
            return module
        
        return None
    
    def _parse_with_fixes(self) -> Optional[ast.AST]:
        """Try parsing with common syntax fixes."""
        # Try adding missing colons
        fixed_source = self._fix_missing_colons(self.source)
        if fixed_source != self.source:
            try:
                tree = ast.parse(fixed_source)
                self.errors.append(ParseError(
                    message="Fixed missing colons",
                    line=0,
                    column=0,
                    error_type="AutoFixed"
                ))
                return tree
            except SyntaxError:
                pass
        
        # Try fixing indentation
        fixed_source = self._fix_indentation(self.source)
        if fixed_source != self.source:
            try:
                tree = ast.parse(fixed_source)
                self.errors.append(ParseError(
                    message="Fixed indentation issues",
                    line=0,
                    column=0,
                    error_type="AutoFixed"
                ))
                return tree
            except SyntaxError:
                pass
        
        return None
    
    def _fix_missing_colons(self, source: str) -> str:
        """Try to fix missing colons after control statements."""
        lines = source.splitlines()
        fixed_lines = []
        
        control_keywords = {'if', 'elif', 'else', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with'}
        
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(kw + ' ') or stripped == kw for kw in control_keywords):
                if not stripped.endswith(':'):
                    line = line.rstrip() + ':'
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_indentation(self, source: str) -> str:
        """Try to fix common indentation issues."""
        # This is a simplified version - real implementation would be more sophisticated
        lines = source.splitlines()
        fixed_lines = []
        expected_indent = 0
        
        for line in lines:
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # Calculate actual indent
            actual_indent = len(line) - len(line.lstrip())
            
            # Try to fix obvious issues
            if actual_indent % 4 != 0:
                # Round to nearest multiple of 4
                fixed_indent = round(actual_indent / 4) * 4
                line = ' ' * fixed_indent + line.lstrip()
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)


def parse_file(filepath: Union[str, Path]) -> ParseResult:
    """
    Parse a Python file with error recovery.
    
    Args:
        filepath: Path to the Python file to parse
        
    Returns:
        ParseResult containing AST and any errors
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file can't be read
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Try to get from cache first
    from .cache import get_cache_manager
    manager = get_cache_manager()
    cached_result = manager.get_cached_ast(filepath)
    if cached_result is not None:
        return cached_result
    
    try:
        source = filepath.read_text(encoding='utf-8')
    except Exception as e:
        raise IOError(f"Failed to read file {filepath}: {e}")
    
    parser = ErrorRecoveringParser(source, filepath)
    result = parser.parse()
    
    # Cache the result
    manager.cache_ast(filepath, result)
    
    return result