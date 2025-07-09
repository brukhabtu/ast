"""Function finder using AST parsing."""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class FunctionFinder:
    """Find functions in Python files using AST."""

    def __init__(self, case_sensitive: bool = False):
        """Initialize finder with search options.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching
        """
        self.case_sensitive = case_sensitive

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches the search pattern.
        
        Args:
            name: Function name to check
            pattern: Search pattern (supports wildcards)
            
        Returns:
            True if name matches pattern
        """
        # Convert pattern to regex
        # Support * as wildcard and ? as single character
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        
        # Add anchors to match full name
        regex_pattern = f'^{regex_pattern}$'
        
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return bool(re.match(regex_pattern, name, flags))

    def _extract_docstring(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract docstring from function node."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value
        return None

    def find_functions(self, file_path: Path, pattern: str) -> List[Dict[str, Any]]:
        """Find functions matching pattern in a Python file.
        
        Args:
            file_path: Path to Python file
            pattern: Function name pattern to search for
            
        Returns:
            List of dictionaries with function information
        """
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            # Return empty results on read error
            return results

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            # Return empty results on parse error
            return results

        # Walk the AST to find functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._matches_pattern(node.name, pattern):
                    result = {
                        'name': node.name,
                        'file': str(file_path),
                        'line': node.lineno,
                        'column': node.col_offset,
                        'end_line': node.end_lineno,
                        'end_column': node.end_col_offset,
                    }
                    
                    # Extract docstring if available
                    docstring = self._extract_docstring(node)
                    if docstring:
                        result['docstring'] = docstring.strip()
                    
                    # Extract decorators
                    if node.decorator_list:
                        decorators = []
                        for dec in node.decorator_list:
                            if isinstance(dec, ast.Name):
                                decorators.append(dec.id)
                            elif isinstance(dec, ast.Attribute):
                                decorators.append(dec.attr)
                        if decorators:
                            result['decorators'] = decorators
                    
                    # Check if it's an async function
                    if isinstance(node, ast.AsyncFunctionDef):
                        result['is_async'] = True
                    
                    results.append(result)

        return results