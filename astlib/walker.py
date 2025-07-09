"""Directory walker with .gitignore support."""

import fnmatch
import os
from pathlib import Path
from typing import Iterator, List, Optional


class DirectoryWalker:
    """Walk directories and find Python files with .gitignore support."""

    def __init__(self, ignore_patterns: Optional[List[str]] = None):
        """Initialize walker with ignore patterns.
        
        Args:
            ignore_patterns: List of glob patterns to ignore
        """
        self.ignore_patterns = ignore_patterns or []
        self._default_ignores = [
            '.git', '__pycache__', '*.pyc', '*.pyo', '*.pyd',
            '.Python', 'env/', 'venv/', '.venv/', 'ENV/',
            'build/', 'develop-eggs/', 'dist/', 'downloads/',
            'eggs/', '.eggs/', 'lib/', 'lib64/', 'parts/',
            'sdist/', 'var/', 'wheels/', '*.egg-info/', '.installed.cfg',
            '*.egg', 'htmlcov/', '.tox/', '.coverage', '.coverage.*',
            '.cache', 'nosetests.xml', 'coverage.xml', '*.cover',
            '.hypothesis/', '.pytest_cache/', '.mypy_cache/',
            '.ruff_cache/', '*.so', '.DS_Store'
        ]

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns."""
        path_str = str(path)
        
        # Check against default ignores
        for pattern in self._default_ignores:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
        
        # Check against user-provided patterns
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
            # Support directory patterns like "build/*"
            if pattern.endswith('/*') and path.is_dir():
                dir_pattern = pattern[:-2]
                if path.name == dir_pattern or path_str.endswith(f'/{dir_pattern}'):
                    return True
        
        return False

    def _load_gitignore(self, directory: Path) -> List[str]:
        """Load .gitignore patterns from a directory."""
        gitignore_path = directory / '.gitignore'
        patterns = []
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception:
                # Silently ignore errors reading .gitignore
                pass
        
        return patterns

    def walk(self, directory: Path) -> Iterator[Path]:
        """Walk directory and yield Python files.
        
        Args:
            directory: Root directory to walk
            
        Yields:
            Path objects for each Python file found
        """
        directory = Path(directory).resolve()
        
        # Load root .gitignore if exists
        root_gitignore = self._load_gitignore(directory)
        
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            
            # Check if current directory should be ignored
            if self._should_ignore(root_path):
                dirs.clear()  # Don't descend into ignored directories
                continue
            
            # Remove ignored directories from dirs list (modifies in-place)
            dirs[:] = [
                d for d in dirs 
                if not self._should_ignore(root_path / d)
            ]
            
            # Process Python files
            for file in files:
                if file.endswith('.py'):
                    file_path = root_path / file
                    if not self._should_ignore(file_path):
                        yield file_path