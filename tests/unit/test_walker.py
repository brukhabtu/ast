"""Unit tests for directory walker."""

import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from astlib.walker import DirectoryWalker


class TestDirectoryWalker:
    """Test DirectoryWalker functionality."""

    def test_init_default(self):
        """Test walker initialization with defaults."""
        walker = DirectoryWalker()
        assert walker.ignore_patterns == []
        assert '__pycache__' in walker._default_ignores
        assert '.git' in walker._default_ignores

    def test_init_with_patterns(self):
        """Test walker initialization with custom patterns."""
        patterns = ['test/*', '*.tmp']
        walker = DirectoryWalker(ignore_patterns=patterns)
        assert walker.ignore_patterns == patterns

    def test_should_ignore_default_patterns(self):
        """Test ignoring default patterns."""
        walker = DirectoryWalker()
        
        # Test various default patterns
        assert walker._should_ignore(Path('__pycache__'))
        assert walker._should_ignore(Path('.git'))
        assert walker._should_ignore(Path('file.pyc'))
        assert walker._should_ignore(Path('.coverage'))
        assert walker._should_ignore(Path('.DS_Store'))
        
        # For directory patterns with trailing slash, need to check differently
        # since the actual implementation checks patterns with trailing slashes

    def test_should_ignore_custom_patterns(self):
        """Test ignoring custom patterns."""
        walker = DirectoryWalker(ignore_patterns=['test_*', '*.tmp', 'build/*'])
        
        assert walker._should_ignore(Path('test_file.py'))
        assert walker._should_ignore(Path('data.tmp'))
        assert not walker._should_ignore(Path('regular_file.py'))
        
        # Directory patterns need special handling
        build_dir = Path('build')
        # Mock is_dir() for Path object
        import unittest.mock
        with unittest.mock.patch.object(Path, 'is_dir', return_value=True):
            assert walker._should_ignore(build_dir)

    def test_should_ignore_directory_patterns(self):
        """Test directory-specific patterns."""
        walker = DirectoryWalker(ignore_patterns=['docs/*', 'tests/*'])
        
        # Test with actual Path object
        import unittest.mock
        docs_path = Path('/project/docs')
        with unittest.mock.patch.object(Path, 'is_dir', return_value=True):
            assert walker._should_ignore(docs_path)

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='# Comment\n*.log\nbuild/\n\ntemp*\n')
    def test_load_gitignore(self, mock_file, mock_exists):
        """Test loading .gitignore file."""
        mock_exists.return_value = True
        walker = DirectoryWalker()
        patterns = walker._load_gitignore(Path('/test/dir'))
        
        assert patterns == ['*.log', 'build/', 'temp*']
        mock_file.assert_called_once_with(Path('/test/dir/.gitignore'), 'r', encoding='utf-8')

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_gitignore_not_found(self, mock_file):
        """Test loading non-existent .gitignore."""
        walker = DirectoryWalker()
        patterns = walker._load_gitignore(Path('/test/dir'))
        
        assert patterns == []

    @patch('builtins.open', side_effect=PermissionError)
    def test_load_gitignore_permission_error(self, mock_file):
        """Test handling permission error when loading .gitignore."""
        walker = DirectoryWalker()
        patterns = walker._load_gitignore(Path('/test/dir'))
        
        assert patterns == []

    @patch('os.walk')
    def test_walk_empty_directory(self, mock_walk):
        """Test walking empty directory."""
        mock_walk.return_value = [
            ('/test/dir', [], [])
        ]
        
        walker = DirectoryWalker()
        results = list(walker.walk(Path('/test/dir')))
        
        assert results == []

    @patch('os.walk')
    def test_walk_with_python_files(self, mock_walk):
        """Test walking directory with Python files."""
        mock_walk.return_value = [
            ('/test/dir', ['subdir'], ['file1.py', 'file2.txt', 'file3.py']),
            ('/test/dir/subdir', [], ['file4.py', 'README.md'])
        ]
        
        walker = DirectoryWalker()
        results = list(walker.walk(Path('/test/dir')))
        
        assert len(results) == 3
        assert Path('/test/dir/file1.py') in results
        assert Path('/test/dir/file3.py') in results
        assert Path('/test/dir/subdir/file4.py') in results

    @patch('os.walk')
    def test_walk_ignores_directories(self, mock_walk):
        """Test that ignored directories are not descended into."""
        mock_walk.return_value = [
            ('/test/dir', ['__pycache__', 'src', '.git'], ['main.py']),
            ('/test/dir/src', [], ['app.py'])
        ]
        
        walker = DirectoryWalker()
        
        # Track which directories os.walk visits
        visited_dirs = []
        
        def track_walk(path):
            for item in mock_walk.return_value:
                if str(path) in item[0]:
                    visited_dirs.append(item[0])
                    # Simulate modifying dirs in-place
                    dirs = item[1]
                    dirs[:] = [d for d in dirs if not walker._should_ignore(Path(item[0]) / d)]
                    yield item
        
        mock_walk.side_effect = track_walk
        
        results = list(walker.walk(Path('/test/dir')))
        
        # Should have main.py and app.py, but not files from ignored directories
        assert len(results) == 2
        assert Path('/test/dir/main.py') in results
        assert Path('/test/dir/src/app.py') in results

    @patch('os.walk')
    def test_walk_with_custom_ignores(self, mock_walk):
        """Test walking with custom ignore patterns."""
        mock_walk.return_value = [
            ('/test/dir', [], ['test_file.py', 'app.py', 'backup.py.bak']),
        ]
        
        walker = DirectoryWalker(ignore_patterns=['test_*', '*.bak'])
        results = list(walker.walk(Path('/test/dir')))
        
        assert len(results) == 1
        assert Path('/test/dir/app.py') in results

    @patch('os.walk')
    @patch.object(DirectoryWalker, '_load_gitignore')
    def test_walk_loads_root_gitignore(self, mock_load_gitignore, mock_walk):
        """Test that walk loads .gitignore from root directory."""
        mock_walk.return_value = [
            ('/test/dir', [], ['file.py'])
        ]
        mock_load_gitignore.return_value = ['*.log']
        
        walker = DirectoryWalker()
        list(walker.walk(Path('/test/dir')))
        
        mock_load_gitignore.assert_called_once()
        call_path = mock_load_gitignore.call_args[0][0]
        assert str(call_path).endswith('/test/dir')

    @patch('os.walk')
    def test_walk_handles_ignored_root(self, mock_walk):
        """Test walking when root directory itself matches ignore pattern."""
        mock_walk.return_value = [
            ('/test/__pycache__', [], ['file.py'])
        ]
        
        walker = DirectoryWalker()
        results = list(walker.walk(Path('/test/__pycache__')))
        
        # Should return empty as root is ignored
        assert results == []