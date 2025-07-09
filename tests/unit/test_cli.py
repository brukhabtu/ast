"""Unit tests for CLI argument parsing and formatting."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from astlib.cli import ASTCli, CLIFormatter


class TestCLIFormatter:
    """Test CLI output formatting."""

    def test_format_plain_empty(self):
        """Test plain format with no results."""
        formatter = CLIFormatter()
        result = formatter.format_plain([])
        assert result == "No results found."

    def test_format_plain_single_result(self):
        """Test plain format with single result."""
        formatter = CLIFormatter()
        results = [{
            'name': 'test_function',
            'file': '/path/to/file.py',
            'line': 10,
            'column': 4,
        }]
        output = formatter.format_plain(results)
        assert output == "/path/to/file.py:10:4 - test_function"

    def test_format_plain_with_docstring(self):
        """Test plain format with docstring."""
        formatter = CLIFormatter()
        results = [{
            'name': 'test_function',
            'file': '/path/to/file.py',
            'line': 10,
            'column': 4,
            'docstring': 'This is a test function that does something interesting.'
        }]
        output = formatter.format_plain(results)
        assert "/path/to/file.py:10:4 - test_function" in output
        assert "This is a test function that does something interesting." in output

    def test_format_plain_truncates_long_docstring(self):
        """Test plain format truncates long docstrings."""
        formatter = CLIFormatter()
        long_docstring = "This is a very long docstring " * 10
        results = [{
            'name': 'test_function',
            'file': '/path/to/file.py',
            'line': 10,
            'column': 4,
            'docstring': long_docstring
        }]
        output = formatter.format_plain(results)
        assert "..." in output

    def test_format_json_empty(self):
        """Test JSON format with no results."""
        formatter = CLIFormatter()
        result = formatter.format_json([])
        assert json.loads(result) == []

    def test_format_json_with_results(self):
        """Test JSON format with results."""
        formatter = CLIFormatter()
        results = [{
            'name': 'test_function',
            'file': '/path/to/file.py',
            'line': 10,
            'column': 4,
            'docstring': 'Test docstring'
        }]
        output = formatter.format_json(results)
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]['name'] == 'test_function'
        assert parsed[0]['line'] == 10

    def test_format_markdown_empty(self):
        """Test Markdown format with no results."""
        formatter = CLIFormatter()
        result = formatter.format_markdown([])
        assert result == "No results found."

    def test_format_markdown_with_results(self):
        """Test Markdown format with results."""
        formatter = CLIFormatter()
        results = [{
            'name': 'test_function',
            'file': '/path/to/file.py',
            'line': 10,
            'column': 4,
            'docstring': 'Test docstring'
        }]
        output = formatter.format_markdown(results)
        assert "# Search Results" in output
        assert "## `test_function`" in output
        assert "Line 10, Column 4" in output
        assert "Test docstring" in output


class TestASTCli:
    """Test AST CLI application."""

    def test_version_flag(self):
        """Test --version flag."""
        cli = ASTCli()
        with pytest.raises(SystemExit) as exc_info:
            cli.run(['--version'])
        assert exc_info.value.code == 0

    def test_no_command(self):
        """Test running without a command."""
        cli = ASTCli()
        with pytest.raises(SystemExit) as exc_info:
            cli.run([])
        assert exc_info.value.code == 2

    def test_find_function_help(self):
        """Test find-function help."""
        cli = ASTCli()
        with pytest.raises(SystemExit) as exc_info:
            cli.run(['find-function', '--help'])
        assert exc_info.value.code == 0

    @patch('astlib.cli.Path')
    def test_find_function_path_not_exists(self, mock_path):
        """Test find-function with non-existent path."""
        mock_path.return_value.resolve.return_value.exists.return_value = False
        
        cli = ASTCli()
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = cli.run(['find-function', 'test', '--path', '/nonexistent'])
        
        assert result == 1
        assert "does not exist" in mock_stderr.getvalue()

    @patch('astlib.cli.Path')
    def test_find_function_path_not_directory(self, mock_path):
        """Test find-function with file instead of directory."""
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = False
        mock_path.return_value.resolve.return_value = mock_resolved
        
        cli = ASTCli()
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = cli.run(['find-function', 'test', '--path', '/some/file.py'])
        
        assert result == 1
        assert "is not a directory" in mock_stderr.getvalue()

    @patch('astlib.cli.DirectoryWalker')
    @patch('astlib.cli.FunctionFinder')
    @patch('astlib.cli.Path')
    def test_find_function_no_python_files(self, mock_path, mock_finder, mock_walker):
        """Test find-function with no Python files found."""
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = True
        mock_path.return_value.resolve.return_value = mock_resolved
        
        mock_walker.return_value.walk.return_value = []
        
        cli = ASTCli()
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = cli.run(['find-function', 'test', '--no-progress'])
        
        assert result == 0
        assert "No Python files found" in mock_stderr.getvalue()

    @patch('astlib.cli.DirectoryWalker')
    @patch('astlib.cli.FunctionFinder')
    @patch('astlib.cli.Path')
    def test_find_function_success_plain(self, mock_path, mock_finder_class, mock_walker_class):
        """Test successful find-function with plain output."""
        # Setup mocks
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = True
        mock_path.return_value.resolve.return_value = mock_resolved
        
        mock_walker = Mock()
        mock_walker.walk.return_value = [Path('/test/file1.py'), Path('/test/file2.py')]
        mock_walker_class.return_value = mock_walker
        
        mock_finder = Mock()
        mock_finder.find_functions.side_effect = [
            [{'name': 'test_func1', 'file': '/test/file1.py', 'line': 10, 'column': 0}],
            [{'name': 'test_func2', 'file': '/test/file2.py', 'line': 20, 'column': 4}]
        ]
        mock_finder_class.return_value = mock_finder
        
        cli = ASTCli()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.run(['find-function', 'test*', '--no-progress'])
        
        assert result == 0
        output = mock_stdout.getvalue()
        assert "/test/file1.py:10:0 - test_func1" in output
        assert "/test/file2.py:20:4 - test_func2" in output

    @patch('astlib.cli.DirectoryWalker')
    @patch('astlib.cli.FunctionFinder')
    @patch('astlib.cli.Path')
    def test_find_function_with_json_format(self, mock_path, mock_finder_class, mock_walker_class):
        """Test find-function with JSON output format."""
        # Setup mocks
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = True
        mock_path.return_value.resolve.return_value = mock_resolved
        
        mock_walker = Mock()
        mock_walker.walk.return_value = [Path('/test/file.py')]
        mock_walker_class.return_value = mock_walker
        
        mock_finder = Mock()
        mock_finder.find_functions.return_value = [
            {'name': 'test_func', 'file': '/test/file.py', 'line': 10, 'column': 0}
        ]
        mock_finder_class.return_value = mock_finder
        
        cli = ASTCli()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.run(['find-function', 'test_func', '--format', 'json', '--no-progress'])
        
        assert result == 0
        output = mock_stdout.getvalue()
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]['name'] == 'test_func'

    @patch('astlib.cli.DirectoryWalker')
    @patch('astlib.cli.FunctionFinder')
    @patch('astlib.cli.Path')
    def test_find_function_with_ignore_patterns(self, mock_path, mock_finder_class, mock_walker_class):
        """Test find-function with ignore patterns."""
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = True
        mock_path.return_value.resolve.return_value = mock_resolved
        
        mock_walker = Mock()
        mock_walker.walk.return_value = []
        mock_walker_class.return_value = mock_walker
        
        cli = ASTCli()
        result = cli.run(['find-function', 'test', '--ignore', 'build/*', '--ignore', 'dist/*', '--no-progress'])
        
        # Verify walker was created with ignore patterns
        mock_walker_class.assert_called_once_with(ignore_patterns=['build/*', 'dist/*'])

    @patch('astlib.cli.DirectoryWalker')
    @patch('astlib.cli.FunctionFinder')
    @patch('astlib.cli.Path')
    def test_find_function_case_sensitive(self, mock_path, mock_finder_class, mock_walker_class):
        """Test find-function with case-sensitive flag."""
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = True
        mock_path.return_value.resolve.return_value = mock_resolved
        
        mock_walker = Mock()
        mock_walker.walk.return_value = []
        mock_walker_class.return_value = mock_walker
        
        mock_finder = Mock()
        mock_finder_class.return_value = mock_finder
        
        cli = ASTCli()
        result = cli.run(['find-function', 'Test', '--case-sensitive', '--no-progress'])
        
        # Verify finder was created with case_sensitive=True
        mock_finder_class.assert_called_once_with(case_sensitive=True)

    @patch('astlib.cli.DirectoryWalker')
    @patch('astlib.cli.FunctionFinder')
    @patch('astlib.cli.Path')
    @patch('astlib.cli.tqdm')
    def test_find_function_handles_errors(self, mock_tqdm, mock_path, mock_finder_class, mock_walker_class):
        """Test find-function handles file processing errors gracefully."""
        mock_resolved = Mock()
        mock_resolved.exists.return_value = True
        mock_resolved.is_dir.return_value = True
        mock_path.return_value.resolve.return_value = mock_resolved
        
        mock_walker = Mock()
        mock_walker.walk.return_value = [Path('/test/file1.py'), Path('/test/file2.py')]
        mock_walker_class.return_value = mock_walker
        
        mock_finder = Mock()
        # First file raises exception, second file works
        mock_finder.find_functions.side_effect = [
            Exception("Parse error"),
            [{'name': 'test_func', 'file': '/test/file2.py', 'line': 10, 'column': 0}]
        ]
        mock_finder_class.return_value = mock_finder
        
        # Mock tqdm to capture warning messages
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        mock_tqdm_instance.__iter__ = Mock(return_value=iter([Path('/test/file1.py'), Path('/test/file2.py')]))
        
        cli = ASTCli()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.run(['find-function', 'test'])
        
        assert result == 0
        output = mock_stdout.getvalue()
        # Should still have results from file2
        assert "test_func" in output
        
        # Verify warning was written
        mock_tqdm.write.assert_called()
        warning_call = mock_tqdm.write.call_args[0][0]
        assert "Warning:" in warning_call
        assert "file1.py" in warning_call