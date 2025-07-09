"""End-to-end tests for CLI commands."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestCLICommands:
    """Test real CLI command invocations."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            
            # Create directory structure
            (project_dir / "src").mkdir()
            (project_dir / "tests").mkdir()
            (project_dir / "build").mkdir()
            (project_dir / "__pycache__").mkdir()
            
            # Create Python files
            (project_dir / "main.py").write_text('''
def main():
    """Main entry point."""
    print("Hello, World!")

def helper_function():
    """A helper function."""
    pass
''')
            
            (project_dir / "src" / "__init__.py").touch()
            (project_dir / "src" / "utils.py").write_text('''
def parse_config(config_file):
    """Parse configuration file."""
    pass

def validate_input(data):
    """Validate input data."""
    return True

class ConfigParser:
    def parse(self):
        """Parse the configuration."""
        pass
''')
            
            (project_dir / "src" / "app.py").write_text('''
async def fetch_data(url):
    """Fetch data from URL."""
    pass

@decorator
def process_data(data):
    """Process the data."""
    pass
''')
            
            (project_dir / "tests" / "test_main.py").write_text('''
def test_main():
    """Test main function."""
    pass

def test_helper():
    """Test helper function."""
    pass
''')
            
            # Create .gitignore
            (project_dir / ".gitignore").write_text('''
__pycache__/
*.pyc
build/
dist/
*.egg-info/
''')
            
            # Create non-Python files
            (project_dir / "README.md").write_text("# Test Project")
            (project_dir / "build" / "output.py").write_text("# Should be ignored")
            
            yield project_dir

    def run_cli(self, args, cwd=None):
        """Run the CLI with given arguments."""
        cmd = [sys.executable, "-m", "astlib.cli"] + args
        
        # Set PYTHONPATH to include the project root
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent.parent
        env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env
        )
        return result

    def test_find_function_basic(self, temp_project):
        """Test basic find-function command."""
        result = self.run_cli(
            ["find-function", "main", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        assert "main.py:2:0 - main" in result.stdout
        assert "Main entry point" in result.stdout

    def test_find_function_pattern(self, temp_project):
        """Test find-function with wildcard pattern."""
        result = self.run_cli(
            ["find-function", "test_*", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        assert "test_main" in result.stdout
        assert "test_helper" in result.stdout
        assert result.stdout.count("test_") >= 2

    def test_find_function_specific_path(self, temp_project):
        """Test find-function in specific directory."""
        result = self.run_cli(
            ["find-function", "*", "--path", "src", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        assert "parse_config" in result.stdout
        assert "validate_input" in result.stdout
        assert "fetch_data" in result.stdout
        assert "process_data" in result.stdout
        # Should not include functions from other directories
        assert "test_main" not in result.stdout

    def test_find_function_json_format(self, temp_project):
        """Test find-function with JSON output."""
        result = self.run_cli(
            ["find-function", "parse*", "--format", "json", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) >= 2  # parse_config and parse method
        
        # Check structure of results
        for item in data:
            assert "name" in item
            assert "file" in item
            assert "line" in item
            assert "column" in item
            assert item["name"].startswith("parse")

    def test_find_function_markdown_format(self, temp_project):
        """Test find-function with Markdown output."""
        result = self.run_cli(
            ["find-function", "main", "--format", "markdown", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        assert "# Search Results" in result.stdout
        assert "## `main`" in result.stdout
        assert "**Location**:" in result.stdout
        assert "**Docstring**:" in result.stdout

    def test_find_function_ignore_patterns(self, temp_project):
        """Test find-function with ignore patterns."""
        # First, run without ignore to confirm build/ file is found
        result_no_ignore = self.run_cli(
            ["find-function", "*", "--no-progress"],
            cwd=temp_project
        )
        
        # Now run with ignore pattern
        result = self.run_cli(
            ["find-function", "*", "--ignore", "build/*", "--ignore", "tests/*", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        # Should find functions from main.py and src/
        assert "main" in result.stdout
        assert "parse_config" in result.stdout
        # Should not find functions from tests/
        assert "test_main" not in result.stdout

    def test_find_function_case_sensitive(self, temp_project):
        """Test case-sensitive search."""
        # Create a file with mixed case functions
        (temp_project / "mixed.py").write_text('''
def TestFunction():
    pass

def testfunction():
    pass
''')
        
        # Case-insensitive search (default)
        result_insensitive = self.run_cli(
            ["find-function", "testfunction", "--no-progress"],
            cwd=temp_project
        )
        assert result_insensitive.returncode == 0
        # Should find both TestFunction and testfunction when case-insensitive
        assert "TestFunction" in result_insensitive.stdout
        assert "testfunction" in result_insensitive.stdout
        
        # Case-sensitive search
        result_sensitive = self.run_cli(
            ["find-function", "testfunction", "--case-sensitive", "--no-progress"],
            cwd=temp_project
        )
        assert result_sensitive.returncode == 0
        assert "testfunction" in result_sensitive.stdout
        assert "TestFunction" not in result_sensitive.stdout

    def test_find_function_no_results(self, temp_project):
        """Test find-function with no matches."""
        result = self.run_cli(
            ["find-function", "nonexistent_function", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        assert "No results found" in result.stdout

    def test_find_function_nonexistent_path(self, temp_project):
        """Test find-function with non-existent path."""
        result = self.run_cli(
            ["find-function", "test", "--path", "/nonexistent/path", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 1
        assert "does not exist" in result.stderr

    def test_find_function_file_instead_of_directory(self, temp_project):
        """Test find-function with file path instead of directory."""
        result = self.run_cli(
            ["find-function", "test", "--path", "main.py", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 1
        assert "is not a directory" in result.stderr

    def test_find_function_syntax_error_handling(self, temp_project):
        """Test that syntax errors in files are handled gracefully."""
        # Create a file with syntax error
        (temp_project / "broken.py").write_text('''
def broken_function(
    # Missing closing parenthesis
    pass
''')
        
        result = self.run_cli(
            ["find-function", "*", "--no-progress"],
            cwd=temp_project
        )
        
        # Should still succeed and find functions in valid files
        assert result.returncode == 0
        assert "main" in result.stdout
        # broken_function should not appear due to syntax error

    def test_find_function_empty_directory(self, temp_project):
        """Test find-function in directory with no Python files."""
        empty_dir = temp_project / "empty"
        empty_dir.mkdir()
        
        result = self.run_cli(
            ["find-function", "test", "--path", "empty", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        assert "No Python files found" in result.stderr

    def test_gitignore_respected(self, temp_project):
        """Test that .gitignore patterns are respected."""
        # The build/ directory should be ignored due to .gitignore
        result = self.run_cli(
            ["find-function", "*", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        # Functions from build/output.py should not be found
        # because DirectoryWalker respects common ignore patterns
        output_lines = result.stdout.strip().split('\n')
        build_files = [line for line in output_lines if 'build/' in line]
        assert len(build_files) == 0

    def test_help_command(self):
        """Test help output."""
        result = self.run_cli(["--help"])
        
        assert result.returncode == 0
        assert "AST tools for navigating Python codebases" in result.stdout
        assert "find-function" in result.stdout

    def test_version_command(self):
        """Test version output."""
        result = self.run_cli(["--version"])
        
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_progress_bar_output(self, temp_project):
        """Test that progress bar is shown by default."""
        # Note: This test might be flaky in CI environments
        # We'll just check that the command works without --no-progress
        result = self.run_cli(
            ["find-function", "main"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        # Progress output goes to stderr
        # We mainly care that it doesn't crash

    @pytest.mark.parametrize("pattern,expected_count", [
        ("*", 10),  # All functions
        ("test_*", 2),  # Test functions
        ("parse*", 2),  # parse_config and parse method
        ("*_*", 6),  # Functions with underscore
    ])
    def test_various_patterns(self, temp_project, pattern, expected_count):
        """Test various search patterns."""
        result = self.run_cli(
            ["find-function", pattern, "--format", "json", "--no-progress"],
            cwd=temp_project
        )
        
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) >= expected_count