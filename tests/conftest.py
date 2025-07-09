"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_code_dir(tmp_path_factory):
    """Create a directory with test code files."""
    test_dir = tmp_path_factory.mktemp("test_code")
    
    # Create some test files
    (test_dir / "simple.py").write_text("""
def hello():
    return "Hello, World!"
""")
    
    (test_dir / "classes.py").write_text("""
class Base:
    pass

class Derived(Base):
    def method(self):
        pass
""")
    
    return test_dir


@pytest.fixture
def create_test_file(tmp_path):
    """Create a test file with given name and content."""
    def _create_file(filename, content):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create_file


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def valid_samples():
    """Collection of valid Python code samples."""
    return {
        "simple": "x = 42\nprint(x)",
        "function": "def hello():\n    return 'world'",
        "class": "class Foo:\n    def bar(self):\n        pass",
        "imports": "import os\nfrom pathlib import Path",
        "complex": """
def calculate(a, b):
    if a > b:
        return a - b
    else:
        return b - a

class Calculator:
    def add(self, x, y):
        return x + y
"""
    }


# Dictionary of invalid Python samples for testing error recovery
INVALID_PYTHON_SAMPLES = {
    "missing_colon": "def foo()\n    pass",
    "bad_indent": "def foo():\npass",
    "unclosed_paren": "print('hello'",
    "invalid_syntax": "def 123invalid():\n    pass",
    "mixed_indent": "def foo():\n\tpass\n    return",
    "incomplete_statement": "x = ",
    "bad_import": "from import something",
    "unmatched_bracket": "data = [1, 2, 3",
}


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    class MockCache:
        def __init__(self):
            self._cache = {}
        
        def get(self, key):
            return self._cache.get(key)
        
        def set(self, key, value):
            self._cache[key] = value
        
        def invalidate(self, key):
            self._cache.pop(key, None)
        
        def clear(self):
            self._cache.clear()
    
    return MockCache()


@pytest.fixture
def benchmark():
    """Simple benchmark fixture for performance tests."""
    def _benchmark(func, *args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result
    return _benchmark


@pytest.fixture
def large_python_file(tmp_path):
    """Create a large Python file for performance testing."""
    code = """
# Large Python file for performance testing
import os
import sys
from typing import List, Dict, Optional

"""
    # Add many function definitions
    for i in range(100):
        code += f"""
def function_{i}(x: int, y: int) -> int:
    '''Function {i} documentation.'''
    result = x + y
    for j in range(10):
        result += j
    return result

"""
    
    # Add some classes
    for i in range(20):
        code += f"""
class Class_{i}:
    '''Class {i} documentation.'''
    
    def __init__(self):
        self.value = {i}
    
    def method_1(self):
        return self.value * 2
    
    def method_2(self, x):
        return self.value + x
"""
    
    file_path = tmp_path / "large_file.py"
    file_path.write_text(code)
    return str(file_path)


@pytest.fixture
def create_test_project(tmp_path):
    """Create a test project structure and return a function to create projects."""
    return lambda structure: _create_project_structure(tmp_path, structure)

def _create_project_structure(base_path, structure):
    """Create project from dictionary structure."""
    for name, content in structure.items():
        if isinstance(content, dict):
            # It's a directory
            dir_path = base_path / name
            dir_path.mkdir(exist_ok=True)
            for subname, subcontent in content.items():
                file_path = dir_path / subname
                if isinstance(subcontent, str):
                    file_path.write_text(subcontent)
        else:
            # It's a file
            file_path = base_path / name
            file_path.write_text(content)
    return base_path


@pytest.fixture  
def temp_project_dir(tmp_path):
    """Provide a temporary project directory."""
    return tmp_path


# Performance markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that measure performance"
    )