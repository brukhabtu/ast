"""Mock repository structures for testing without downloading actual repos."""

from pathlib import Path
from typing import Dict, List


def create_mock_python_file(path: Path, content: str = None) -> None:
    """Create a mock Python file with default content if not provided."""
    if content is None:
        content = '''"""Mock module for testing."""

def hello_world():
    """Simple test function."""
    return "Hello, World!"


class TestClass:
    """Simple test class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello from {self.name}"
'''
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def create_mock_repo(base_path: Path, name: str, structure: Dict[str, str]) -> Path:
    """Create a mock repository structure.
    
    Args:
        base_path: Base directory for the mock repo
        name: Repository name
        structure: Dict mapping file paths to content
        
    Returns:
        Path to the created repository
    """
    repo_path = base_path / name.lower()
    repo_path.mkdir(parents=True, exist_ok=True)
    
    for file_path, content in structure.items():
        full_path = repo_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    return repo_path


def create_mini_django_structure(base_path: Path) -> Path:
    """Create a minimal Django-like structure for testing."""
    structure = {
        "django/__init__.py": "",
        "django/core/__init__.py": "",
        "django/core/management/__init__.py": "",
        "django/core/management/base.py": '''"""Django management base."""

class BaseCommand:
    """Base command class."""
    
    def handle(self, *args, **options):
        """Handle command execution."""
        raise NotImplementedError("Subclasses must implement handle()")
''',
        "django/db/__init__.py": "",
        "django/db/models/__init__.py": "",
        "django/db/models/base.py": '''"""Django models base."""

class Model:
    """Base model class."""
    
    objects = None  # Manager instance
    
    def save(self, **kwargs):
        """Save the model instance."""
        pass
''',
        "django/http/__init__.py": "",
        "django/http/response.py": '''"""Django HTTP responses."""

class HttpResponse:
    """HTTP response class."""
    
    def __init__(self, content="", content_type=None, status=200):
        self.content = content
        self.content_type = content_type
        self.status_code = status
''',
    }
    
    return create_mock_repo(base_path, "django", structure)


def create_mini_flask_structure(base_path: Path) -> Path:
    """Create a minimal Flask-like structure for testing."""
    structure = {
        "flask/__init__.py": '''"""Flask web framework."""

from .app import Flask

__version__ = "3.0.0"
''',
        "flask/app.py": '''"""Flask application."""

class Flask:
    """The Flask application class."""
    
    def __init__(self, import_name, **kwargs):
        self.import_name = import_name
        self.config = {}
    
    def route(self, rule, **options):
        """Decorator to register a route."""
        def decorator(f):
            return f
        return decorator
''',
        "flask/blueprints.py": '''"""Flask blueprints."""

class Blueprint:
    """Blueprint for organizing application."""
    
    def __init__(self, name, import_name):
        self.name = name
        self.import_name = import_name
''',
    }
    
    return create_mock_repo(base_path, "flask", structure)


def create_mini_fastapi_structure(base_path: Path) -> Path:
    """Create a minimal FastAPI-like structure for testing."""
    structure = {
        "fastapi/__init__.py": '''"""FastAPI framework."""

from .applications import FastAPI

__version__ = "0.100.0"
''',
        "fastapi/applications.py": '''"""FastAPI application."""

from typing import Callable

class FastAPI:
    """FastAPI application class."""
    
    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "FastAPI")
    
    def get(self, path: str):
        """GET route decorator."""
        def decorator(func: Callable):
            return func
        return decorator
    
    def post(self, path: str):
        """POST route decorator."""
        def decorator(func: Callable):
            return func
        return decorator
''',
        "fastapi/routing.py": '''"""FastAPI routing."""

class APIRouter:
    """API router for organizing routes."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
''',
    }
    
    return create_mock_repo(base_path, "fastapi", structure)


def create_mini_requests_structure(base_path: Path) -> Path:
    """Create a minimal Requests-like structure for testing."""
    structure = {
        "requests/__init__.py": '''"""Requests HTTP library."""

from .api import get, post, put, delete

__version__ = "2.31.0"
''',
        "requests/api.py": '''"""Requests API."""

def get(url, **kwargs):
    """Send a GET request."""
    return request("GET", url, **kwargs)

def post(url, data=None, json=None, **kwargs):
    """Send a POST request."""
    return request("POST", url, data=data, json=json, **kwargs)

def put(url, data=None, **kwargs):
    """Send a PUT request."""
    return request("PUT", url, data=data, **kwargs)

def delete(url, **kwargs):
    """Send a DELETE request."""
    return request("DELETE", url, **kwargs)

def request(method, url, **kwargs):
    """Send an HTTP request."""
    from .models import Response
    return Response()
''',
        "requests/models.py": '''"""Requests models."""

class Response:
    """HTTP response object."""
    
    def __init__(self):
        self.status_code = 200
        self.headers = {}
        self.text = ""
        self._content = b""
    
    @property
    def content(self):
        """Response content in bytes."""
        return self._content
    
    def json(self):
        """Parse response as JSON."""
        import json
        return json.loads(self.text)
''',
    }
    
    return create_mock_repo(base_path, "requests", structure)


def create_all_mock_repos(base_path: Path) -> List[Path]:
    """Create all mock repositories for testing.
    
    Args:
        base_path: Base directory for mock repositories
        
    Returns:
        List of paths to created repositories
    """
    repos = []
    
    repos.append(create_mini_django_structure(base_path))
    repos.append(create_mini_flask_structure(base_path))
    repos.append(create_mini_fastapi_structure(base_path))
    repos.append(create_mini_requests_structure(base_path))
    
    # Create a README in the base path
    readme_path = base_path / "README.md"
    readme_path.write_text("""# Mock Test Repositories

These are minimal mock implementations of popular Python frameworks for testing.
They contain just enough structure to test AST parsing and analysis without
downloading the full repositories.

## Repositories

- **django/**: Minimal Django structure
- **flask/**: Minimal Flask structure  
- **fastapi/**: Minimal FastAPI structure
- **requests/**: Minimal Requests structure

These mock repos are used in unit tests to avoid network dependencies and
provide fast, reliable test execution.
""")
    
    return repos