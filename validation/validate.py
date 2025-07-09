#!/usr/bin/env python3
"""
Large-Scale Validation System for AST Library

Tests the AST library against popular Python repositories to identify:
- Compatibility issues across Python versions
- Edge cases and failure patterns
- Performance characteristics at scale
- Real-world usage patterns
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Popular Python repositories for testing
TEST_REPOSITORIES = [
    # Core Python ecosystem
    {"name": "requests", "url": "https://github.com/psf/requests.git", "stars": 51000},
    {"name": "django", "url": "https://github.com/django/django.git", "stars": 77000},
    {"name": "flask", "url": "https://github.com/pallets/flask.git", "stars": 66000},
    {"name": "numpy", "url": "https://github.com/numpy/numpy.git", "stars": 26000},
    {"name": "pandas", "url": "https://github.com/pandas-dev/pandas.git", "stars": 42000},
    
    # Modern Python projects
    {"name": "fastapi", "url": "https://github.com/tiangolo/fastapi.git", "stars": 72000},
    {"name": "black", "url": "https://github.com/psf/black.git", "stars": 37000},
    {"name": "poetry", "url": "https://github.com/python-poetry/poetry.git", "stars": 29000},
    {"name": "httpx", "url": "https://github.com/encode/httpx.git", "stars": 12000},
    {"name": "pydantic", "url": "https://github.com/pydantic/pydantic.git", "stars": 19000},
    
    # Large codebases for scale testing
    {"name": "home-assistant", "url": "https://github.com/home-assistant/core.git", "stars": 69000},
    {"name": "scikit-learn", "url": "https://github.com/scikit-learn/scikit-learn.git", "stars": 58000},
    {"name": "transformers", "url": "https://github.com/huggingface/transformers.git", "stars": 125000},
]

@dataclass
class ValidationResult:
    """Results from validating a single repository"""
    repo_name: str
    repo_url: str
    success: bool
    python_version: str
    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    parse_time_ms: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class FailureCase:
    """Detailed information about a parsing failure"""
    repo_name: str
    file_path: str
    python_version: str
    error_type: str
    error_message: str
    code_snippet: Optional[str] = None
    line_number: Optional[int] = None
    traceback: Optional[str] = None

class RepositoryValidator:
    """Validates AST library against a Python repository"""
    
    def __init__(self, ast_lib_path: Path, output_dir: Path):
        self.ast_lib_path = ast_lib_path
        self.output_dir = output_dir
        self.failure_cases_dir = output_dir / "failure_cases"
        self.failure_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Import AST library (when available)
        sys.path.insert(0, str(ast_lib_path))
        
    def clone_repository(self, repo_url: str, target_dir: Path) -> bool:
        """Clone a repository for testing"""
        try:
            logger.info(f"Cloning {repo_url} to {target_dir}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {repo_url}: {e}")
            return False
    
    def find_python_files(self, repo_path: Path) -> List[Path]:
        """Find all Python files in a repository"""
        python_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Skip test files for initial validation
                    if 'test' not in str(file_path).lower():
                        python_files.append(file_path)
        return python_files
    
    def detect_python_version(self, repo_path: Path) -> str:
        """Detect Python version used in repository"""
        # Check for pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, 'rb') as f:
                    data = tomllib.load(f)
                    python_req = data.get('tool', {}).get('poetry', {}).get('dependencies', {}).get('python', '')
                    if python_req:
                        return python_req
            except Exception:
                pass
        
        # Check setup.py
        setup_py = repo_path / "setup.py"
        if setup_py.exists():
            try:
                content = setup_py.read_text()
                if 'python_requires' in content:
                    import re
                    match = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
            except Exception:
                pass
        
        # Default to current Python version
        return f">={sys.version_info.major}.{sys.version_info.minor}"
    
    def validate_file(self, file_path: Path, repo_name: str) -> Tuple[bool, Optional[FailureCase]]:
        """Validate a single Python file"""
        try:
            # This is where we'll use the AST library once it's available
            # For now, we'll use the standard library AST as a placeholder
            import ast
            
            content = file_path.read_text(encoding='utf-8')
            ast.parse(content, filename=str(file_path))
            
            # TODO: Replace with actual AST library validation
            # ast_lib.parse_file(file_path)
            # ast_lib.extract_symbols(file_path)
            # ast_lib.analyze_imports(file_path)
            
            return True, None
            
        except SyntaxError as e:
            failure = FailureCase(
                repo_name=repo_name,
                file_path=str(file_path),
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                error_type="SyntaxError",
                error_message=str(e),
                line_number=e.lineno,
                code_snippet=self._extract_code_snippet(file_path, e.lineno) if e.lineno else None
            )
            return False, failure
            
        except Exception as e:
            failure = FailureCase(
                repo_name=repo_name,
                file_path=str(file_path),
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=self._get_traceback()
            )
            return False, failure
    
    def _extract_code_snippet(self, file_path: Path, line_number: int, context: int = 3) -> str:
        """Extract code snippet around error line"""
        try:
            lines = file_path.read_text().splitlines()
            start = max(0, line_number - context - 1)
            end = min(len(lines), line_number + context)
            
            snippet_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
            
            return "\n".join(snippet_lines)
        except Exception:
            return ""
    
    def _get_traceback(self) -> str:
        """Get current exception traceback"""
        import traceback
        return traceback.format_exc()
    
    def validate_repository(self, repo_info: Dict[str, Any]) -> ValidationResult:
        """Validate entire repository"""
        repo_name = repo_info["name"]
        repo_url = repo_info["url"]
        
        logger.info(f"Validating {repo_name}")
        
        result = ValidationResult(
            repo_name=repo_name,
            repo_url=repo_url,
            success=True,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / repo_name
            
            # Clone repository
            if not self.clone_repository(repo_url, repo_path):
                result.success = False
                result.errors.append({"type": "CloneError", "message": "Failed to clone repository"})
                return result
            
            # Detect Python version
            result.python_version = self.detect_python_version(repo_path)
            
            # Find Python files
            python_files = self.find_python_files(repo_path)
            result.total_files = len(python_files)
            
            # Validate each file
            start_time = time.time()
            failures = []
            
            for file_path in python_files:
                success, failure_case = self.validate_file(file_path, repo_name)
                if success:
                    result.parsed_files += 1
                else:
                    result.failed_files += 1
                    if failure_case:
                        failures.append(failure_case)
                        result.errors.append({
                            "file": str(file_path.relative_to(repo_path)),
                            "error_type": failure_case.error_type,
                            "error_message": failure_case.error_message
                        })
            
            result.parse_time_ms = (time.time() - start_time) * 1000
            
            # Calculate metrics
            result.metrics = {
                "success_rate": result.parsed_files / result.total_files if result.total_files > 0 else 0,
                "avg_parse_time_ms": result.parse_time_ms / result.total_files if result.total_files > 0 else 0,
                "files_per_second": result.total_files / (result.parse_time_ms / 1000) if result.parse_time_ms > 0 else 0
            }
            
            # Save failure cases
            if failures:
                self._save_failure_cases(repo_name, failures)
                result.success = result.failed_files == 0
            
        return result
    
    def _save_failure_cases(self, repo_name: str, failures: List[FailureCase]):
        """Save failure cases for analysis"""
        repo_failures_dir = self.failure_cases_dir / repo_name
        repo_failures_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary_file = repo_failures_dir / "failures_summary.json"
        summary_data = [asdict(f) for f in failures]
        summary_file.write_text(json.dumps(summary_data, indent=2))
        
        # Save individual failure examples
        for i, failure in enumerate(failures[:10]):  # Save first 10 examples
            example_file = repo_failures_dir / f"example_{i+1}.md"
            example_content = f"""# Failure Example {i+1}

**Repository**: {failure.repo_name}
**File**: {failure.file_path}
**Python Version**: {failure.python_version}
**Error Type**: {failure.error_type}
**Error Message**: {failure.error_message}

## Code Snippet
```python
{failure.code_snippet or "N/A"}
```

## Traceback
```
{failure.traceback or "N/A"}
```
"""
            example_file.write_text(example_content)

class ValidationHarness:
    """Main validation harness for running large-scale tests"""
    
    def __init__(self, ast_lib_path: Path, output_dir: Path):
        self.ast_lib_path = ast_lib_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = RepositoryValidator(ast_lib_path, output_dir)
        self.results: List[ValidationResult] = []
    
    def run_validation(self, repositories: Optional[List[Dict[str, Any]]] = None):
        """Run validation on all repositories"""
        repos_to_test = repositories or TEST_REPOSITORIES
        
        logger.info(f"Starting validation on {len(repos_to_test)} repositories")
        
        for repo_info in repos_to_test:
            try:
                result = self.validator.validate_repository(repo_info)
                self.results.append(result)
                self._save_intermediate_results()
            except Exception as e:
                logger.error(f"Failed to validate {repo_info['name']}: {e}")
                self.results.append(ValidationResult(
                    repo_name=repo_info["name"],
                    repo_url=repo_info["url"],
                    success=False,
                    python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                    errors=[{"type": "ValidationError", "message": str(e)}]
                ))
    
    def _save_intermediate_results(self):
        """Save results after each repository"""
        results_file = self.output_dir / "validation_results.json"
        results_data = [asdict(r) for r in self.results]
        results_file.write_text(json.dumps(results_data, indent=2))
    
    def generate_compatibility_matrix(self):
        """Generate compatibility matrix from results"""
        matrix_data = {
            "python_versions": {},
            "feature_support": {},
            "repository_results": []
        }
        
        for result in self.results:
            # Repository summary
            repo_summary = {
                "name": result.repo_name,
                "success_rate": result.metrics.get("success_rate", 0),
                "total_files": result.total_files,
                "failed_files": result.failed_files,
                "parse_time_ms": result.parse_time_ms
            }
            matrix_data["repository_results"].append(repo_summary)
            
            # Python version compatibility
            py_version = result.python_version
            if py_version not in matrix_data["python_versions"]:
                matrix_data["python_versions"][py_version] = {
                    "repos": [],
                    "total_files": 0,
                    "success_files": 0
                }
            
            matrix_data["python_versions"][py_version]["repos"].append(result.repo_name)
            matrix_data["python_versions"][py_version]["total_files"] += result.total_files
            matrix_data["python_versions"][py_version]["success_files"] += result.parsed_files
        
        # Save compatibility matrix
        matrix_file = self.output_dir / "compatibility_matrix.json"
        matrix_file.write_text(json.dumps(matrix_data, indent=2))
        
        return matrix_data
    
    def generate_statistics_report(self):
        """Generate statistical analysis report"""
        stats = {
            "total_repositories": len(self.results),
            "total_files_analyzed": sum(r.total_files for r in self.results),
            "total_files_parsed": sum(r.parsed_files for r in self.results),
            "total_files_failed": sum(r.failed_files for r in self.results),
            "overall_success_rate": 0.0,
            "performance_metrics": {},
            "error_distribution": {},
            "repository_rankings": []
        }
        
        # Calculate overall success rate
        if stats["total_files_analyzed"] > 0:
            stats["overall_success_rate"] = stats["total_files_parsed"] / stats["total_files_analyzed"]
        
        # Performance metrics
        parse_times = [r.parse_time_ms for r in self.results if r.parse_time_ms > 0]
        if parse_times:
            stats["performance_metrics"] = {
                "avg_parse_time_ms": sum(parse_times) / len(parse_times),
                "min_parse_time_ms": min(parse_times),
                "max_parse_time_ms": max(parse_times),
                "total_parse_time_s": sum(parse_times) / 1000
            }
        
        # Error distribution
        for result in self.results:
            for error in result.errors:
                error_type = error.get("error_type", "Unknown")
                if error_type not in stats["error_distribution"]:
                    stats["error_distribution"][error_type] = 0
                stats["error_distribution"][error_type] += 1
        
        # Repository rankings
        repo_rankings = []
        for result in self.results:
            repo_rankings.append({
                "name": result.repo_name,
                "success_rate": result.metrics.get("success_rate", 0),
                "total_files": result.total_files,
                "performance": result.metrics.get("files_per_second", 0)
            })
        
        stats["repository_rankings"] = sorted(
            repo_rankings,
            key=lambda x: x["success_rate"],
            reverse=True
        )
        
        # Save statistics
        stats_file = self.output_dir / "statistics_report.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        
        return stats
    
    def generate_markdown_report(self):
        """Generate human-readable markdown report"""
        stats = self.generate_statistics_report()
        matrix = self.generate_compatibility_matrix()
        
        report_content = f"""# AST Library Validation Report

Generated: {datetime.now().isoformat()}

## Executive Summary

- **Repositories Tested**: {stats['total_repositories']}
- **Total Files Analyzed**: {stats['total_files_analyzed']:,}
- **Successfully Parsed**: {stats['total_files_parsed']:,} ({stats['overall_success_rate']:.1%})
- **Failed to Parse**: {stats['total_files_failed']:,}

## Performance Metrics

- **Average Parse Time**: {stats['performance_metrics'].get('avg_parse_time_ms', 0):.2f}ms per repository
- **Total Processing Time**: {stats['performance_metrics'].get('total_parse_time_s', 0):.2f} seconds

## Repository Success Rates

| Repository | Success Rate | Files Analyzed | Performance (files/sec) |
|------------|--------------|----------------|------------------------|
"""
        
        for repo in stats['repository_rankings']:
            report_content += f"| {repo['name']} | {repo['success_rate']:.1%} | {repo['total_files']} | {repo['performance']:.1f} |\n"
        
        report_content += f"""
## Error Distribution

| Error Type | Count |
|------------|-------|
"""
        
        for error_type, count in sorted(stats['error_distribution'].items(), key=lambda x: x[1], reverse=True):
            report_content += f"| {error_type} | {count} |\n"
        
        report_content += f"""
## Python Version Compatibility

| Python Version | Repositories | Success Rate |
|----------------|--------------|--------------|
"""
        
        for version, data in matrix['python_versions'].items():
            success_rate = data['success_files'] / data['total_files'] if data['total_files'] > 0 else 0
            report_content += f"| {version} | {len(data['repos'])} | {success_rate:.1%} |\n"
        
        report_content += """
## Recommendations

Based on the validation results:

1. **Priority Error Fixes**: Focus on the most common error types identified
2. **Performance Optimization**: Target repositories with lowest files/sec rates
3. **Compatibility**: Ensure support for Python version ranges used by tested repositories
4. **Edge Cases**: Review failure cases in the `failure_cases/` directory

## Next Steps

1. Analyze specific failure patterns in each repository
2. Create regression tests from identified edge cases
3. Implement fixes for common error types
4. Re-run validation to measure improvements
"""
        
        # Save report
        report_file = self.output_dir / "validation_report.md"
        report_file.write_text(report_content)
        
        logger.info(f"Generated validation report: {report_file}")

def main():
    """Main entry point for validation script"""
    parser = argparse.ArgumentParser(description="Validate AST library on real Python codebases")
    parser.add_argument(
        "--ast-lib-path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Path to AST library"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for results"
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        help="Specific repository URLs to test"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation on subset of repositories"
    )
    
    args = parser.parse_args()
    
    # Create harness
    harness = ValidationHarness(args.ast_lib_path, args.output_dir)
    
    # Determine repositories to test
    if args.repos:
        repos = [{"name": urlparse(url).path.split('/')[-1].replace('.git', ''), "url": url} for url in args.repos]
    elif args.quick:
        repos = TEST_REPOSITORIES[:3]  # Test first 3 repositories
    else:
        repos = TEST_REPOSITORIES
    
    # Run validation
    harness.run_validation(repos)
    
    # Generate reports
    harness.generate_compatibility_matrix()
    harness.generate_markdown_report()
    
    logger.info("Validation complete!")

if __name__ == "__main__":
    main()