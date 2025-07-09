#!/usr/bin/env python3
"""Analyze test quality and categorization in the AST library."""

import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

# Define characteristics of each test level
UNIT_TEST_PATTERNS = {
    'mocking': ['mock', 'Mock', 'patch', '@patch'],
    'isolation': ['single function', 'one method', 'individual'],
    'no_io': ['no file access', 'no network', 'in-memory'],
    'fast': ['< 100ms', 'milliseconds', 'fast']
}

INTEGRATION_TEST_PATTERNS = {
    'multi_component': ['multiple modules', 'cross-module', 'integration'],
    'real_components': ['real implementation', 'actual', 'no mocks for internal'],
    'external_mocked': ['external mocked', 'mock external', 'patch third-party']
}

E2E_TEST_PATTERNS = {
    'full_workflow': ['end-to-end', 'complete workflow', 'full stack'],
    'no_mocks': ['no mocking', 'real services', 'actual files'],
    'subprocess': ['subprocess', 'command line', 'CLI invocation']
}

def analyze_test_file(file_path: Path) -> Dict[str, any]:
    """Analyze a single test file."""
    content = file_path.read_text()
    tree = ast.parse(content)
    
    results = {
        'file': str(file_path),
        'test_count': 0,
        'uses_mocks': False,
        'uses_fixtures': False,
        'uses_real_files': False,
        'uses_subprocess': False,
        'test_functions': [],
        'imports': [],
        'markers': [],
        'issues': []
    }
    
    # Count test functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            results['test_count'] += 1
            results['test_functions'].append(node.name)
            
        elif isinstance(node, ast.Import):
            for alias in node.names:
                results['imports'].append(alias.name)
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                results['imports'].append(node.module)
    
    # Check for mocking
    mock_patterns = ['mock', 'Mock', 'patch', 'MagicMock']
    results['uses_mocks'] = any(pattern in content for pattern in mock_patterns)
    
    # Check for fixtures
    results['uses_fixtures'] = '@pytest.fixture' in content or 'fixture' in content
    
    # Check for real file access
    file_patterns = ['open(', 'Path(', '.read_text()', '.write_text()', 'parse_file(']
    results['uses_real_files'] = any(pattern in content for pattern in file_patterns)
    
    # Check for subprocess
    results['uses_subprocess'] = 'subprocess' in content or 'run(' in content
    
    # Extract pytest markers
    marker_pattern = r'@pytest\.mark\.(\w+)'
    results['markers'] = re.findall(marker_pattern, content)
    
    return results

def categorize_test(analysis: Dict) -> str:
    """Categorize test based on its characteristics."""
    file_path = analysis['file']
    
    # E2E tests
    if 'e2e' in file_path or analysis['uses_subprocess']:
        return 'e2e'
    
    # Integration tests  
    if 'integration' in file_path:
        return 'integration'
        
    # Unit tests
    if 'unit' in file_path:
        return 'unit'
        
    # Guess based on characteristics
    if analysis['uses_mocks'] and not analysis['uses_real_files']:
        return 'unit'
    elif analysis['uses_real_files'] and not analysis['uses_mocks']:
        return 'integration'
    else:
        return 'unknown'

def check_test_quality(analysis: Dict, category: str) -> List[str]:
    """Check if test follows best practices for its category."""
    issues = []
    
    if category == 'unit':
        # Unit tests should mock dependencies
        if analysis['uses_real_files'] and 'mock' not in str(analysis['imports']):
            issues.append("Unit test uses real files without mocking")
            
        # Unit tests should be isolated
        if 'integration' in ' '.join(analysis['test_functions']):
            issues.append("Unit test name suggests integration testing")
            
    elif category == 'integration':
        # Integration tests should test multiple components
        if analysis['test_count'] < 2:
            issues.append("Integration test file has very few tests")
            
        # Should mock external dependencies
        imports = ' '.join(analysis['imports'])
        if 'requests' in imports and not analysis['uses_mocks']:
            issues.append("Integration test uses external services without mocking")
            
    elif category == 'e2e':
        # E2E tests should not use mocks
        if analysis['uses_mocks']:
            issues.append("E2E test uses mocking - should test real system")
    
    return issues

# Analyze all test files
print("=== AST Library Test Quality Analysis ===\n")

test_dirs = ['tests/unit', 'tests/integration', 'tests/e2e']
category_stats = defaultdict(lambda: {'count': 0, 'tests': 0, 'issues': []})

for test_dir in test_dirs:
    test_path = Path(test_dir)
    if not test_path.exists():
        continue
        
    print(f"\n## {test_dir.upper()} ##")
    
    for test_file in sorted(test_path.glob("**/test_*.py")):
        analysis = analyze_test_file(test_file)
        category = categorize_test(analysis)
        expected_category = test_dir.split('/')[-1]
        
        # Check quality
        issues = check_test_quality(analysis, expected_category)
        analysis['issues'] = issues
        
        # Update stats
        category_stats[expected_category]['count'] += 1
        category_stats[expected_category]['tests'] += analysis['test_count']
        category_stats[expected_category]['issues'].extend(issues)
        
        # Report
        relative_path = test_file.relative_to('tests')
        status = "✅" if not issues else "⚠️"
        
        print(f"{status} {relative_path}")
        print(f"   Tests: {analysis['test_count']}, " 
              f"Mocks: {'Yes' if analysis['uses_mocks'] else 'No'}, "
              f"Files: {'Yes' if analysis['uses_real_files'] else 'No'}, "
              f"Markers: {analysis['markers'] or 'None'}")
        
        if issues:
            for issue in issues:
                print(f"   ⚠️  {issue}")

# Summary
print("\n## SUMMARY ##")
print(f"Total test files analyzed: {sum(s['count'] for s in category_stats.values())}")
print(f"Total test functions: {sum(s['tests'] for s in category_stats.values())}")

print("\nTest Distribution:")
for category, stats in category_stats.items():
    percentage = (stats['tests'] / sum(s['tests'] for s in category_stats.values())) * 100
    print(f"  {category.capitalize()}: {stats['tests']} tests ({percentage:.1f}%)")

print("\nQuality Issues:")
total_issues = sum(len(s['issues']) for s in category_stats.values())
if total_issues == 0:
    print("  ✅ No quality issues found!")
else:
    print(f"  ⚠️  {total_issues} issues found")
    for category, stats in category_stats.items():
        if stats['issues']:
            print(f"\n  {category.capitalize()}:")
            for issue in set(stats['issues']):
                count = stats['issues'].count(issue)
                print(f"    - {issue} ({count}x)")

# Check specific test examples
print("\n## SPECIFIC TEST ANALYSIS ##")

# Check a unit test
print("\n### Example Unit Test Check ###")
unit_test = Path("tests/unit/test_parser.py")
if unit_test.exists():
    content = unit_test.read_text()
    print(f"Analyzing {unit_test}:")
    
    # Check for proper mocking
    if 'patch' in content or 'Mock' in content:
        print("  ✅ Uses mocking appropriately")
    else:
        print("  ⚠️  No mocking found - may be testing too much")
        
    # Check for fast tests
    if 'time.sleep' in content:
        print("  ⚠️  Contains sleep - unit tests should be instant")
    else:
        print("  ✅ No sleep statements found")
        
    # Check test isolation
    if 'self.' not in content or 'cls.' not in content:
        print("  ✅ Tests appear isolated (no shared state)")
    else:
        print("  ⚠️  May have shared state between tests")

# Check an integration test
print("\n### Example Integration Test Check ###")
integration_test = Path("tests/integration/test_cross_file_navigation.py")
if integration_test.exists():
    content = integration_test.read_text()
    print(f"Analyzing {integration_test}:")
    
    # Should test multiple components
    if 'ProjectIndex' in content and 'SymbolTable' in content:
        print("  ✅ Tests multiple components together")
    else:
        print("  ⚠️  May not be testing component integration")
        
    # Should use test fixtures/projects
    if 'test_projects' in content or 'fixtures' in content:
        print("  ✅ Uses test fixtures appropriately")
    else:
        print("  ⚠️  May not have proper test data")

# Check an E2E test
print("\n### Example E2E Test Check ###")
e2e_test = Path("tests/e2e/test_cli_commands.py")
if e2e_test.exists():
    content = e2e_test.read_text()
    print(f"Analyzing {e2e_test}:")
    
    # Should use subprocess for CLI testing
    if 'subprocess' in content or 'run' in content:
        print("  ✅ Tests actual CLI invocation")
    else:
        print("  ⚠️  May not be testing real CLI")
        
    # Should not mock
    if 'mock' not in content.lower():
        print("  ✅ No mocking (tests real system)")
    else:
        print("  ⚠️  Contains mocking - E2E should test real system")

print("\n=== Analysis Complete ===")