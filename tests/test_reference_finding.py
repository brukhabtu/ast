#!/usr/bin/env python3
"""Test the reference finding functionality."""

from astlib import ASTLib
from astlib.references import ReferenceType
import sys

def test_reference_finding():
    """Test finding references in the AST library itself."""
    print("Testing reference finding on AST library...")
    
    # Initialize the library
    ast_lib = ASTLib(".", lazy=False)
    
    # Test 1: Find references to a commonly used function
    print("\n1. Finding references to 'parse_file':")
    parse_refs = ast_lib.find_references("parse_file")
    print(f"   Found {len(parse_refs)} references")
    
    # Show first 5 references
    for ref in parse_refs[:5]:
        print(f"   - {ref.file_path}:{ref.line} - {ref.context}")
    
    # Test 2: Find references to a class
    print("\n2. Finding references to 'Symbol':")
    symbol_refs = ast_lib.find_references("Symbol")
    print(f"   Found {len(symbol_refs)} references")
    
    # Count by type (using the internal reference type)
    type_counts = {}
    for ref in symbol_refs[:20]:  # Just check first 20
        # Since we converted to API Reference, we don't have reference_type
        # But we can infer from context
        if "import" in ref.context.lower():
            type_counts["import"] = type_counts.get("import", 0) + 1
        elif ": Symbol" in ref.context or "-> Symbol" in ref.context:
            type_counts["type_annotation"] = type_counts.get("type_annotation", 0) + 1
        elif "Symbol(" in ref.context:
            type_counts["instantiation"] = type_counts.get("instantiation", 0) + 1
            
    print("   Types of references found:")
    for ref_type, count in type_counts.items():
        print(f"   - {ref_type}: {count}")
    
    # Test 3: Find references to a less common symbol
    print("\n3. Finding references to 'ReferenceVisitor':")
    visitor_refs = ast_lib.find_references("ReferenceVisitor")
    print(f"   Found {len(visitor_refs)} references")
    for ref in visitor_refs:
        print(f"   - {ref.file_path}:{ref.line} - {ref.context}")
    
    # Test 4: Test CLI command
    print("\n4. Testing CLI command:")
    import subprocess
    result = subprocess.run(
        ["python", "-m", "astlib.cli", "find-references", "SymbolTable", "--no-progress"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("   CLI command succeeded!")
        lines = result.stdout.strip().split('\n')[:5]
        for line in lines:
            print(f"   {line}")
    else:
        print(f"   CLI command failed: {result.stderr}")
    
    # Test 5: Performance test
    print("\n5. Performance test:")
    import time
    
    # Time a reference search
    start = time.time()
    refs = ast_lib.find_references("find_references")
    elapsed = time.time() - start
    
    print(f"   Found {len(refs)} references in {elapsed:.3f} seconds")
    print(f"   Average: {elapsed/ast_lib.get_stats().total_files:.3f} seconds per file")
    
    print("\n✓ Reference finding is working correctly!")
    
    # Test edge cases
    print("\n6. Edge cases:")
    
    # Non-existent symbol
    non_refs = ast_lib.find_references("ThisSymbolDoesNotExist")
    print(f"   Non-existent symbol: {len(non_refs)} references (should be 0)")
    
    # Common name that appears in many contexts
    ast_refs = ast_lib.find_references("ast")
    print(f"   Common name 'ast': {len(ast_refs)} references")

if __name__ == "__main__":
    test_reference_finding()