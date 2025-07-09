#!/bin/bash
# Run tests for the AST symbol extraction component

echo "=== Running AST Symbol Extraction Tests ==="

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo -e "\n1. Running unit tests..."
python -m pytest tests/unit/test_symbols.py -v

echo -e "\n2. Running symbol table tests..."
python -m pytest tests/unit/test_symbol_table.py -v

echo -e "\n3. Running edge case tests..."
python -m pytest tests/unit/test_edge_cases.py -v

echo -e "\n4. Running integration tests..."
python -m pytest tests/integration/test_parser_symbols.py -v

echo -e "\n5. Running performance tests..."
python -m pytest tests/unit/test_symbol_table.py::TestSymbolTablePerformance -v

echo -e "\n6. Running demo..."
python examples/demo_symbol_extraction.py

echo -e "\n=== Test Summary ==="
python -m pytest tests/ --tb=short -q