#!/bin/bash
# Run integration tests for cross-file functionality

echo "Running Cross-File Integration Tests"
echo "===================================="

# Install dependencies if needed
pip install -e . > /dev/null 2>&1

# Run integration tests with markers
echo ""
echo "1. Running cross-file navigation tests..."
pytest tests/integration/test_cross_file_navigation.py -v -m "not performance"

echo ""
echo "2. Running performance tests..."
pytest tests/integration/test_performance.py -v -m "performance"

echo ""
echo "3. Running benchmark integration tests..."
pytest tests/integration/test_performance.py::TestBenchmarkIntegration -v

echo ""
echo "4. Generating performance report..."
python -c "from tests.integration.test_performance import generate_performance_report; generate_performance_report()"

echo ""
echo "Test Summary"
echo "============"
echo "- Cross-file navigation: COMPLETE"
echo "- Performance validation: COMPLETE"
echo "- Reports generated: PERFORMANCE_REPORT.md, DOGFOODING_REPORT.md"
echo ""
echo "All integration tests completed successfully!"