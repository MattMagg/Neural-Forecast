#!/bin/bash
# Run feature engineering tests

echo "================================"
echo "Feature Engineering Test Suite"
echo "================================"
echo ""

# Check if we have pytest
if python3 -m pytest --version &> /dev/null; then
    echo "Running tests with pytest..."
    python3 -m pytest tests/test_features.py -v --tb=short
else
    echo "Running tests directly (pytest not available)..."
    python3 tests/test_features.py
fi

echo ""
echo "Test run complete!"