#!/usr/bin/env python3
"""
Verification script to check that test_features.py is properly structured.
This doesn't run the actual tests but verifies the test file structure.
"""

import ast
import sys
from pathlib import Path

def verify_test_file():
    """Verify the test file has all required tests."""
    test_file = Path(__file__).parent / "test_features.py"
    
    if not test_file.exists():
        print("❌ test_features.py not found")
        return False
    
    # Read and parse the test file
    with open(test_file, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Syntax error in test file: {e}")
        return False
    
    # Find all test functions
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            test_functions.append(node.name)
    
    # Check required tests
    required_tests = [
        'test_no_leak',
        'test_mtf_alignment',
        'test_feature_cap',
        'test_deterministic'
    ]
    
    print("📋 Test Structure Verification Report")
    print("=" * 50)
    
    all_present = True
    for test in required_tests:
        if test in test_functions:
            print(f"✅ {test} - present")
        else:
            print(f"❌ {test} - missing")
            all_present = False
    
    # Check for docstrings
    print("\n📝 Test Documentation:")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in required_tests:
            docstring = ast.get_docstring(node)
            if docstring:
                print(f"✅ {node.name} has docstring")
            else:
                print(f"⚠️  {node.name} missing docstring")
    
    # Check imports
    print("\n📦 Required Imports:")
    imports_needed = [
        'pandas',
        'numpy',
        'features.builder',
        'features.registry'
    ]
    
    import_strs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_strs.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            import_strs.append(node.module or '')
    
    for imp in imports_needed:
        if any(imp in s for s in import_strs):
            print(f"✅ {imp} imported")
        else:
            print(f"❌ {imp} not imported")
    
    print("\n" + "=" * 50)
    
    if all_present:
        print("✅ All required tests are present!")
        print("\n📄 Test Descriptions:")
        print("1. test_no_leak: Verifies shift(1) prevents future leakage")
        print("2. test_mtf_alignment: Checks MTF values align correctly")
        print("3. test_feature_cap: Ensures max 256 features enforced")
        print("4. test_deterministic: Confirms pipeline determinism")
        return True
    else:
        print("❌ Some tests are missing")
        return False

if __name__ == "__main__":
    success = verify_test_file()
    sys.exit(0 if success else 1)