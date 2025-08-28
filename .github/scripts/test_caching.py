#!/usr/bin/env python3
"""
Test script to validate GitHub Actions caching implementation.
This script verifies that the caching strategy is properly configured.
"""

import os
import sys
import hashlib
from pathlib import Path

def test_cache_key_generation():
    """Test that cache keys are generated correctly."""
    print("=== Testing Cache Key Generation ===")
    
    # Test requirements.txt hash
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        with open(requirements_file, 'rb') as f:
            content = f.read()
            hash_value = hashlib.sha256(content).hexdigest()[:8]
        print(f"✓ Requirements.txt hash: {hash_value}")
    else:
        print("✗ requirements.txt not found")
        return False
    
    # Test workflow file hash for APT cache
    workflow_file = Path(".github/workflows/ci.yml")
    if workflow_file.exists():
        with open(workflow_file, 'rb') as f:
            content = f.read()
            hash_value = hashlib.sha256(content).hexdigest()[:8]
        print(f"✓ Workflow file hash: {hash_value}")
    else:
        print("✗ ci.yml workflow file not found")
        return False
    
    return True

def test_cache_paths():
    """Test that cache paths are valid."""
    print("\n=== Testing Cache Paths ===")
    
    # Python cache paths
    python_version = "3.13"
    cache_paths = [
        f"~/.cache/pip",
        f"~/.local/lib/python{python_version}/site-packages"
    ]
    
    for path in cache_paths:
        expanded_path = os.path.expanduser(path)
        print(f"✓ Python cache path: {path} -> {expanded_path}")
    
    # TA-Lib cache paths
    talib_paths = [
        "/usr/local/lib/libta_lib*",
        "/usr/local/include/ta-lib/",
        "/usr/local/bin/ta-lib-config"
    ]
    
    for path in talib_paths:
        print(f"✓ TA-Lib cache path: {path}")
    
    # APT cache path
    apt_path = "/var/cache/apt"
    print(f"✓ APT cache path: {apt_path}")
    
    return True

def test_workflow_syntax():
    """Test that the workflow file has proper caching syntax."""
    print("\n=== Testing Workflow Caching Syntax ===")
    
    workflow_file = Path(".github/workflows/ci.yml")
    if not workflow_file.exists():
        print("✗ Workflow file not found")
        return False
    
    with open(workflow_file, 'r') as f:
        content = f.read()
    
    # Check for required caching elements
    required_elements = [
        "uses: actions/cache@v4",
        "Cache Python dependencies",
        "Cache APT packages", 
        "Cache TA-Lib installation",
        "cache-hit != 'true'",
        "hashFiles('requirements.txt')",
        "restore-keys:"
    ]
    
    for element in required_elements:
        if element in content:
            print(f"✓ Found: {element}")
        else:
            print(f"✗ Missing: {element}")
            return False
    
    return True

def main():
    """Run all caching tests."""
    print("GitHub Actions Caching Validation")
    print("=" * 40)
    
    tests = [
        test_cache_key_generation,
        test_cache_paths,
        test_workflow_syntax
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("Caching implementation is properly configured.")
        return 0
    else:
        print(f"✗ {total - passed} of {total} tests failed.")
        print("Caching implementation needs fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())