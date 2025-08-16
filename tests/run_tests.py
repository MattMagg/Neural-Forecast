#!/usr/bin/env python3
"""
Test runner for CV and metrics test suites.

This script runs both unit and integration tests with coverage reporting.
"""

import sys
import subprocess
from pathlib import Path

def run_tests(test_type='all', verbose=True, coverage=True):
    """
    Run CV and metrics tests.
    
    Args:
        test_type: 'unit', 'integration', or 'all'
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    test_dir = Path(__file__).parent
    
    # Determine which tests to run
    if test_type == 'unit':
        test_files = ['test_cv.py']
    elif test_type == 'integration':
        test_files = ['test_cv_integration.py']
    elif test_type == 'foundation':
        test_files = ['test_cv_foundation.py']
    else:  # 'all'
        test_files = ['test_cv.py', 'test_cv_integration.py', 'test_cv_foundation.py']
    
    # Build pytest command
    cmd = ['pytest']
    
    # Add test files
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))
    
    # Add flags
    if verbose:
        cmd.append('-v')
    cmd.append('--tb=short')
    
    if coverage:
        cmd.extend([
            '--cov=cv',
            '--cov=uq',
            '--cov=utils',
            '--cov-report=term-missing',
            '--cov-report=html:tests/coverage_report'
        ])
    
    # Add color output
    cmd.append('--color=yes')
    
    print(f"Running command: {' '.join(cmd)}\n")
    
    # Run tests
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode


def run_quick_tests():
    """Run quick smoke tests for CI/CD."""
    print("=" * 60)
    print("RUNNING QUICK SMOKE TESTS")
    print("=" * 60)
    
    # Run only foundation tests (fastest)
    return run_tests('foundation', verbose=False, coverage=False)


def run_full_tests():
    """Run full test suite with coverage."""
    print("=" * 60)
    print("RUNNING FULL TEST SUITE WITH COVERAGE")
    print("=" * 60)
    
    return run_tests('all', verbose=True, coverage=True)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CV and metrics tests')
    parser.add_argument(
        '--type',
        choices=['unit', 'integration', 'foundation', 'all', 'quick', 'full'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--no-coverage',
        action='store_true',
        help='Disable coverage reporting'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Handle special test types
    if args.type == 'quick':
        return run_quick_tests()
    elif args.type == 'full':
        return run_full_tests()
    else:
        # Run specified tests
        return run_tests(
            test_type=args.type,
            verbose=not args.quiet,
            coverage=not args.no_coverage
        )


if __name__ == '__main__':
    sys.exit(main())