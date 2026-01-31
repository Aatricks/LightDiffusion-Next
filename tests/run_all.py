#!/usr/bin/env python
"""
Test runner script for LightDiffusion-Next test suite.

This script discovers and runs all tests in the tests/ directory using pytest.
It supports running unit tests, integration tests, or both.

Usage:
    python tests/run_all.py              # Run all tests
    python tests/run_all.py --unit       # Run only unit tests
    python tests/run_all.py --integration # Run only integration tests
    python tests/run_all.py --coverage   # Run with coverage report
    python tests/run_all.py -v           # Verbose output
    python tests/run_all.py -k model     # Run tests matching 'model'
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Ensure we're in the project root for imports
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))


def check_pytest_installed():
    """Check if pytest is installed, install if not."""
    try:
        import pytest
        return True
    except ImportError:
        print("pytest not found. Installing pytest...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        return True


def run_tests(
    unit: bool = True,
    integration: bool = True,
    coverage: bool = False,
    verbose: bool = False,
    pattern: str = None,
    extra_args: list = None,
) -> int:
    """
    Run tests using pytest.
    
    Args:
        unit: Run unit tests
        integration: Run integration tests  
        coverage: Generate coverage report
        verbose: Verbose output
        pattern: Pattern to filter tests (-k option)
        extra_args: Additional pytest arguments
        
    Returns:
        Exit code from pytest
    """
    import pytest
    
    # Build pytest arguments
    args = []
    
    # Determine which test directories to run
    test_dirs = []
    if unit:
        unit_dir = project_root / "tests" / "unit"
        if unit_dir.exists():
            test_dirs.append(str(unit_dir))
    if integration:
        int_dir = project_root / "tests" / "integration"
        if int_dir.exists():
            test_dirs.append(str(int_dir))
    
    if not test_dirs:
        # Default to all tests
        test_dirs.append(str(project_root / "tests"))
    
    args.extend(test_dirs)
    
    # Add verbose flag
    if verbose:
        args.append("-v")
    else:
        args.append("-q")
    
    # Add pattern filter
    if pattern:
        args.extend(["-k", pattern])
    
    # Add coverage options
    if coverage:
        args.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
        ])
    
    # Show short test summary
    args.append("-rsfE")
    
    # Show local variables in tracebacks
    args.append("--tb=short")
    
    # Add any extra arguments
    if extra_args:
        args.extend(extra_args)
    
    print(f"\n{'=' * 60}")
    print(f"Running LightDiffusion-Next Test Suite")
    print(f"{'=' * 60}")
    print(f"Test directories: {', '.join(test_dirs)}")
    print(f"Options: unit={unit}, integration={integration}, coverage={coverage}")
    print(f"Command: pytest {' '.join(args)}")
    print(f"{'=' * 60}\n")
    
    # Run pytest
    return pytest.main(args)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LightDiffusion-Next test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_all.py                 # Run all tests
    python tests/run_all.py --unit          # Run only unit tests
    python tests/run_all.py --integration   # Run only integration tests
    python tests/run_all.py --coverage      # Run with coverage report
    python tests/run_all.py -v              # Verbose output
    python tests/run_all.py -k model        # Run tests matching 'model'
    python tests/run_all.py --unit -v -k detect  # Unit tests matching 'detect', verbose
        """,
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (tests/unit/)",
    )
    parser.add_argument(
        "--integration",
        action="store_true", 
        help="Run only integration tests (tests/integration/)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-k", "--pattern",
        type=str,
        default=None,
        help="Only run tests matching this pattern",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests without running them",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Additional pytest arguments",
    )
    
    args = parser.parse_args()
    
    # Check pytest is available
    if not check_pytest_installed():
        print("ERROR: pytest is required but could not be installed")
        sys.exit(1)
    
    # Handle --list option
    if args.list:
        import pytest
        pytest.main(["--collect-only", "-q", str(project_root / "tests")])
        sys.exit(0)
    
    # Determine which tests to run
    run_unit = args.unit or (not args.unit and not args.integration)
    run_integration = args.integration or (not args.unit and not args.integration)
    
    # Run tests
    exit_code = run_tests(
        unit=run_unit,
        integration=run_integration,
        coverage=args.coverage,
        verbose=args.verbose,
        pattern=args.pattern,
        extra_args=args.extra,
    )
    
    # Print summary
    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests finished with exit code {exit_code}")
    print(f"{'=' * 60}\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
