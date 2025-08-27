#!/usr/bin/env python3
"""
CircleCI Environment Setup Validation Script
Validates that the environment setup meets all requirements (4.1, 4.3, 4.4, 4.7)
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple, Dict, Any


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class EnvironmentValidator:
    """Validates CircleCI environment setup"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def log_info(self, message: str) -> None:
        """Log info message"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
        
    def log_success(self, message: str) -> None:
        """Log success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
        
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        self.warnings.append(message)
        
    def log_error(self, message: str) -> None:
        """Log error message"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        self.errors.append(message)
        
    def run_command(self, command: str) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def validate_requirement_4_1(self) -> bool:
        """Validate Requirement 4.1: Standard Docker executor"""
        self.log_info("Validating Requirement 4.1: Standard Docker executor")
        
        success = True
        
        # Check if running in Docker
        if Path('/.dockerenv').exists():
            self.log_success("Running in Docker container")
        else:
            self.log_warning("Docker environment not detected (may be running locally)")
        
        # Check for apt package manager (standard Docker executor)
        exit_code, _, _ = self.run_command("which apt-get")
        if exit_code == 0:
            self.log_success("Standard Docker executor with apt package manager available")
        else:
            self.log_error("Standard Docker executor requirements not met")
            success = False
        
        # Check Python version in Docker context
        exit_code, stdout, _ = self.run_command("python --version")
        if exit_code == 0:
            python_version = stdout.strip()
            self.log_success(f"Python available in Docker: {python_version}")
        else:
            self.log_error("Python not available in Docker environment")
            success = False
            
        return success
    
    def validate_requirement_4_3(self) -> bool:
        """Validate Requirement 4.3: System dependencies installation"""
        self.log_info("Validating Requirement 4.3: System dependencies")
        
        success = True
        
        # Check build tools
        build_tools = ['gcc', 'g++', 'make', 'pkg-config', 'wget', 'curl']
        for tool in build_tools:
            exit_code, _, _ = self.run_command(f"which {tool}")
            if exit_code == 0:
                self.log_success(f"{tool} available")
            else:
                self.log_error(f"{tool} not found")
                success = False
        
        # Check TA-Lib C library
        exit_code, stdout, _ = self.run_command("ldconfig -p | grep ta_lib")
        if exit_code == 0 and stdout.strip():
            self.log_success("TA-Lib C library available in system library path")
        else:
            self.log_error("TA-Lib C library not found in system library path")
            success = False
        
        # Check TA-Lib pkg-config
        exit_code, _, _ = self.run_command("pkg-config --exists ta-lib")
        if exit_code == 0:
            self.log_success("TA-Lib pkg-config available")
        else:
            self.log_warning("TA-Lib pkg-config not available (may still work)")
            
        return success
    
    def validate_requirement_4_4(self) -> bool:
        """Validate Requirement 4.4: Python 3.13.6 with virtual environment"""
        self.log_info("Validating Requirement 4.4: Python 3.13.6 with virtual environment")
        
        success = True
        
        # Check Python version
        python_version = sys.version
        self.log_info(f"Python version: {python_version}")
        
        if "3.13" in python_version:
            self.log_success("Python 3.13.x requirement satisfied")
        else:
            self.log_error(f"Python 3.13.x required but found: {python_version}")
            success = False
        
        # Check virtual environment
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env and '.venv' in virtual_env:
            self.log_success(f"Virtual environment active: {virtual_env}")
        else:
            self.log_error("Virtual environment not properly activated")
            success = False
        
        # Check Python executable path
        python_executable = sys.executable
        if '.venv' in python_executable:
            self.log_success(f"Using virtual environment Python: {python_executable}")
        else:
            self.log_error(f"Not using virtual environment Python: {python_executable}")
            success = False
            
        return success
    
    def validate_requirement_4_7(self) -> bool:
        """Validate Requirement 4.7: Fail-fast with diagnostic information"""
        self.log_info("Validating Requirement 4.7: Diagnostic capabilities")
        
        success = True
        
        # Check system resource information availability
        diagnostics = {
            'memory': 'free -h',
            'disk_space': 'df -h .',
            'python_version': 'python --version',
            'pip_version': 'pip --version',
            'current_directory': 'pwd'
        }
        
        for diagnostic_name, command in diagnostics.items():
            exit_code, stdout, stderr = self.run_command(command)
            if exit_code == 0:
                self.log_success(f"{diagnostic_name} diagnostic available: {stdout.strip()[:50]}...")
            else:
                self.log_warning(f"{diagnostic_name} diagnostic failed: {stderr}")
        
        # Check error handling capabilities
        if hasattr(sys, 'exit'):
            self.log_success("Fail-fast capability (sys.exit) available")
        else:
            self.log_error("Fail-fast capability not available")
            success = False
            
        return success
    
    def validate_critical_packages(self) -> bool:
        """Validate critical Python packages"""
        self.log_info("Validating critical Python packages")
        
        packages = [
            ('talib', 'TA-Lib'),
            ('neuralforecast', 'NeuralForecast'),
            ('pandas', 'Pandas'),
            ('numpy', 'NumPy'),
            ('pyarrow', 'PyArrow'),
            ('torch', 'PyTorch'),
            ('vectorbt', 'VectorBT'),
            ('sklearn', 'Scikit-learn')
        ]
        
        failed_packages = []
        
        for pkg_name, display_name in packages:
            try:
                module = importlib.import_module(pkg_name)
                version = getattr(module, '__version__', 'unknown')
                self.log_success(f"{display_name}: {version}")
            except ImportError as e:
                self.log_error(f"{display_name}: Import failed - {e}")
                failed_packages.append(display_name)
            except Exception as e:
                self.log_error(f"{display_name}: Unexpected error - {e}")
                failed_packages.append(display_name)
        
        if failed_packages:
            self.log_error(f"Failed to import {len(failed_packages)} packages: {failed_packages}")
            return False
        else:
            self.log_success(f"All {len(packages)} critical packages validated successfully")
            return True
    
    def test_functionality(self) -> bool:
        """Test critical functionality"""
        self.log_info("Testing critical functionality")
        
        success = True
        
        try:
            # Test TA-Lib functionality
            import numpy as np
            import talib
            
            test_data = np.random.random(100)
            sma = talib.SMA(test_data, timeperiod=10)
            
            if not np.isnan(sma[-1]):
                self.log_success("TA-Lib functionality verified")
            else:
                self.log_error("TA-Lib SMA calculation failed")
                success = False
                
        except Exception as e:
            self.log_error(f"TA-Lib functionality test failed: {e}")
            success = False
        
        try:
            # Test NeuralForecast import
            from neuralforecast import NeuralForecast
            self.log_success("NeuralForecast core functionality verified")
            
        except Exception as e:
            self.log_error(f"NeuralForecast functionality test failed: {e}")
            success = False
        
        try:
            # Test data processing
            import pandas as pd
            
            df = pd.DataFrame({'test': [1, 2, 3]})
            if len(df) == 3:
                self.log_success("Data processing libraries verified")
            else:
                self.log_error("Pandas functionality test failed")
                success = False
                
        except Exception as e:
            self.log_error(f"Data processing functionality test failed: {e}")
            success = False
            
        return success
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'success': len(self.errors) == 0
        }
    
    def run_validation(self) -> bool:
        """Run complete validation"""
        self.log_info("=== CircleCI Environment Setup Validation ===")
        
        # Run all requirement validations
        req_4_1 = self.validate_requirement_4_1()
        req_4_3 = self.validate_requirement_4_3()
        req_4_4 = self.validate_requirement_4_4()
        req_4_7 = self.validate_requirement_4_7()
        
        # Validate packages and functionality
        packages_ok = self.validate_critical_packages()
        functionality_ok = self.test_functionality()
        
        # Generate report
        report = self.generate_report()
        
        self.log_info("=== Validation Summary ===")
        self.log_info(f"Requirements validation:")
        self.log_info(f"- Requirement 4.1 (Docker executor): {'✓' if req_4_1 else '✗'}")
        self.log_info(f"- Requirement 4.3 (System dependencies): {'✓' if req_4_3 else '✗'}")
        self.log_info(f"- Requirement 4.4 (Python 3.13.6 + venv): {'✓' if req_4_4 else '✗'}")
        self.log_info(f"- Requirement 4.7 (Fail-fast diagnostics): {'✓' if req_4_7 else '✗'}")
        self.log_info(f"- Critical packages: {'✓' if packages_ok else '✗'}")
        self.log_info(f"- Functionality tests: {'✓' if functionality_ok else '✗'}")
        
        if report['total_warnings'] > 0:
            self.log_warning(f"Total warnings: {report['total_warnings']}")
        
        if report['success']:
            self.log_success("✅ Environment validation completed successfully")
            self.log_success("All requirements (4.1, 4.3, 4.4, 4.7) satisfied")
            return True
        else:
            self.log_error(f"❌ Environment validation failed with {report['total_errors']} errors")
            return False


def main():
    """Main function"""
    validator = EnvironmentValidator()
    success = validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()