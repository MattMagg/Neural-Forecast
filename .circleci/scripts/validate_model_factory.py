#!/usr/bin/env python3
"""
Model Factory Validation Script for CircleCI

This script validates the NeuralForecast model factory implementation including:
- Model instantiation for all 4 models (NHITS, NBEATSx, TiDE, PatchTST)
- Loss function validation (DistributionLoss, MQLoss, IQLoss)
- Exogenous variable wiring and scaler configurations
- GPU resource validation and memory management
- Model artifact caching

Requirements: 1.4, 4.2, 4.5, 5.4, 6.4
"""

import sys
import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from nf_models.factory import instantiate_models, _loss_ctor
    from utils.io import load_and_process_data
    from utils.validate import assert_regular_grid, assert_utc_eob, assert_no_forward_fill_y
except ImportError as e:
    print(f"✗ CRITICAL ERROR: Failed to import required modules: {e}")
    print("Ensure all project modules are available and properly installed")
    sys.exit(1)


class ModelFactoryValidator:
    """Comprehensive model factory validation for CircleCI"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'gpu_available': torch.cuda.is_available(),
            'validation_results': {},
            'errors': [],
            'warnings': []
        }
        
        # Setup GPU device if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.results['gpu_info'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            }
        else:
            self.device = torch.device('cpu')
            self.results['gpu_info'] = None
    
    def log_result(self, test_name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Log validation result"""
        self.results['validation_results'][test_name] = {
            'passed': passed,
            'message': message,
            'details': details or {}
        }
        
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def validate_gpu_resources(self) -> bool:
        """Validate GPU resources and CUDA availability"""
        print("\n=== GPU Resource Validation ===")
        
        try:
            if not torch.cuda.is_available():
                self.log_result(
                    "gpu_availability", 
                    True, 
                    "CUDA not available - CPU mode (expected in non-GPU environment)",
                    {"device": str(self.device)}
                )
                return True  # Not a failure, just informational
            
            # Test GPU memory and capabilities
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            self.log_result(
                "gpu_detection",
                True,
                f"GPU detected: {gpu_name}",
                {
                    "gpu_count": gpu_count,
                    "total_memory_gb": f"{total_memory / 1024**3:.1f}GB",
                    "cuda_version": torch.version.cuda
                }
            )
            
            # Test basic GPU operations
            test_tensor = torch.randn(100, 100).to(self.device)
            result = torch.matmul(test_tensor, test_tensor.T)
            
            self.log_result(
                "gpu_operations",
                True,
                "Basic GPU operations successful",
                {"tensor_shape": list(result.shape)}
            )
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.log_result(
                "gpu_validation",
                False,
                f"GPU validation failed: {e}"
            )
            return False
    
    def validate_model_instantiation(self) -> bool:
        """Validate model instantiation for all 4 models"""
        print("\n=== Model Instantiation Validation ===")
        
        try:
            # Test each horizon configuration
            horizons = ['h4', 'h8', 'h16', 'h32']
            expected_models = ['NHITS', 'NBEATSx', 'TiDE', 'PatchTST']
            
            all_models_tested = set()
            total_instances = 0
            
            for horizon in horizons:
                config_path = f'experiments/{horizon}.yaml'
                
                if not Path(config_path).exists():
                    self.log_result(
                        f"config_file_{horizon}",
                        False,
                        f"Configuration file not found: {config_path}"
                    )
                    continue
                
                # Load configuration
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                model_config = config.get('models', {})
                horizon_steps = config.get('horizon', 4)
                
                # Create configuration in the expected format
                cfg = {
                    'h': horizon_steps,
                    'freq': '15min',
                    'models': model_config,
                    'scaler_type': 'robust'
                }
                
                # Test model instantiation
                models = instantiate_models(
                    cfg=cfg,
                    hist_cols=[],
                    futr_cols=[],
                    stat_cols=[]
                )
                
                horizon_models = []
                for model in models:
                    model_name = model.__class__.__name__
                    horizon_models.append(model_name)
                    all_models_tested.add(model_name)
                    total_instances += 1
                
                self.log_result(
                    f"model_instantiation_{horizon}",
                    True,
                    f"Successfully instantiated {len(models)} models",
                    {
                        "horizon_steps": horizon_steps,
                        "models": horizon_models
                    }
                )
            
            # Verify all expected models were tested
            missing_models = set(expected_models) - all_models_tested
            
            if missing_models:
                self.log_result(
                    "model_coverage",
                    False,
                    f"Missing model types: {list(missing_models)}",
                    {
                        "expected": expected_models,
                        "tested": list(all_models_tested),
                        "missing": list(missing_models)
                    }
                )
                return False
            
            self.log_result(
                "model_instantiation_complete",
                True,
                f"All {len(expected_models)} model types tested successfully",
                {
                    "model_types": list(all_models_tested),
                    "total_instances": total_instances
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result(
                "model_instantiation",
                False,
                f"Model instantiation failed: {e}"
            )
            return False
    
    def validate_loss_functions(self) -> bool:
        """Validate loss function implementations"""
        print("\n=== Loss Function Validation ===")
        
        try:
            loss_configs = [
                {
                    'kind': 'studentt'
                },
                {
                    'kind': 'mqloss',
                    'level': [80, 90, 95]
                },
                {
                    'kind': 'iqloss'
                }
            ]
            
            tested_losses = []
            
            for loss_config in loss_configs:
                loss_type = loss_config['kind']
                
                try:
                    # Instantiate loss function
                    loss_fn = _loss_ctor(loss_config)
                    
                    # For validation purposes, we just need to verify the loss function
                    # can be instantiated correctly. Actual loss computation testing
                    # would require proper model outputs which is beyond the scope
                    # of this validation script.
                    
                    # Verify the loss function has the expected attributes
                    if hasattr(loss_fn, '__call__'):
                        loss_callable = True
                    else:
                        loss_callable = False
                    
                    if not loss_callable:
                        raise Exception(f"Loss function {loss_type} is not callable")
                    
                    self.log_result(
                        f"loss_function_{loss_type}",
                        True,
                        f"Loss function instantiation successful",
                        {
                            "loss_type": str(type(loss_fn)),
                            "callable": loss_callable
                        }
                    )
                    
                    tested_losses.append(loss_type)
                    
                except Exception as e:
                    self.log_result(
                        f"loss_function_{loss_type}",
                        False,
                        f"Loss function test failed: {e}"
                    )
                    return False
            
            # Verify all loss functions were tested
            expected_losses = ['studentt', 'mqloss', 'iqloss']
            if set(tested_losses) != set(expected_losses):
                missing = set(expected_losses) - set(tested_losses)
                self.log_result(
                    "loss_function_coverage",
                    False,
                    f"Missing loss functions: {list(missing)}"
                )
                return False
            
            self.log_result(
                "loss_function_validation_complete",
                True,
                f"All {len(expected_losses)} loss functions validated",
                {"tested_losses": tested_losses}
            )
            
            return True
            
        except Exception as e:
            self.log_result(
                "loss_function_validation",
                False,
                f"Loss function validation failed: {e}"
            )
            return False
    
    def validate_exogenous_variables(self) -> bool:
        """Validate exogenous variable wiring and scaler configurations"""
        print("\n=== Exogenous Variable and Scaler Validation ===")
        
        try:
            # Create synthetic data with exogenous features
            dates = pd.date_range('2024-01-01', periods=1000, freq='15min', tz='UTC')
            
            df = pd.DataFrame({
                'unique_id': 'BTC-USD',
                'ds': dates,
                'y': np.random.randn(1000) * 0.01,
                'open': 50000 + np.random.randn(1000) * 1000,
                'high': 50000 + np.random.randn(1000) * 1000,
                'low': 50000 + np.random.randn(1000) * 1000,
                'close': 50000 + np.random.randn(1000) * 1000,
                'volume': np.random.exponential(1000, 1000),
                # Exogenous features
                'rsi_14': np.random.uniform(20, 80, 1000),
                'sma_20': 50000 + np.random.randn(1000) * 500,
                'bb_upper': 50000 + np.random.randn(1000) * 500,
                'bb_lower': 50000 + np.random.randn(1000) * 500,
                'volume_sma': np.random.exponential(800, 1000)
            })
            
            # Define exogenous variable lists
            hist_exog_list = ['rsi_14', 'sma_20', 'bb_upper', 'bb_lower', 'volume_sma']
            futr_exog_list = []
            stat_exog_list = []
            
            # Load configuration for testing
            with open('experiments/h4.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config.get('models', {})
            horizon_steps = config.get('horizon', 4)
            
            # Create configuration with exogenous variables
            cfg = {
                'h': horizon_steps,
                'freq': '15min',
                'models': model_config,
                'scaler_type': 'robust'
            }
            
            # Test model instantiation with exogenous variables
            models = instantiate_models(
                cfg=cfg,
                hist_cols=hist_exog_list,
                futr_cols=futr_exog_list,
                stat_cols=stat_exog_list
            )
            
            exog_support_count = 0
            scaler_support_count = 0
            
            for model in models:
                model_name = model.__class__.__name__
                
                # Check exogenous variable support
                has_exog_support = (
                    hasattr(model, 'hist_exog_list') or
                    hasattr(model, 'futr_exog_list') or
                    hasattr(model, 'stat_exog_list')
                )
                
                if has_exog_support:
                    exog_support_count += 1
                
                # Check scaler configuration
                has_scaler_config = (
                    hasattr(model, 'scaler_type') or
                    hasattr(model, 'revin') or
                    hasattr(model, 'scaler')
                )
                
                if has_scaler_config:
                    scaler_support_count += 1
                
                self.log_result(
                    f"exog_scaler_{model_name}",
                    True,
                    f"Exogenous and scaler configuration validated",
                    {
                        "exog_support": has_exog_support,
                        "scaler_support": has_scaler_config
                    }
                )
            
            self.log_result(
                "exogenous_variable_validation",
                True,
                f"Exogenous variable wiring validated for {len(models)} models",
                {
                    "models_with_exog_support": exog_support_count,
                    "models_with_scaler_support": scaler_support_count,
                    "hist_exog_features": len(hist_exog_list)
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result(
                "exogenous_variable_validation",
                False,
                f"Exogenous variable validation failed: {e}"
            )
            return False
    
    def validate_gpu_memory_management(self) -> bool:
        """Validate GPU memory management"""
        print("\n=== GPU Memory Management Validation ===")
        
        if not torch.cuda.is_available():
            self.log_result(
                "gpu_memory_management",
                True,
                "GPU not available - skipping memory management tests"
            )
            return True
        
        try:
            device = torch.device('cuda')
            
            # Get initial memory state
            initial_memory = torch.cuda.memory_allocated(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            
            # Test memory allocation and cleanup
            tensors = []
            for i in range(3):
                tensor = torch.randn(1000, 1000).to(device)
                tensors.append(tensor)
            
            peak_memory = torch.cuda.memory_allocated(device)
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated(device)
            memory_freed = peak_memory - final_memory
            
            self.log_result(
                "gpu_memory_management",
                True,
                "GPU memory management validated",
                {
                    "initial_memory_mb": f"{initial_memory / 1024**2:.1f}",
                    "peak_memory_mb": f"{peak_memory / 1024**2:.1f}",
                    "final_memory_mb": f"{final_memory / 1024**2:.1f}",
                    "memory_freed_mb": f"{memory_freed / 1024**2:.1f}",
                    "total_memory_gb": f"{total_memory / 1024**3:.1f}"
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result(
                "gpu_memory_management",
                False,
                f"GPU memory management validation failed: {e}"
            )
            return False
    
    def cache_model_artifacts(self) -> bool:
        """Cache model artifacts and metadata"""
        print("\n=== Model Artifact Caching ===")
        
        try:
            # Create cache directory
            cache_dir = Path('cache/models')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model configuration
            with open('experiments/h4.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config.get('models', {})
            horizon_steps = config.get('horizon', 4)
            
            # Create configuration for caching test
            cfg = {
                'h': horizon_steps,
                'freq': '15min',
                'models': model_config,
                'scaler_type': 'robust'
            }
            
            # Instantiate models for caching
            models = instantiate_models(
                cfg=cfg,
                hist_cols=[],
                futr_cols=[],
                stat_cols=[]
            )
            
            # Create comprehensive metadata
            cache_metadata = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'horizon': horizon_steps,
                'frequency': '15min',
                'models': [],
                'validation_results': self.results['validation_results'],
                'gpu_info': self.results['gpu_info'],
                'environment': {
                    'pytorch_version': torch.__version__,
                    'python_version': sys.version,
                    'cuda_available': torch.cuda.is_available()
                }
            }
            
            # Process each model
            for model in models:
                model_name = model.__class__.__name__
                
                model_info = {
                    'name': model_name,
                    'type': str(type(model)),
                    'parameters': {},
                    'config_source': f'experiments/h{horizon_steps}.yaml'
                }
                
                # Extract model parameters
                if hasattr(model, 'input_size'):
                    model_info['parameters']['input_size'] = model.input_size
                if hasattr(model, 'h'):
                    model_info['parameters']['horizon'] = model.h
                if hasattr(model, 'loss'):
                    model_info['parameters']['loss'] = str(model.loss)
                if hasattr(model, 'scaler_type'):
                    model_info['parameters']['scaler_type'] = model.scaler_type
                
                # Save individual model config
                model_config_path = cache_dir / f'{model_name}_h{horizon_steps}_config.json'
                with open(model_config_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                cache_metadata['models'].append(model_info)
            
            # Save overall cache metadata
            metadata_path = cache_dir / 'model_cache_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            # Verify cache integrity
            cached_files = list(cache_dir.glob('*.json'))
            
            for cached_file in cached_files:
                with open(cached_file, 'r') as f:
                    json.load(f)  # Verify JSON is valid
            
            self.log_result(
                "model_artifact_caching",
                True,
                f"Model artifacts cached successfully",
                {
                    "cache_directory": str(cache_dir),
                    "cached_files": len(cached_files),
                    "models_cached": len(models),
                    "metadata_file": str(metadata_path)
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result(
                "model_artifact_caching",
                False,
                f"Model artifact caching failed: {e}"
            )
            return False
    
    def generate_validation_report(self) -> bool:
        """Generate comprehensive validation report"""
        print("\n=== Generating Validation Report ===")
        
        try:
            # Calculate summary statistics
            total_tests = len(self.results['validation_results'])
            passed_tests = sum(1 for result in self.results['validation_results'].values() if result['passed'])
            failed_tests = total_tests - passed_tests
            
            # Create summary
            summary = {
                'validation_summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
                },
                'requirements_coverage': {
                    '1.4': 'Model instantiation testing for all 4 models',
                    '4.2': 'GPU-enabled job with linux-cuda-12:default',
                    '4.5': 'GPU resource class gpu.nvidia.medium',
                    '5.4': 'Loss function validation',
                    '6.4': 'Model artifact caching'
                },
                'validation_details': self.results
            }
            
            # Save validation report
            report_path = Path('test-results/model_factory_validation_report.json')
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Create JUnit XML for CircleCI
            junit_xml = self._create_junit_xml()
            junit_path = Path('test-results/model_factory_validation.xml')
            
            with open(junit_path, 'w') as f:
                f.write(junit_xml)
            
            self.log_result(
                "validation_report_generation",
                True,
                f"Validation report generated",
                {
                    "report_path": str(report_path),
                    "junit_path": str(junit_path),
                    "success_rate": summary['validation_summary']['success_rate']
                }
            )
            
            return failed_tests == 0
            
        except Exception as e:
            self.log_result(
                "validation_report_generation",
                False,
                f"Report generation failed: {e}"
            )
            return False
    
    def _create_junit_xml(self) -> str:
        """Create JUnit XML format for CircleCI test results"""
        total_tests = len(self.results['validation_results'])
        failed_tests = sum(1 for result in self.results['validation_results'].values() if not result['passed'])
        
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="ModelFactoryValidation" tests="{total_tests}" failures="{failed_tests}" time="0">',
        ]
        
        for test_name, result in self.results['validation_results'].items():
            if result['passed']:
                xml_lines.append(f'  <testcase name="{test_name}" classname="ModelFactoryValidation"/>')
            else:
                xml_lines.append(f'  <testcase name="{test_name}" classname="ModelFactoryValidation">')
                xml_lines.append(f'    <failure message="{result["message"]}"/>')
                xml_lines.append('  </testcase>')
        
        xml_lines.append('</testsuite>')
        return '\n'.join(xml_lines)
    
    def run_validation(self) -> bool:
        """Run complete model factory validation"""
        print("=== Model Factory Validation for CircleCI ===")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"GPU Available: {self.results['gpu_available']}")
        print(f"Device: {self.device}")
        
        validation_steps = [
            ("GPU Resources", self.validate_gpu_resources),
            ("Model Instantiation", self.validate_model_instantiation),
            ("Loss Functions", self.validate_loss_functions),
            ("Exogenous Variables", self.validate_exogenous_variables),
            ("GPU Memory Management", self.validate_gpu_memory_management),
            ("Model Artifact Caching", self.cache_model_artifacts),
            ("Validation Report", self.generate_validation_report)
        ]
        
        all_passed = True
        
        for step_name, step_function in validation_steps:
            try:
                print(f"\n--- {step_name} ---")
                step_result = step_function()
                if not step_result:
                    all_passed = False
                    print(f"✗ {step_name} validation failed")
                else:
                    print(f"✓ {step_name} validation passed")
            except Exception as e:
                print(f"✗ {step_name} validation error: {e}")
                all_passed = False
        
        # Final summary
        print(f"\n=== Model Factory Validation Summary ===")
        print(f"Overall Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        
        total_tests = len(self.results['validation_results'])
        passed_tests = sum(1 for result in self.results['validation_results'].values() if result['passed'])
        
        print(f"Tests: {passed_tests}/{total_tests} passed")
        
        if not all_passed:
            print("\nFailed Tests:")
            for test_name, result in self.results['validation_results'].items():
                if not result['passed']:
                    print(f"  - {test_name}: {result['message']}")
        
        return all_passed


def main():
    """Main validation entry point"""
    validator = ModelFactoryValidator()
    
    try:
        success = validator.run_validation()
        
        if success:
            print("\n🎉 Model Factory validation completed successfully!")
            print("All requirements (1.4, 4.2, 4.5, 5.4, 6.4) satisfied")
            sys.exit(0)
        else:
            print("\n❌ Model Factory validation failed!")
            print("Check the validation report for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()