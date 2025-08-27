#!/usr/bin/env python3
"""
CircleCI Data Processing Validation Script
Validates data processing pipeline and quality gates (Requirements 1.2, 5.2, 6.3)
"""

import sys
import os
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from utils.io import (
        load_raw_1min_data, 
        aggregate_1min_to_15min, 
        make_nf_canonical,
        load_and_process_data,
        save_parquet,
        timestamped_path
    )
    from utils.validate import (
        assert_regular_grid,
        assert_utc_eob, 
        assert_shifted,
        assert_no_forward_fill_y
    )
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Ensure you're running from project root with proper environment")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class DataProcessingValidator:
    """Validates data processing pipeline and quality gates"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.processed_data: Dict[str, Any] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def log_info(self, message: str) -> None:
        """Log info message"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
        self.logger.info(message)
        
    def log_success(self, message: str) -> None:
        """Log success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
        self.logger.info(f"SUCCESS: {message}")
        
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        self.warnings.append(message)
        self.logger.warning(message)
        
    def log_error(self, message: str) -> None:
        """Log error message"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        self.errors.append(message)
        self.logger.error(message)
        
    def validate_requirement_1_2(self) -> bool:
        """
        Validate Requirement 1.2: Foundation CI/CD Pipeline Support
        - Data processing pipeline validation
        - All 4 quality gates validation
        """
        self.log_info("Validating Requirement 1.2: Foundation CI/CD Pipeline Support")
        
        success = True
        
        # Check if raw data exists
        raw_data_path = "data/raw/btcusd_1-min_data.csv"
        if not Path(raw_data_path).exists():
            self.log_error(f"Raw data file not found: {raw_data_path}")
            return False
        
        try:
            # Test complete data processing pipeline
            self.log_info("Testing complete data processing pipeline...")
            
            # Load and process data using main entry point
            df_canonical = load_and_process_data(raw_data_path)
            
            self.log_success(f"Data processing completed: {len(df_canonical)} rows processed")
            self.processed_data['canonical_frame'] = df_canonical
            
            # Validate data shape and basic properties
            if len(df_canonical) == 0:
                self.log_error("Processed data is empty")
                success = False
            else:
                self.log_success(f"Processed {len(df_canonical)} canonical bars")
                
            # Check required columns
            required_cols = ['unique_id', 'ds', 'y', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df_canonical.columns]
            if missing_cols:
                self.log_error(f"Missing required NF canonical columns: {missing_cols}")
                success = False
            else:
                self.log_success("All required NF canonical columns present")
                
        except Exception as e:
            self.log_error(f"Data processing pipeline failed: {e}")
            self.log_error(f"Traceback: {traceback.format_exc()}")
            success = False
            
        return success
        
    def validate_requirement_5_2(self) -> bool:
        """
        Validate Requirement 5.2: Integration with existing workflow
        - Use existing utils/io.py functions
        - NeuralForecast canonical schema compliance
        """
        self.log_info("Validating Requirement 5.2: Integration with existing workflow")
        
        success = True
        
        if 'canonical_frame' not in self.processed_data:
            self.log_error("No processed data available for schema validation")
            return False
            
        df = self.processed_data['canonical_frame']
        
        try:
            # Test NeuralForecast schema compliance
            self.log_info("Testing NeuralForecast canonical schema compliance...")
            
            # Check unique_id column
            if 'unique_id' in df.columns:
                unique_ids = df['unique_id'].unique()
                if len(unique_ids) == 1 and unique_ids[0] == "BTC-USD":
                    self.log_success("unique_id column correctly set to 'BTC-USD'")
                else:
                    self.log_error(f"Invalid unique_id values: {unique_ids}")
                    success = False
            else:
                self.log_error("Missing unique_id column")
                success = False
                
            # Check ds column (datetime)
            if 'ds' in df.columns:
                if df['ds'].dtype.name.startswith('datetime64'):
                    self.log_success("ds column has correct datetime64 dtype")
                    
                    # Check timezone
                    if hasattr(df['ds'].dtype, 'tz') and str(df['ds'].dtype.tz) == 'UTC':
                        self.log_success("ds column has correct UTC timezone")
                    else:
                        self.log_error("ds column missing UTC timezone")
                        success = False
                else:
                    self.log_error(f"ds column has incorrect dtype: {df['ds'].dtype}")
                    success = False
            else:
                self.log_error("Missing ds column")
                success = False
                
            # Check y column (log returns)
            if 'y' in df.columns:
                if df['y'].dtype.name in ['float64', 'float32']:
                    self.log_success("y column has correct numeric dtype")
                    
                    # Check for reasonable log return values
                    y_valid = df['y'].dropna()
                    if len(y_valid) > 0:
                        y_range = (y_valid.min(), y_valid.max())
                        if -1.0 < y_range[0] and y_range[1] < 1.0:
                            self.log_success(f"y values in reasonable range: {y_range}")
                        else:
                            self.log_warning(f"y values may be extreme: {y_range}")
                    else:
                        self.log_error("All y values are NaN")
                        success = False
                else:
                    self.log_error(f"y column has incorrect dtype: {df['y'].dtype}")
                    success = False
            else:
                self.log_error("Missing y column (target)")
                success = False
                
            # Check OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in df.columns:
                    if df[col].dtype.name in ['float64', 'float32', 'int64', 'int32']:
                        self.log_success(f"{col} column has correct numeric dtype")
                    else:
                        self.log_error(f"{col} column has incorrect dtype: {df[col].dtype}")
                        success = False
                else:
                    self.log_error(f"Missing {col} column")
                    success = False
                    
        except Exception as e:
            self.log_error(f"Schema validation failed: {e}")
            self.log_error(f"Traceback: {traceback.format_exc()}")
            success = False
            
        return success
        
    def validate_requirement_6_3(self) -> bool:
        """
        Validate Requirement 6.3: Artifact management and reporting
        - Processed data caching and artifact storage
        - Comprehensive error reporting
        """
        self.log_info("Validating Requirement 6.3: Artifact management and reporting")
        
        success = True
        
        if 'canonical_frame' not in self.processed_data:
            self.log_error("No processed data available for artifact testing")
            return False
            
        df = self.processed_data['canonical_frame']
        
        try:
            # Test processed data caching
            self.log_info("Testing processed data caching...")
            
            # Create processed data directory
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamped filename
            cache_path = timestamped_path("data/processed", "btc_canonical", "parquet")
            
            # Save processed data
            save_parquet(df, cache_path)
            
            if Path(cache_path).exists():
                file_size = Path(cache_path).stat().st_size
                self.log_success(f"Processed data cached successfully: {cache_path} ({file_size} bytes)")
                self.processed_data['cache_path'] = cache_path
            else:
                self.log_error(f"Failed to cache processed data: {cache_path}")
                success = False
                
            # Test artifact metadata
            artifact_info = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'rows': len(df),
                'columns': list(df.columns),
                'file_path': cache_path,
                'file_size_bytes': Path(cache_path).stat().st_size if Path(cache_path).exists() else 0
            }
            
            self.processed_data['artifact_info'] = artifact_info
            self.log_success(f"Artifact metadata generated: {len(artifact_info)} fields")
            
        except Exception as e:
            self.log_error(f"Artifact management failed: {e}")
            self.log_error(f"Traceback: {traceback.format_exc()}")
            success = False
            
        return success
        
    def validate_quality_gates(self) -> bool:
        """
        Validate all 4 quality gate validations
        - assert_regular_grid
        - assert_utc_eob  
        - assert_shifted (if exogenous features present)
        - assert_no_forward_fill_y
        """
        self.log_info("Validating all 4 quality gate validations")
        
        if 'canonical_frame' not in self.processed_data:
            self.log_error("No processed data available for quality gate validation")
            return False
            
        df = self.processed_data['canonical_frame']
        success = True
        
        # Quality Gate 1: assert_regular_grid
        try:
            self.log_info("Testing Quality Gate 1: assert_regular_grid")
            assert_regular_grid(df, "15min")
            self.log_success("✓ Quality Gate 1 (assert_regular_grid) passed")
        except AssertionError as e:
            self.log_error(f"✗ Quality Gate 1 (assert_regular_grid) failed: {e}")
            success = False
        except Exception as e:
            self.log_error(f"✗ Quality Gate 1 (assert_regular_grid) error: {e}")
            success = False
            
        # Quality Gate 2: assert_utc_eob
        try:
            self.log_info("Testing Quality Gate 2: assert_utc_eob")
            assert_utc_eob(df, "15min")
            self.log_success("✓ Quality Gate 2 (assert_utc_eob) passed")
        except AssertionError as e:
            self.log_error(f"✗ Quality Gate 2 (assert_utc_eob) failed: {e}")
            success = False
        except Exception as e:
            self.log_error(f"✗ Quality Gate 2 (assert_utc_eob) error: {e}")
            success = False
            
        # Quality Gate 3: assert_shifted (only if historical exogenous features present)
        try:
            self.log_info("Testing Quality Gate 3: assert_shifted")
            
            # Find potential historical columns (exclude standard NF columns)
            standard_cols = {'unique_id', 'ds', 'y', 'open', 'high', 'low', 'close', 'volume'}
            hist_cols = [col for col in df.columns if col not in standard_cols]
            
            if hist_cols:
                self.log_info(f"Found {len(hist_cols)} potential historical columns: {hist_cols}")
                assert_shifted(df, hist_cols)
                self.log_success("✓ Quality Gate 3 (assert_shifted) passed")
            else:
                self.log_info("No historical exogenous features found - skipping assert_shifted")
                self.log_success("✓ Quality Gate 3 (assert_shifted) skipped (no exogenous features)")
                
        except AssertionError as e:
            self.log_error(f"✗ Quality Gate 3 (assert_shifted) failed: {e}")
            success = False
        except Exception as e:
            self.log_error(f"✗ Quality Gate 3 (assert_shifted) error: {e}")
            success = False
            
        # Quality Gate 4: assert_no_forward_fill_y
        try:
            self.log_info("Testing Quality Gate 4: assert_no_forward_fill_y")
            assert_no_forward_fill_y(df)
            self.log_success("✓ Quality Gate 4 (assert_no_forward_fill_y) passed")
        except AssertionError as e:
            self.log_error(f"✗ Quality Gate 4 (assert_no_forward_fill_y) failed: {e}")
            success = False
        except Exception as e:
            self.log_error(f"✗ Quality Gate 4 (assert_no_forward_fill_y) error: {e}")
            success = False
            
        return success
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'validation_summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'success': len(self.errors) == 0
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'processed_data_info': self.processed_data.get('artifact_info', {}),
            'requirements_status': {
                'requirement_1_2': 'canonical_frame' in self.processed_data,
                'requirement_5_2': len(self.errors) == 0,
                'requirement_6_3': 'cache_path' in self.processed_data
            }
        }
        
        return report
        
    def run_validation(self) -> bool:
        """Run complete data processing validation"""
        self.log_info("=== CircleCI Data Processing Validation ===")
        
        # Run requirement validations
        req_1_2 = self.validate_requirement_1_2()
        req_5_2 = self.validate_requirement_5_2() if req_1_2 else False
        req_6_3 = self.validate_requirement_6_3() if req_1_2 else False
        
        # Run quality gate validations
        quality_gates = self.validate_quality_gates() if req_1_2 else False
        
        # Generate report
        report = self.generate_validation_report()
        
        # Print summary
        self.log_info("=== Data Processing Validation Summary ===")
        self.log_info(f"Requirements validation:")
        self.log_info(f"- Requirement 1.2 (Foundation pipeline): {'✓' if req_1_2 else '✗'}")
        self.log_info(f"- Requirement 5.2 (Existing workflow): {'✓' if req_5_2 else '✗'}")
        self.log_info(f"- Requirement 6.3 (Artifact management): {'✓' if req_6_3 else '✗'}")
        self.log_info(f"- Quality Gates (all 4): {'✓' if quality_gates else '✗'}")
        
        if report['validation_summary']['total_warnings'] > 0:
            self.log_warning(f"Total warnings: {report['validation_summary']['total_warnings']}")
            
        # Print processed data statistics
        if 'canonical_frame' in self.processed_data:
            df = self.processed_data['canonical_frame']
            self.log_info(f"Processed data statistics:")
            self.log_info(f"- Rows: {len(df)}")
            self.log_info(f"- Columns: {len(df.columns)}")
            self.log_info(f"- Date range: {df['ds'].min()} to {df['ds'].max()}")
            self.log_info(f"- Non-null y values: {df['y'].notna().sum()}")
            
        if report['validation_summary']['success']:
            self.log_success("✅ Data processing validation completed successfully")
            self.log_success("All requirements (1.2, 5.2, 6.3) and quality gates satisfied")
            return True
        else:
            self.log_error(f"❌ Data processing validation failed with {report['validation_summary']['total_errors']} errors")
            return False


def main():
    """Main function"""
    # Change to project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    validator = DataProcessingValidator()
    success = validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()