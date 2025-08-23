#!/usr/bin/env python3
"""
Download and verify Bitcoin historical data from Kaggle.

This script downloads the BTC-USD 1-minute OHLCV data and performs
integrity checks to ensure the data is complete and valid.
"""

import kagglehub
import shutil
import os
import pandas as pd

def verify_btc_data(file_path):
    """Verify BTC data file integrity and basic statistics."""
    # Expected characteristics
    EXPECTED_MIN_ROWS = 7_000_000  # ~7M 1-minute bars
    EXPECTED_MIN_SIZE_MB = 400     # ~400-500MB file
    EXPECTED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb < EXPECTED_MIN_SIZE_MB:
        raise ValueError(f"File too small: {file_size_mb:.1f}MB, expected >= {EXPECTED_MIN_SIZE_MB}MB")
    
    # Load sample to verify structure
    df = pd.read_csv(file_path, nrows=100000)
    
    # Verify columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Count total rows
    print("Counting total rows...")
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
    if total_rows < EXPECTED_MIN_ROWS:
        raise ValueError(f"Too few rows: {total_rows:,}, expected >= {EXPECTED_MIN_ROWS:,}")
    
    # Verify date range
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    
    print(f"\n✅ Data verification passed:")
    print(f"   File size: {file_size_mb:.1f}MB")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Date range (sample): {date_range}")
    print(f"   Columns: {list(df.columns)}")
    
    return True

def main():
    """Main download and verification workflow."""
    # Download latest version
    print("Downloading Bitcoin historical data from Kaggle...")
    path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
    
    print("Path to dataset files:", path)
    
    # Copy to project directory
    source_file = os.path.join(path, "btcusd_1-min_data.csv")
    dest_file = os.path.join(os.path.dirname(__file__), "btcusd_1-min_data.csv")
    
    print(f"Copying to: {dest_file}")
    shutil.copy2(source_file, dest_file)
    print("File copied successfully!")
    
    # Verify the downloaded data
    print("\nVerifying data integrity...")
    try:
        verify_btc_data(dest_file)
    except Exception as e:
        print(f"\n❌ Data verification failed: {e}")
        print("The downloaded file may be corrupted or incomplete.")
        raise
    
    print("\n🎉 Download and verification complete!")
    print(f"Data ready at: {dest_file}")

if __name__ == "__main__":
    main()