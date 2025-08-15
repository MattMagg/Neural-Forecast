import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

print("Path to dataset files:", path)

# Copy to project directory
source_file = os.path.join(path, "btcusd_1-min_data.csv")
dest_file = os.path.join(os.path.dirname(__file__), "btcusd_1-min_data.csv")

print(f"Copying to: {dest_file}")
shutil.copy2(source_file, dest_file)
print("File copied successfully!")