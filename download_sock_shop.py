# ./download_sock_shop.py
"""Script for downloading the Sock-shop 2 dataset for use in experiments. 

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import requests
import zipfile
from pathlib import Path

def download_file(url: str, save_dir: str = "./datasets", filename: str = None, unzip: bool = True):
    """
    Download a file from a given URL into a specified directory, 
    unless it already exists. Optionally unzip and remove the archive.

    Parameters
    ----------
    url : str
        The file download URL.
    save_dir : str, optional
        Directory to save the file (default: "./datasets").
    filename : str, optional
        Name for the saved file (default: inferred from URL).
    unzip : bool, optional
        Whether to unzip the file if it's a .zip archive (default: True).
    """
    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Infer filename if not provided
    if filename is None:
        filename = url.split("?")[0].split("/")[-1]

    file_path = save_path / filename

    if file_path.exists() and not unzip:
        print(f"✅ File already exists at {file_path}. Skipping download.")
        return file_path

    if file_path.exists() and unzip and (file_path.suffix == ".zip"):
        print(f"✅ File already exists at {file_path}. Skipping download.")
    else:
        print(f"⬇️ Downloading {url} ...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()  # Raise error for bad status codes

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"✅ Downloaded to {file_path}")
        except requests.RequestException as e:
            print(f"❌ Error downloading file: {e}")
            return None

    # Handle unzipping if requested
    if unzip and file_path.suffix == ".zip":
        print(f"📦 Extracting {file_path} ...")
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(save_path)
            print(f"✅ Extracted to {save_path}")

            # Remove the zip file after extraction
            file_path.unlink()
            print(f"🗑️ Clean up")
        except zipfile.BadZipFile:
            print(f"❌ Error: {file_path} is not a valid zip archive.")
            return None

    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset if not already present.")
    parser.add_argument("--dir", type=str, default="./datasets", help="Directory to save the dataset.")
    parser.add_argument("--url", type=str, default="https://zenodo.org/records/13305663/files/sock-shop-2.zip?download=1", help="URL of the dataset.")
    parser.add_argument("--unzip", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True, help="Whether to unzip the file (default: True).")
    args = parser.parse_args()

    download_file(args.url, args.dir, unzip=args.unzip)