"""
Automated downloader for MovieLens dataset.
Downloads and extracts the dataset into data/raw directory.
"""

import os
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = os.path.join(RAW_DIR, "ml-latest-small.zip")


def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset from {DATASET_URL} ...")
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(DATASET_URL, context=ctx) as response, open(ZIP_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    else:
        print("Zip file already exists.")

    print("Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print(f"Dataset ready in: {os.path.join(RAW_DIR, 'ml-latest-small')}")


if __name__ == "__main__":
    download_and_extract()
