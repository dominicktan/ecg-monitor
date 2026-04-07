"""Download MIT-BIH Arrhythmia Database from PhysioNet."""

import os
import wfdb

DATA_DIR = "mit-bih-arrhythmia-database-1.0.0"


def main():
    if os.path.exists(DATA_DIR) and any(
        f.endswith(".dat") for f in os.listdir(DATA_DIR)
    ):
        print(f"Data already exists in {DATA_DIR}/")
        return

    print(f"Downloading MIT-BIH Arrhythmia Database to {DATA_DIR}/...")
    os.makedirs(DATA_DIR, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=DATA_DIR)
    print("Download complete.")


if __name__ == "__main__":
    main()
