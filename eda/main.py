import os
import pandas as pd
from pathlib import Path
from pandas_profiling import ProfileReport

INPUT_DATA = os.environ.get("INPUT_DATA")
OUTPUT_DATA = os.environ.get("OUTPUT_DATA")


def load_data():
    df = pd.read_csv(INPUT_DATA, header=0)
    return df


def main():
    df = load_data()
    profile = ProfileReport(df, title="Wine Quality", explorative=True)
    profile.to_file(Path(OUTPUT_DATA))


if __name__ == "__main__":
    main()
