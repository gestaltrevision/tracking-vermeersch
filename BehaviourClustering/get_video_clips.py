import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", type=str,
                    help="File with config settings for testing")

if __name__ == "__main__":
    args = parser.parse_args()