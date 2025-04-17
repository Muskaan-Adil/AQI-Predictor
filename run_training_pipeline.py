import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
