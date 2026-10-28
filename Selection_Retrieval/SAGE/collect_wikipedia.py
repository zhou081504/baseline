"""SAGE (adapted): collect-wikipedia. See this folder's README.md."""
from pathlib import Path
import sys

MODEL_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(MODEL_DIR.parents[1]))
from table_baselines.entry import run

if __name__=='__main__':
    run('sage','collect-wikipedia',MODEL_DIR)
