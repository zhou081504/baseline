"""Entry point for the requested comparison-table methods."""
import os
from pathlib import Path

os.environ.setdefault('TIKTOKEN_CACHE_DIR',str(Path(__file__).resolve().parent/'.cache'/'tiktoken'))
from table_baselines.cli import main

if __name__ == '__main__':
    main()
