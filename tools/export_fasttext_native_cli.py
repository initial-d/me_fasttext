"""Direct checkout script for the same exporter as meft-fasttext-export."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fasttext_native_bundle import main

if __name__=='__main__':main()
