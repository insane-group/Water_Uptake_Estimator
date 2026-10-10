"""Repository-local reference and output paths for reproduction scripts."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'reference'


def output_directory(path=None):
    out = (Path(path) if path is not None else ROOT / 'results').resolve()
    if out == REFERENCE or REFERENCE in out.parents:
        raise ValueError('Choose an output directory outside the archived reference directory')
    out.mkdir(parents=True, exist_ok=True)
    return out


def analysis_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output-dir', type=Path, help='Generated files (default: repository/results)')
    return parser
