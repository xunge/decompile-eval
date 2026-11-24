#!/usr/bin/env python3
"""Download evaluation-only subsets from official Hugging Face repositories.

No training data is downloaded. ExeBench is original v1.01 test_real, not
LLM4Decompile's historical 2,621-sample subset.
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


FILES = {
    "humaneval": ("LLM4Binary/decompile-eval", [
        "humaneval/data-00000-of-00001.arrow",
        "humaneval/dataset_info.json", "humaneval/state.json"]),
    "mbpp": ("LLM4Binary/decompile-eval", [
        "mbpp/data-00000-of-00001.arrow",
        "mbpp/dataset_info.json", "mbpp/state.json"]),
    "exebench": ("jordiae/exebench", ["test_real.tar.gz", "README.md"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="Download directory")
    parser.add_argument("--datasets", nargs="+", choices=list(FILES), default=list(FILES))
    args = parser.parse_args()
    root = Path(args.out).resolve()
    for name in args.datasets:
        repo, names = FILES[name]
        base = root / ("llm4binary-decompile-eval" if name != "exebench" else "jordiae-exebench-v1.01")
        print(f"\n=== {name.upper()} ({repo}) ===", flush=True)
        for filename in names:
            downloaded = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename,
                                         local_dir=str(base))
            print(f"[OK] {downloaded}", flush=True)
    print("\nDownloads complete. ExeBench test_real.tar.gz is archived raw data; it is not ready for direct RE evaluation.")


if __name__ == "__main__":
    main()
