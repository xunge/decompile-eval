#!/usr/bin/env bash
# No Python package required. Downloads ONLY benchmark evaluation files.
set -euo pipefail
ROOT="${1:-data/raw}"
BASE="${HF_ENDPOINT:-https://huggingface.co}"
BASE="${BASE%/}"
mkdir -p "$ROOT/llm4binary-decompile-eval/humaneval" "$ROOT/llm4binary-decompile-eval/mbpp" "$ROOT/jordiae-exebench-v1.01"
fetch() {
  local remote="$1" local_path="$2"
  mkdir -p "$(dirname "$local_path")"
  echo "Downloading $remote -> $local_path"
  curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 \
    --continue-at - --output "$local_path" "$remote"
}
for ds in humaneval mbpp; do
  for name in data-00000-of-00001.arrow dataset_info.json state.json; do
    fetch "$BASE/datasets/LLM4Binary/decompile-eval/resolve/main/$ds/$name" \
      "$ROOT/llm4binary-decompile-eval/$ds/$name"
  done
done
fetch "$BASE/datasets/jordiae/exebench/resolve/main/test_real.tar.gz" \
  "$ROOT/jordiae-exebench-v1.01/test_real.tar.gz"
echo "Done. Run Python converter after installing datasets and pyarrow."
