#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
python3 -m pytest -q
curl --fail -s http://127.0.0.1:8010/api/health
printf '\n'
