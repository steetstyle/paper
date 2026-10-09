#!/usr/bin/env bash
# Create checker/.venv and install the pinned dependencies.
#
#   bash scripts/bootstrap.sh            # CUDA-enabled torch (default wheels, ~5 GB)
#   bash scripts/bootstrap.sh --cpu      # CPU-only torch (~1 GB, no GPU needed)
#   bash scripts/bootstrap.sh --dev      # also installs ruff/mypy/pytest
#   bash scripts/bootstrap.sh --no-deps  # only creates the venv
#
# Everything is installed from requirements.txt; the only choice the caller makes
# is which torch build to pull, because torch dominates the install size.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CPU=0
DEV=0
NO_DEPS=0

for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    --dev) DEV=1 ;;
    --no-deps) NO_DEPS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "bilinmeyen argüman: $arg" >&2; exit 2 ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null || PYTHON_BIN=python3

echo "==> venv: $VENV ($PYTHON_BIN)"
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip

if [ "$NO_DEPS" -eq 1 ]; then
  echo "==> bağımlılık kurulmadı (--no-deps)"
  exit 0
fi

REQ="$HERE/requirements.txt"
if [ "$CPU" -eq 1 ]; then
  echo "==> CPU-only torch"
  # requirements.txt lists plain torch==; override with the CPU wheel index.
  "$VENV/bin/python" -m pip install --quiet -r "$REQ" --extra-index-url https://download.pytorch.org/whl/cpu
else
  echo "==> varsayılan (CUDA) torch"
  "$VENV/bin/python" -m pip install --quiet -r "$REQ"
fi

if [ "$DEV" -eq 1 ]; then
  echo "==> geliştirme bağımlılıkları"
  "$VENV/bin/python" -m pip install --quiet -r "$HERE/requirements-dev.txt"
fi

"$VENV/bin/python" - <<'PY'
import importlib.metadata as md

def version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "yok"

print("==> kurulum tamam")
for name in ("torch", "transformers", "typer", "rich", "pydantic", "beautifulsoup4", "pypdf"):
    print(f"    {name:16s} {version(name)}")
PY

cat <<'EOF'

Doğrulama:
    .venv/bin/checker doctor
    .venv/bin/python -m pytest -q
EOF