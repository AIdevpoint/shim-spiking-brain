#!/usr/bin/env bash
# File interno, caricato dagli script Linux in un processo dedicato.
SHIM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly SHIM_ROOT
cd -- "$SHIM_ROOT"

# Le variabili valgono solo per lo script e i suoi figli. PATH non viene cambiato.
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$SHIM_ROOT/.cache/pycache"
export XDG_CACHE_HOME="$SHIM_ROOT/.cache"
export XDG_DATA_HOME="$SHIM_ROOT/.local/share"
export XDG_CONFIG_HOME="$SHIM_ROOT/.local/config"
export XDG_STATE_HOME="$SHIM_ROOT/.local/state"
export TMPDIR="$SHIM_ROOT/.tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export UV_PYTHON_INSTALL_DIR="$SHIM_ROOT/.python"
export UV_PYTHON_BIN_DIR="$SHIM_ROOT/.python/bin"
export UV_PYTHON_INSTALL_BIN=0
export UV_PYTHON_INSTALL_REGISTRY=0
export UV_CACHE_DIR="$SHIM_ROOT/.cache/uv"
export UV_NO_CONFIG=1
export UV_MANAGED_PYTHON=1
export UV_LINK_MODE=copy
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="$SHIM_ROOT/.cache/pip"
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCH_HOME="$SHIM_ROOT/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$SHIM_ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$SHIM_ROOT/.cache/triton"
export TRITON_DUMP_DIR="$SHIM_ROOT/.cache/triton-dump"
export TRITON_OVERRIDE_DIR="$SHIM_ROOT/.cache/triton-override"
export CUDA_CACHE_PATH="$SHIM_ROOT/.cache/cuda"
export HF_HOME="$SHIM_ROOT/.cache/huggingface"
export NUMBA_CACHE_DIR="$SHIM_ROOT/.cache/numba"
export MPLCONFIGDIR="$SHIM_ROOT/.cache/matplotlib"
export PYTHONIOENCODING=utf-8

mkdir -p -- "$TMPDIR" "$UV_CACHE_DIR" "$XDG_DATA_HOME" \
    "$XDG_CONFIG_HOME" "$XDG_STATE_HOME"
SHIM_PY="$SHIM_ROOT/.venv/bin/python"
readonly SHIM_PY

shim_carica_runtime_intel() {
    local runtime="$SHIM_ROOT/.local/intel-runtime"
    [[ -f "$runtime/manifest.json" ]] || return 0
    local dir
    for dir in "$runtime/usr/lib/x86_64-linux-gnu" "$runtime/usr/local/lib" "$runtime/usr/lib"; do
        if [[ -d "$dir" ]]; then
            export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export LIBRARY_PATH="$dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
        fi
    done
    export LEVEL_ZERO_V1_SDK_PATH="$runtime/usr"
}
shim_carica_runtime_intel

shim_verifica_ambiente() {
    "$SHIM_PY" - "$SHIM_ROOT" <<'PY'
from pathlib import Path
import site
import sys

root = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != root / ".venv":
    raise SystemExit("Errore: l'interprete non appartiene a .venv del repository.")
if not Path(sys.base_prefix).resolve().is_relative_to(root / ".python"):
    raise SystemExit("Errore: .venv deve usare il Python locale in .python.")
if sys.version_info[:2] != (3, 11) or site.ENABLE_USER_SITE:
    raise SystemExit("Errore: sono richiesti Python 3.11 e user site disabilitato.")
outside = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)]
if outside:
    raise SystemExit("Errore: percorsi Python esterni al repository: " + repr(outside))
PY
}
