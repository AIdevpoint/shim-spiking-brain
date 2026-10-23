#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    --solo-base) backend=base ;;
    --xpu) backend=xpu ;;
    --cuda) backend=cuda ;;
    --runtime-intel) backend=runtime-intel ;;
    -h|--help)
        cat <<'HELP'
Uso: ./installa.sh [--solo-base | --xpu | --cuda | --runtime-intel]
  --solo-base  Python 3.11 locale, NumPy e SciPy; simulazione non avviabile.
  --xpu        Aggiunge PyTorch e Triton per GPU Intel da wheel ufficiali.
  --cuda       Aggiunge PyTorch e Triton per GPU NVIDIA (CUDA 12.4).
  --runtime-intel  Estrae Level Zero e il runtime Intel nel repo (Linux x86_64).
Senza opzioni installa la base e aggiunge CUDA solo se nvidia-smi funziona.
Per Intel occorre scegliere --xpu dopo aver verificato i prerequisiti WSL/Linux.
Richiede uv gia disponibile. Tutti i download e le installazioni restano nel repo.
HELP
        exit 0 ;;
    "") backend=auto ;;
    *) printf 'Opzione sconosciuta: %s\n' "$1" >&2; exit 1 ;;
esac
if (( $# > 1 )); then
    printf 'Specificare una sola opzione.\n' >&2
    exit 1
fi

source "$(dirname -- "${BASH_SOURCE[0]}")/linux/ambiente.sh"
if [[ $(uname -s) != Linux ]]; then
    printf 'Questo script richiede Linux o WSL2.\n' >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    printf 'uv non trovato. Non viene installato automaticamente.\n' >&2
    exit 1
fi

uv --no-config python install 3.11 --no-bin
if [[ ! -e "$SHIM_ROOT/.venv" && ! -L "$SHIM_ROOT/.venv" ]]; then
    uv --no-config venv --no-project --managed-python --python 3.11 "$SHIM_ROOT/.venv"
fi
shim_verifica_ambiente
if [[ $backend == runtime-intel ]]; then
    "$SHIM_PY" "$SHIM_ROOT/linux/installa_runtime_intel.py"
    shim_carica_runtime_intel
    "$SHIM_PY" "$SHIM_ROOT/linux/verifica_gpu.py" xpu
    exit 0
fi
uv --no-config pip install --python "$SHIM_PY" --no-build \
    --index-url https://pypi.org/simple -r "$SHIM_ROOT/requirements.txt"

if [[ $backend == auto ]]; then
    backend=base
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        backend=cuda
    elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]] && /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1; then
        backend=cuda
    fi
fi
case "$backend" in
    xpu)
        # La wheel XPU include la versione compatibile di Triton per Intel.
        uv --no-config pip install --python "$SHIM_PY" --no-build \
            --index-url https://download.pytorch.org/whl/xpu torch
        ;;
    cuda)
        # Stesso ramo CUDA dell'installer Windows; niente triton-windows su Linux.
        uv --no-config pip install --python "$SHIM_PY" --no-build \
            --index-url https://download.pytorch.org/whl/cu124 torch
        ;;
esac
uv --no-config pip check --python "$SHIM_PY"
mkdir -p -- "$SHIM_ROOT/.local"
uv --no-config pip freeze --python "$SHIM_PY" > "$SHIM_ROOT/.local/pacchetti-installati.txt"
"$SHIM_PY" -c 'import sys, numpy, scipy; print("Python:", sys.version.split()[0]); print("Interprete:", sys.executable); print("NumPy:", numpy.__version__, "SciPy:", scipy.__version__)'
if [[ $backend == base ]]; then
    printf '\nBase installata nel repository. PyTorch/Triton GPU non installati da questo comando.\n'
    printf 'Per Intel: leggere LINUX.md e poi usare ./installa.sh --xpu.\n'
    printf 'La simulazione richiede una GPU utilizzabile; non ha un backend CPU.\n'
else
    "$SHIM_PY" "$SHIM_ROOT/linux/verifica_gpu.py" "$backend"
    printf '\nInstallazione completata. Avvio: ./avvia.sh\n'
fi
