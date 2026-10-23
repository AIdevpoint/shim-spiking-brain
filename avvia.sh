#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'Uso: ./avvia.sh [--porta NUMERO]\n'
    printf 'Server in primo piano; aprire http://127.0.0.1:8800/. Arresto: Ctrl+C.\n'
    exit 0
fi
porta=8800
if (( $# )); then
    if [[ $# != 2 || $1 != --porta || ! $2 =~ ^[0-9]{1,5}$ ]]; then
        printf 'Uso: ./avvia.sh [--porta NUMERO]\n' >&2
        exit 1
    fi
    porta=$((10#$2))
    if (( porta < 1 || porta > 65535 )); then
        printf 'Porta non valida.\n' >&2
        exit 1
    fi
fi

source "$(dirname -- "${BASH_SOURCE[0]}")/linux/ambiente.sh"
if [[ ! -x "$SHIM_PY" ]]; then
    printf 'Ambiente locale assente: eseguire prima ./installa.sh.\n' >&2
    exit 1
fi
shim_verifica_ambiente
"$SHIM_PY" "$SHIM_ROOT/linux/verifica_gpu.py"
printf '\nChat: http://127.0.0.1:%s/ — arresto: Ctrl+C\n' "$porta"
exec "$SHIM_PY" -u -m shim --porta "$porta"
