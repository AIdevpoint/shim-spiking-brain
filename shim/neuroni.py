"""I neuroni di SHIM: integra-e-spara con corrente sinaptica esponenziale, e le sinapsi ordinate per mittente."""
from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]

DT = float(os.environ.get("SHIM_DT", "0.1"))
TAU_M, C_M, V_SOGLIA, V_RESET, E_L, T_REF, TAU_S = 10.0, 250.0, -50.0, -65.0, -65.0, 2.0, 0.5
FREQ_FONDO = 8.0


@njit(cache=True)
def _conta(pre, n):
    c = np.zeros(n + 1, np.int64)
    for p in pre:
        c[p + 1] += 1
    for i in range(n):
        c[i + 1] += c[i]
    return c


@njit(cache=True)
def _riempi(pre, post, peso, rit, indptr, indices, pesi, ritardi):
    pos = indptr[:-1].copy()
    for s in range(pre.shape[0]):
        p = pre[s]
        q = pos[p]
        indices[q] = post[s]
        pesi[q] = peso[s]
        ritardi[q] = rit[s]
        pos[p] = q + 1


def _csr_per_mittente(pre, post, peso, rit, n):
    indptr = _conta(pre, n)
    indices = np.empty_like(post)
    pesi = np.empty_like(peso)
    ritardi = np.empty_like(rit)
    _riempi(pre, post, peso, rit, indptr, indices, pesi, ritardi)
    return indptr, indices, pesi, ritardi
