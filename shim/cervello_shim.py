from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parents[1]
MAPPA = RADICE / "shim_mappa.json"
RISULTATI = RADICE / "results"
FRAZIONE_E, DOMINANZA_I = 0.8, 1.25
W_BASE, K_RIF = 87.8, 1000.0
RIT_E_MS, RIT_I_MS, RIT_LUNGO_MS = 1.5, 0.75, 5.0
FREQ_E, FREQ_I = 3.0, 6.0
GUADAGNO_TARA, PASSO_MAX_PA, GIRI_TARA = 2.0, 30.0, 25


class CervelloUmano:

    def __init__(self, N=None, seme=0, verboso=True):
        from shim.neuroni import DT, FREQ_FONDO, _csr_per_mittente, TAU_S
        t0 = time.time()
        rng = np.random.default_rng(seme)
        M = json.loads(MAPPA.read_text(encoding="utf-8"))
        self.aree = M["aree"]
        bl = M["blocchi"]
        self.nomi = [b["nome"] for b in bl]
        self.na = na = len(bl)
        self.area_blocco = np.array([self.aree.index(b["area"]) for b in bl], np.int64)
        self.grandezze = np.array([b["neuroni"] for b in bl], np.int64)
        self.inizio = np.concatenate([[0], np.cumsum(self.grandezze)[:-1]]).astype(np.int64)
        self.n = n = int(self.grandezze.sum())
        area = np.repeat(np.arange(na), self.grandezze)
        n_e_zona = np.round(M.get("frazione_e", FRAZIONE_E) * self.grandezze).astype(np.int64)
        e_mask = (np.arange(n) - self.inizio[area]) < n_e_zona[area]
        self.area, self.eccitatorio, self.n_e_zona = area, e_mask, n_e_zona
        self.pop = (2 * area + (~e_mask)).astype(np.int64)
        k_loc = float(M["k_locale"])
        self.K = 2 * k_loc
        pre, post = [], []
        m = rng.poisson(k_loc, n)
        s = np.repeat(np.arange(n), m)
        post.append(self.inizio[area[s]] + (rng.random(len(s)) * self.grandezze[area[s]]).astype(np.int64))
        pre.append(s)
        P = np.asarray(M["lunghe"], np.float64)
        cum = np.cumsum(P / P.sum(1, keepdims=True), 1)
        nA = len(self.aree)
        macro = self.area_blocco[area]
        tot_area = np.bincount(self.area_blocco, weights=self.grandezze, minlength=nA).astype(np.int64)
        e_idx = np.flatnonzero(e_mask)
        m = rng.poisson(k_loc / FRAZIONE_E, len(e_idx))
        s = np.repeat(e_idx, m)
        b = np.empty(len(s), np.int64)
        za = macro[s]
        for a_ in range(nA):
            sel = np.flatnonzero(za == a_)
            b[sel] = np.minimum(np.searchsorted(cum[a_], rng.random(len(sel))), nA - 1)
        ordine_area = np.argsort(macro, kind="stable")
        inizio_area = np.concatenate([[0], np.cumsum(tot_area)[:-1]])
        post.append(ordine_area[inizio_area[b] + (rng.random(len(s)) * tot_area[b]).astype(np.int64)])
        pre.append(s)
        n_loc = len(pre[0])
        pre, post = np.concatenate(pre).astype(np.int64), np.concatenate(post).astype(np.int64)
        J = W_BASE * math.sqrt(K_RIF / max(self.K, 1.0))
        n_e_sin = int(e_mask[pre].sum())
        self.g_inib = DOMINANZA_I * n_e_sin / max(len(pre) - n_e_sin, 1)
        peso = np.where(e_mask[pre], J, -self.g_inib * J).astype(np.float32)
        peso *= (1.0 + 0.1 * rng.standard_normal(len(peso))).astype(np.float32)
        rit_ms = np.where(e_mask[pre], RIT_E_MS, RIT_I_MS)
        rit_ms[n_loc:] = RIT_LUNGO_MS
        rit = np.clip(np.round(rng.normal(rit_ms, rit_ms * 0.5) / DT), 1, 250).astype(np.uint8)
        self.n_sinapsi = len(pre)
        self.indptr, self.indices, self.pesi, self.ritardi = _csr_per_mittente(pre, post, peso, rit, n)
        del pre, post, peso, rit
        self.max_ritardo = int(self.ritardi.max()) + 1
        self.lam_est = np.full(n, 2000 * FREQ_FONDO * DT * 1e-3, np.float32)
        self.w_est = 87.8
        fondo = float(self.lam_est[0]) / (DT * 1e-3) * self.w_est * TAU_S * 1e-3
        self.dc = np.full(n, 375.0 - fondo - 30.0, np.float32)
        self.V0 = rng.normal(-58.0, 5.0, n).astype(np.float32)
        if verboso:
            print("[SHIM] %d aree, %d gruppi = %d neuroni, %d sinapsi (%.0f per neurone), in %.1f s" % (
                nA, na, n, self.n_sinapsi, self.n_sinapsi / n, time.time() - t0), flush=True)

    def gruppo(self, *nomi, eccitatori=True):
        ix = {n: i for i, n in enumerate(self.nomi)}
        parti = [np.flatnonzero((self.area == ix[n]) & (self.eccitatorio if eccitatori else True)) for n in nomi if n in ix]
        return np.concatenate(parti) if parti else np.zeros(0, np.int64)


def frequenze_pop(motore, c, ms):
    import torch
    from shim.neuroni import DT
    motore.conta.zero_()
    motore.passi(int(ms / DT))
    motore.sincro()
    conta = motore.conta.float()
    pop = torch.as_tensor(c.pop, device=conta.device)
    somma = torch.zeros(2 * c.na, device=conta.device).index_add_(0, pop, conta)
    quanti = torch.bincount(pop, minlength=2 * c.na).float()
    return (somma / quanti / (ms / 1000.0)).cpu().numpy()


def tara(motore, c, giri=GIRI_TARA, ms=500.0, verboso=True):
    import torch
    bersaglio = np.tile([FREQ_E, FREQ_I], c.na)
    pop = torch.as_tensor(c.pop, device=motore.dc.device)
    for g in range(giri):
        f = frequenze_pop(motore, c, ms)
        err = bersaglio - f
        passo = torch.as_tensor(np.clip(GUADAGNO_TARA * err, -PASSO_MAX_PA, PASSO_MAX_PA), dtype=torch.float32, device=motore.dc.device)
        motore.dc += passo[pop]
        if verboso:
            fe = f[0::2]
            print("  taratura %d: E media %.2f Hz (%.2f-%.2f), I media %.2f Hz" % (g, fe.mean(), fe.min(), fe.max(), f[1::2].mean()), flush=True)
    return f
