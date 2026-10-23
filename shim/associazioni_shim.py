from __future__ import annotations

from pathlib import Path

import numpy as np

FINESTRA = 3
INDIETRO = 0.5
DIM = 16
MISCELA = 0.3
TRACCIA = 0.7


class Associazioni:
    def __init__(self, parole, file=None):
        self.parole = list(parole)
        self.K = len(self.parole)
        self.idx = {w: i for i, w in enumerate(self.parole)}
        self.N = np.zeros((self.K, self.K), np.float32)
        self.nk = np.zeros(self.K, np.float32)
        self.file = file
        self._ppmi = None
        self._vet = None
        if file is not None and Path(file).exists():
            d = np.load(file)
            if d["N"].shape == self.N.shape:
                self.N, self.nk = d["N"].astype(np.float32), d["nk"].astype(np.float32)

    def salva(self):


        if self.file is not None and not self.nk.any() and Path(self.file).exists():
            return
        if self.file is not None:
            Path(self.file).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.file, N=self.N, nk=self.nk)

    def azzera(self):
        self.N[:] = 0
        self.nk[:] = 0
        self._ppmi = None
        self._vet = None

    def impara(self, frase):
        ids = [self.idx[w] for w in frase if w in self.idx]
        for i, a in enumerate(ids):
            self.nk[a] += 1
            for d in range(1, FINESTRA + 1):
                if i + d < len(ids):
                    b = ids[i + d]
                    self.N[a, b] += 1.0 / d
                    self.N[b, a] += INDIETRO / d
        self._ppmi = None
        self._vet = None

    def ppmi(self):
        if self._ppmi is None:
            tot = float(self.N.sum())
            if tot <= 0:
                self._ppmi = np.zeros_like(self.N)
            else:
                P = self.N / tot
                pk = self.N.sum(1) / tot
                pj = self.N.sum(0) / tot
                with np.errstate(divide="ignore", invalid="ignore"):
                    pmi = np.log(P / (pk[:, None] * pj[None, :]))
                self._ppmi = np.where(self.N > 0, np.maximum(pmi, 0.0), 0.0).astype(np.float32)
        return self._ppmi

    def vettori(self):
        if self._vet is None:
            P = self.ppmi()
            U, S, _ = np.linalg.svd(P, full_matrices=False)
            V = U[:, :DIM] * np.sqrt(S[:DIM])
            self._vet = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        return self._vet

    def forza(self, k):
        P = self.ppmi()
        r = P[k] / (P[k].max() + 1e-9)
        sim = np.clip(self.vettori() @ self.vettori()[k], 0.0, None)
        sim[k] = 0
        return (1 - MISCELA) * r + MISCELA * sim * (self.nk > 0)

    def prossime(self, k, n=5, escludi=()):
        r = self.forza(k).copy()
        r[k] = -1
        for e in escludi:
            r[e] = -1
        o = np.argsort(-r)[:n]
        return [int(j) for j in o if r[j] > 0]

    def simili(self, k, n=3):
        V = self.vettori()
        s = V @ V[k]
        s[k] = -1
        return [int(j) for j in np.argsort(-s)[:n]]

    def catena(self, k, rng, lunghezza=5, tau=0.6):
        out = [k]
        for _ in range(lunghezza - 1):
            r = self.forza(out[-1]) + TRACCIA * self.forza(k)
            r[out] = 0
            piu = np.flatnonzero(r > 0)
            if len(piu) == 0:
                break
            p = np.exp((r[piu] - r[piu].max()) / tau)
            p /= p.sum()
            out.append(int(rng.choice(piu, p=p)))
        return out
