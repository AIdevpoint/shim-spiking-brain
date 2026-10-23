from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

MISCELA = 0.3
DIM = 100
PESO_FATTO = 3.0
REL_FATTO = 2.0

FINESTRA = 3
INDIETRO = 0.5


class AssociazioniGrandi:
    def __init__(self, parole, N, nk=None, file_extra=None, file_vettori=None):
        self.parole = list(parole)
        self.K = len(self.parole)
        self.idx = {w: i for i, w in enumerate(self.parole)}
        self.N = N.tocsr().astype(np.float32)
        self.nk = np.asarray(nk if nk is not None else (self.N.sum(1).A1 + self.N.sum(0).A1) / 2.0, np.float32)
        self.extra = {}
        self._ppmi = None
        self._vet = None
        self.file_extra, self.file_vettori = file_extra, file_vettori
        if file_extra is not None and Path(file_extra).exists():
            d = np.load(file_extra)
            for a, b, w in zip(d["a"], d["b"], d["w"]):
                self.extra.setdefault(int(a), {})[int(b)] = float(w)


    def salva(self):
        if self.file_extra is not None:
            Path(self.file_extra).parent.mkdir(parents=True, exist_ok=True)
            ter = [(a, b, w) for a, riga in self.extra.items() for b, w in riga.items()]
            np.savez(self.file_extra, a=np.array([t[0] for t in ter], np.int64), b=np.array([t[1] for t in ter], np.int64),
                     w=np.array([t[2] for t in ter], np.float32))

    def azzera(self):
        self.extra = {}

    def impara(self, frase):
        ids = [self.idx[w] for w in frase if w in self.idx]
        for i, a in enumerate(ids):
            for d in range(1, FINESTRA + 1):
                if i + d < len(ids):
                    b = ids[i + d]
                    if a != b:
                        self.extra.setdefault(a, {})[b] = self.extra.get(a, {}).get(b, 0.0) + PESO_FATTO / d
                        self.extra.setdefault(b, {})[a] = self.extra.get(b, {}).get(a, 0.0) + INDIETRO * PESO_FATTO / d


    def ppmi(self):
        if self._ppmi is None:
            N = self.N.tocoo()
            tot = float(N.sum())
            r = np.asarray(self.N.sum(1)).ravel()
            c = np.asarray(self.N.sum(0)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log(N.data * tot / (r[N.row] * c[N.col]))
            v = np.where(np.isfinite(v) & (v > 0), v, 0.0).astype(np.float32)
            self._ppmi = sp.csr_matrix((v, (N.row, N.col)), shape=N.shape)
            self._ppmi.eliminate_zeros()
        return self._ppmi

    def vettori(self):
        if self._vet is None:
            if self.file_vettori is not None and Path(self.file_vettori).exists():
                self._vet = np.load(self.file_vettori)
            else:
                from scipy.sparse.linalg import svds
                P = self.ppmi().astype(np.float64)
                U, S, _ = svds(P, k=min(DIM, self.K - 2))
                V = U * np.sqrt(S)
                self._vet = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
                if self.file_vettori is not None:
                    Path(self.file_vettori).parent.mkdir(parents=True, exist_ok=True)
                    np.save(self.file_vettori, self._vet)
        return self._vet

    def riga(self, k):
        P = self.ppmi()
        r = np.zeros(self.K, np.float32)
        a, b = P.indptr[k], P.indptr[k + 1]
        r[P.indices[a:b]] = P.data[a:b]
        m = max(float(r.max()), 1.0)
        for y, w in self.extra.get(k, {}).items():
            r[y] += w * m * REL_FATTO / PESO_FATTO
        return r

    def forza(self, k):
        r = self.riga(k)
        r = r / (r.max() + 1e-9)
        V = self.vettori()
        sim = np.clip(V @ V[k], 0.0, None)
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
