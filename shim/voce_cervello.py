from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

NB = 24
VOCALI = ["a", "e", "i", "o", "u"]
CONSONANTI = ["p", "b", "t", "d", "k", "g", "m", "n", "l", "r", "s", "f", "v", "ts", "tʃ"]
SCRITTE = {"k": "c", "ts": "z", "tʃ": "c"}
I_COCLEA_PA = 45.0


I_CONTRASTO_PA = float(__import__("os").environ.get("SHIM_CONTRASTO_PA", "80"))
CONTRASTO_DB, VOLUME_QUOTA = 12.0, 0.3
Z_INIZIO, Z_FINE, Z_LETTERA = 0.6, 0.25, 0.6
MAX_SILLABE, PAUSA_DOPO_S = 10, 1.5


class Coclea:
    def __init__(self, c):
        neuroni, banda = [], []
        for k in range(10):
            if True:
                nn = c.gruppo("udito_%d" % (k + 1))
                neuroni.append(nn)
                banda.append(np.minimum((np.arange(len(nn)) * NB) // max(len(nn), 1), NB - 1))
        self.neuroni = np.concatenate(neuroni)
        self.banda = np.concatenate(banda)
        self.fid = np.concatenate([np.full(len(nn), k) for k, nn in enumerate(neuroni)]) * NB + self.banda
        f = np.geomspace(100, 7000, NB + 1)
        self.f = f

    def correnti(self, pcm):
        pcm = np.asarray(pcm, np.float32)
        n_fin = int(math.ceil(len(pcm) / 1600))
        if n_fin == 0:
            return []
        lung, salto = 400, 160
        x = np.pad(pcm, (0, max(0, n_fin * 1600 + lung - len(pcm))))
        fr = np.lib.stride_tricks.sliding_window_view(x, lung)[::salto][:n_fin * 10] * np.hanning(lung)
        spettro = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        freq = np.fft.rfftfreq(lung, 1 / 16000)
        bande = np.stack([spettro[:, (freq >= a) & (freq < b)].sum(1) + 1e-10 for a, b in zip(self.f[:-1], self.f[1:])], 1)
        db = 10 * np.log10(bande)
        livello = np.clip((db - (db.max() - 50)) / 50, 0, 1)
        contrasto = np.clip((db - db.mean(1, keepdims=True)) / CONTRASTO_DB, 0, 1)
        contrasto *= (livello.mean(1, keepdims=True) > 0.2)
        spinta = I_CONTRASTO_PA * contrasto + I_COCLEA_PA * VOLUME_QUOTA * livello
        per_fin = spinta.reshape(n_fin, 10, NB).mean(1)
        return [v[self.banda].astype(np.float32) for v in per_fin]


class Bocca:
    def __init__(self, c, seme=11):
        nn = c.gruppo("bocca")
        rng = np.random.default_rng(seme)
        nn = rng.permutation(nn)
        self.unita = VOCALI + CONSONANTI
        self.gruppi = np.array_split(nn, len(self.unita))
        self.neuroni = np.concatenate(self.gruppi)
        self.dove = np.concatenate([np.full(len(g), k) for k, g in enumerate(self.gruppi)])
        self.quanti = np.array([len(g) for g in self.gruppi], np.float32)
        self.media = None
        self.var = None
        self.n = 0
        self.parla, self.sopra, self.sotto, self.sillabe, self.attesa = False, 0, 0, [], 0.0
        self.z_medio = 0.0

    def finestra(self, conta_bocca, sente, dt=0.1, propria=False):
        x = np.bincount(self.dove, weights=conta_bocca, minlength=len(self.unita)).astype(np.float32) / self.quanti / dt
        if self.media is None:
            self.media, self.var = x.copy(), np.full_like(x, 1.0)
        self.n += 1
        z = (x - self.media) / np.sqrt(self.var + 0.25)
        self.z_medio = float(z.mean())
        self.z_ultimo, self.x_ultimo = z.copy(), x.copy()

        self.z_liscio = 0.5 * getattr(self, "z_liscio", 0.0) + 0.5 * self.z_medio
        if not self.parla and not sente:
            a = 0.05 if self.n < 60 else 0.01
            self.media += a * (x - self.media)
            self.var += a * ((x - self.media) ** 2 - self.var)
        if self.n < 30:
            return None
        self.attesa = max(0.0, self.attesa - dt)
        if propria:
            self.attesa = max(self.attesa, 0.8)
        if not self.parla:
            self.sopra = self.sopra + 1 if self.z_liscio > Z_INIZIO else 0
            if self.sopra >= 2 and self.attesa <= 0:
                self.parla, self.sotto, self.sillabe = True, 0, []
            return None
        zc, zv = z[len(VOCALI):], z[:len(VOCALI)]
        cons = CONSONANTI[int(zc.argmax())] if zc.max() > Z_LETTERA else ""
        voc = VOCALI[int(zv.argmax())]
        self.sillabe.append((cons, voc))
        self.sotto = self.sotto + 1 if self.z_liscio < Z_FINE else 0
        if self.sotto >= 2 or len(self.sillabe) >= MAX_SILLABE:
            frase = self.sillabe[:-self.sotto] if self.sotto else self.sillabe
            self.parla, self.sillabe, self.attesa, self.sopra = False, [], PAUSA_DOPO_S, 0
            return frase or None
        return None


def scrivi(sillabe):
    out = []
    for c, v in sillabe:
        if c == "tʃ":
            s = "c" + v if v in "ie" else "ci" + v
        elif c == "k":
            s = ("ch" + v) if v in "ie" else "c" + v
        elif c == "g" and v in "ie":
            s = "gh" + v
        else:
            s = SCRITTE.get(c, c) + v
        out.append(s)

    parole, i = [], 0
    while i < len(out):
        k = 2 if (len(out) - i) != 3 else 3
        parole.append("".join(out[i:i + k]))
        i += k
    return " ".join(parole)


G_ARCUATO_PA = 35.0
MIN_ESEMPI = 3

LDA_SHR = float(os.environ.get("SHIM_LDA", "0.3")) or None
MEMORIA = 0.6


class Arcuato:

    def __init__(self, coclea, bocca, file=None):
        self.fid = coclea.fid
        self.nf = int(self.fid.max()) + 1
        self.nu = len(bocca.unita)
        self.nv = 5
        self.W = np.zeros((self.nu, self.nf), np.float32)
        self.esempi = np.zeros(self.nu, np.int64)
        self.mu = np.zeros(self.nf, np.float32)
        self.var = np.ones(self.nf, np.float32)
        self.n_norm = 0


        self.S = [np.zeros((self.nf, self.nf), np.float32) for _ in range(2)]
        self.nS = [0, 0]
        self.P = [None, None]
        self.b = [None, None]
        self.P_a = [-10 ** 9, -10 ** 9]
        self.riposo = None
        self.memoria = np.zeros(self.nu, np.float32)
        self.prove = np.zeros(self.nu, np.float32)
        self.forza = 0.0
        self.percepito = []
        self.file = file
        self.impara = True
        self.a_ora = None
        self.solo_adatta = False
        if file is not None and Path(file).exists():
            d = np.load(file)
            if d["W"].shape == self.W.shape and "mu" in d:
                self.W, self.esempi, self.mu, self.var = d["W"].astype(np.float32), d["esempi"], d["mu"], d["var"]
                self.n_norm = int(d["n_norm"])
                self.forza = float(d["forza"]) if "forza" in d else 0.0
                if "S0" in d:
                    self.S = [d["S0"].astype(np.float32), d["S1"].astype(np.float32)]
                    self.nS = [int(d["nS0"]), int(d["nS1"])]

    def salva(self):
        if self.file is not None:
            Path(self.file).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.file, W=self.W, esempi=self.esempi, mu=self.mu, var=self.var, n_norm=self.n_norm, forza=self.forza,
                     S0=self.S[0], S1=self.S[1], nS0=self.nS[0], nS1=self.nS[1])

    def azzera(self):
        for g in range(2):
            self.S[g][:] = 0
            self.nS[g] = 0
            self.P[g] = None
            self.P_a[g] = -10 ** 9
        self.W[:] = 0
        self.esempi[:] = 0
        self.mu[:] = 0
        self.var[:] = 1
        self.n_norm = 0
        self.forza = 0.0

    def z(self, a):
        return (a - self.mu) / np.sqrt(self.var + 1.0)

    def punteggi(self, a):
        z = self.z(a)
        out = np.zeros(self.nu, np.float32)
        for g, (lo, hi) in enumerate(((0, self.nv), (self.nv, self.nu))):
            if LDA_SHR is None or self.nS[g] < 30:
                out[lo:hi] = z @ self.W[lo:hi].T - 0.5 * (self.W[lo:hi] ** 2).sum(1)
                continue
            if self.P[g] is None or self.nS[g] - self.P_a[g] >= 50:
                Sg = self.S[g]
                Sg = (1 - LDA_SHR) * Sg + LDA_SHR * (np.trace(Sg) / self.nf) * np.eye(self.nf, dtype=np.float32)
                self.P[g] = np.linalg.solve(Sg + 1e-6 * np.eye(self.nf, dtype=np.float32), self.W[lo:hi].T).astype(np.float32)
                self.b[g] = 0.5 * (self.W[lo:hi].T * self.P[g]).sum(0)
                self.P_a[g] = self.nS[g]
            out[lo:hi] = z @ self.P[g] - self.b[g]
        return out

    def finestra(self, conta_coclea, sente, etichetta):
        f = np.bincount(self.fid, weights=conta_coclea, minlength=self.nf).astype(np.float32)
        if self.riposo is None:
            self.riposo = f.copy()
        if not sente:
            self.riposo += 0.02 * (f - self.riposo)
            self.prove *= 0.7
            return self._spinta()
        a = f - self.riposo
        self.a_ora = a if etichetta is None else None
        if getattr(self, "registra", None) is not None:
            self.registra.append((a.copy(), etichetta))
        if etichetta is not None:
            if self.impara and etichetta:
                self.forza += max(0.005, 1.0 / (self.n_norm + 1)) * (float(np.linalg.norm(a)) - self.forza)
                self.n_norm += 1
                b = max(0.005, 1.0 / self.n_norm)
                self.mu += b * (a - self.mu)
                self.var += b * ((a - self.mu) ** 2 - self.var)
                z = self.z(a)
                for k in ([] if self.solo_adatta else etichetta):
                    self.esempi[k] += 1
                    eta = max(0.02, 1.0 / self.esempi[k])
                    self.W[k] += eta * (z - self.W[k])
                    g = 0 if k < self.nv else 1
                    r = z - self.W[k]
                    self.nS[g] += 1
                    self.S[g] += max(0.002, 1.0 / self.nS[g]) * (np.outer(r, r) - self.S[g])


            self.prove *= 0.8
            return self._spinta()

        self.prove *= 0.8
        if np.linalg.norm(a) > 0.5 * self.forza:
            s = self.punteggi(a)
            for lo, hi in ((0, self.nv), (self.nv, self.nu)):
                noti = np.flatnonzero(self.esempi[lo:hi] >= MIN_ESEMPI) + lo
                if len(noti) >= 2:
                    o = noti[np.argsort(-s[noti])]
                    self.prove[o[0]] += 1.0
                    self.prove[o[1]] += 0.3
                    if lo == 0:
                        self.percepito.append(int(o[0]))
        return self._spinta()

    def _spinta(self):
        out = np.zeros(self.nu, np.float32)
        for lo, hi in ((0, self.nv), (self.nv, self.nu)):
            p = self.prove[lo:hi]
            if p.max() > 0.5:
                o = np.argsort(-p) + lo
                out[o[0]] = G_ARCUATO_PA
                if self.prove[o[1]] > 0.5:
                    out[o[1]] = G_ARCUATO_PA / 3
        return out
