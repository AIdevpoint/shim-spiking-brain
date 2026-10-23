from __future__ import annotations

import os

import numpy as np

N_NUM = 10
M_NUM = int(os.environ.get("SHIM_NUM_M", "60"))
K_SUCC = int(os.environ.get("SHIM_NUM_KSUCC", "15"))
G_SUCC = float(os.environ.get("SHIM_NUM_G", "1.5"))
SIGMA_LOG = 0.25
SIGMA_PAROLA = 0.08
ACCESO = os.environ.get("SHIM_NUMERI", "0") == "1"
G_TIENI = float(os.environ.get("SHIM_NUM_TIENI", "200"))
X_TIENI = float(os.environ.get("SHIM_NUM_X", "10"))

INIB_NUM = float(os.environ.get("SHIM_NUM_INIB", "30"))
TIC_S, PASSO_S = 0.15, 0.6
PAROLE_NUMERO = ["uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci"]


def impronta():
    return "|num%d,%d,%.3f" % (M_NUM, K_SUCC, G_SUCC) if ACCESO else ""


class Numeri:
    def __init__(self, c, seme=41):
        from shim.neuroni import DT
        rng = np.random.default_rng(seme)
        pool = rng.permutation(c.gruppo("ips"))
        if len(pool) < N_NUM * M_NUM:
            raise RuntimeError("troppo pochi neuroni nell'IPS: %d" % len(pool))
        self.gruppi = pool[:N_NUM * M_NUM].reshape(N_NUM, M_NUM).astype(np.int64)
        self.neuroni = self.gruppi.ravel()
        J = float(np.median(c.pesi[c.pesi > 0]))
        self.J = J
        pre, post = [], []
        tipo = []
        for p in range(N_NUM - 1):
            for j in self.gruppi[p + 1]:
                pre.append(rng.choice(self.gruppi[p], K_SUCC, replace=False))
                post.append(np.full(K_SUCC, j))
                tipo.append(np.zeros(K_SUCC, np.int8))
        for p in range(1, N_NUM):
            for j in self.gruppi[p - 1]:
                pre.append(rng.choice(self.gruppi[p], K_SUCC, replace=False))
                post.append(np.full(K_SUCC, j))
                tipo.append(np.ones(K_SUCC, np.int8))
        pre, post, tipo = np.concatenate(pre), np.concatenate(post), np.concatenate(tipo)
        rit = np.clip(np.round(rng.normal(1.5, 0.75, len(pre)) / DT), 1, 250).astype(np.uint8)
        from .memoria_lavoro import MemoriaLavoro
        fake = MemoriaLavoro.__new__(MemoriaLavoro)
        MemoriaLavoro._unisci(fake, c, pre, post, np.where(tipo == 0, G_SUCC * J, 0.0).astype(np.float32), rit, tipo)
        self.pos_succ, self.pos_pred = fake.pos[0], fake.pos[1]
        self.media = None
        self.var = None
        self.z = np.zeros(N_NUM, np.float32)
        self.n_fin = 0
        self.spinta = None
        self.via_fino = 0.0
        self.via_pa = float(os.environ.get("SHIM_NUM_VIA", "40"))
        self.s = np.zeros(N_NUM, np.float32)
        self.g_tieni, self.x_tieni, self.inib = G_TIENI, X_TIENI, INIB_NUM
        self.conteggio = None
        print("[numeri] %d gruppi x %d neuroni nell'IPS (%d neuroni E), %d sinapsi del successore (G %.2f)" % (
            N_NUM, M_NUM, len(pool), len(pre), G_SUCC), flush=True)

    def corrente(self, t, pa_pensa=300.0):
        out = np.zeros((N_NUM, M_NUM), np.float32)
        att = False
        if self.spinta is not None and self.spinta[2] > t:
            s, pa = self.spinta[0], self.spinta[1]
            sg = self.spinta[3] if len(self.spinta) > 3 else SIGMA_PAROLA
            prof = np.exp(-(np.log(np.arange(1, N_NUM + 1)) - np.log(s)) ** 2 / (2 * sg ** 2))
            out += (pa * prof)[:, None]
            att = True
        if self.via_fino > t:
            out += self.via_pa
            att = True
        if t < getattr(self, "svuota_fino", 0.0):
            out -= 300.0
            self.s[:] = 0.0
            return out.ravel()
        if self.g_tieni > 0 and self.s.any():
            out += (self.g_tieni * self.s - self.inib * (self.s.sum() - self.s))[:, None]
            att = True
        c = self.conteggio
        if c is not None and t < c["tic_fino"]:
            k = c["da"]
            spinta = np.full(N_NUM, self.via_pa, np.float32)
            spinta[k] = 0.0
            out += spinta[:, None]
            att = True
        if c is not None and t < c.get("stacca_fino", 0.0):
            out[c["staccato"]] -= 2.0 * self.g_tieni
            att = True
        return out.ravel() if att else None

    def avvia_conteggio(self, t, passi, verso=1):
        if getattr(self, "m", None) is not None:
            g = getattr(self, "g_lavoro", G_SUCC)
            self.imposta(self.m, g if verso > 0 else 0.0, 0.0 if verso > 0 else g)
        self.conteggio = {"prossimo_tic": t + 0.2, "tic_fino": 0.0, "passi": int(passi), "fatti": 0, "lista": [], "da": None}

    def riposo(self):
        if getattr(self, "m", None) is not None:
            self.imposta(self.m, 0.0, 0.0)

    def _passo(self, t):
        c = self.conteggio
        if c is None:
            return
        if c["tic_fino"] and t >= c["tic_fino"]:
            c["tic_fino"] = 0.0
            c["staccato"], c["stacca_fino"] = c["da"], t + 0.1
            self.s[c["da"]] = 0.0
            c["dopo_tic"] = t + 0.3
        if c.get("dopo_tic") and t >= c["dopo_tic"]:
            c["dopo_tic"] = None
            c["lista"].append(self.in_mente())
            c["fatti"] += 1
            c["prossimo_tic"] = t + PASSO_S - 0.4
            if c["fatti"] >= c["passi"]:
                c["finito"] = True
                self.riposo()
        if not c.get("finito") and not c["tic_fino"] and not c.get("dopo_tic") and t >= c["prossimo_tic"]:
            k = self.in_mente()
            c["da"] = (k - 1) if k else int(np.argmax(self.s))
            c["tic_fino"] = t + TIC_S

    def leggi(self, conta, dt=0.1):
        x = np.asarray(conta, np.float32).reshape(N_NUM, M_NUM).mean(1) / dt
        self.n_fin += 1
        if self.media is None:
            self.media, self.var = x.copy(), np.full_like(x, 1.0)
        z = (x - self.media) / np.sqrt(self.var + 0.25)
        self.z = 0.5 * self.z + 0.5 * z
        sopra = np.clip((x - self.media - self.x_tieni) / 10.0, 0.0, None)
        self.s = np.clip(self.s * np.exp(-1.0) + 0.5 * sopra * (1.0 - self.s), 0.0, 1.0)
        self.x = x
        if self.conteggio is not None:
            self._passo(getattr(self, "_t", 0.0))
        quieto = (self.spinta is None or self.spinta[2] < getattr(self, "_t", 0.0)) and self.conteggio is None and not self.s.any()
        if quieto:
            a = 0.05 if self.n_fin < 60 else 0.01
            self.media += a * (x - self.media)
            self.var += a * ((x - self.media) ** 2 - self.var)
        return self.z

    def in_mente(self, soglia=3.0):
        if self.s.max() > 0.5:
            return int(np.argmax(self.s)) + 1
        k = int(np.argmax(self.z))
        return (k + 1) if self.z[k] > soglia else None

    def grandezza(self, base=None):
        w = np.clip(self.z - (0.0 if base is None else np.clip(base, 0.0, None)), 0.0, None)
        if w.sum() < 3.0:
            return None
        return float(np.exp((w * np.log(np.arange(1, N_NUM + 1))).sum() / w.sum()))

    def svuota(self):
        self.s[:] = 0.0
        self.conteggio = None
        self.svuota_fino = getattr(self, "_t", 0.0) + 0.3
        self.riposo()

    def imposta(self, m, g_succ=None, g_pred=None):
        import torch
        if g_succ is not None:
            m.pesi[torch.as_tensor(self.pos_succ, device=m.pesi.device)] = float(g_succ) * self.J
            self.g_succ = float(g_succ)
        if g_pred is not None:
            m.pesi[torch.as_tensor(self.pos_pred, device=m.pesi.device)] = float(g_pred) * self.J
            self.g_pred = float(g_pred)
