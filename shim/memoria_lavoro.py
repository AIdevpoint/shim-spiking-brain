from __future__ import annotations

import os

import numpy as np

PREFRONTALE = ["46", "9-46d", "a9-46v", "p9-46v", "9a", "9p", "9m", "8Av", "8Ad", "8BL", "8BM", "8C", "10d", "10v", "10r", "10pp",
               "a10p", "p10p", "11l", "13l", "a47r", "p47r", "47l", "47m", "47s", "IFJa", "IFJp", "IFSa", "IFSp", "SFL", "s6-8", "i6-8",
               "FEF", "24dd", "24dv", "a24", "p24", "p32", "d32", "s32", "a32pr", "25", "pOFC", "OFC"]
M_PF = int(os.environ.get("SHIM_MEM_M", "40"))
K_IN = int(os.environ.get("SHIM_MEM_KIN", "20"))
K_FF = int(os.environ.get("SHIM_MEM_KFF", "10"))
K_FB = int(os.environ.get("SHIM_MEM_KFB", "5"))
G_REC = float(os.environ.get("SHIM_MEM_G", "1.0"))
G_FF = float(os.environ.get("SHIM_MEM_GFF", "1.0"))
G_FB = float(os.environ.get("SHIM_MEM_GFB", "0.5"))
Z_MEM = float(os.environ.get("SHIM_MEM_Z", "3.0"))
N_MEM = int(os.environ.get("SHIM_MEM_N", "1000"))
RIT_MS, RIT_LUNGO_MS = 1.5, 5.0
ACCESA = os.environ.get("SHIM_MEMORIA", "0") == "1"
SOGLIA_RELATIVA = os.environ.get("SHIM_MEM_XREL", "0") == "1"

SELETTIVO = os.environ.get("SHIM_MEM_SEL", "0") == "1"

G_NMDA = float(os.environ.get("SHIM_MEM_NMDA", "0"))
INIB_NMDA = float(os.environ.get("SHIM_MEM_INIB", "0"))
X_NMDA = float(os.environ.get("SHIM_MEM_XNMDA", "12.0"))
ALFA_NMDA = float(os.environ.get("SHIM_MEM_ALFA", "0.5"))


def impronta():
    if not ACCESA:
        return ""
    return ("|mem%d,%d,%d,%d,%.3f,%.3f,%.3f,%d" % (M_PF, K_IN, K_FF, K_FB, G_REC, G_FF, G_FB, N_MEM) + ("" if _K_COSTRUITE[0] in (None, N_MEM) else "|k%d" % _K_COSTRUITE[0])
            + ("|sel" if SELETTIVO else "") + ("|sep%s" % os.environ.get("SHIM_CODICI_SEPARATI") if os.environ.get("SHIM_CODICI_SEPARATI") else "")
            + ("|nomi2" if os.environ.get("SHIM_SEPARATI_NOMI") == "1" else ""))


_K_COSTRUITE = [None]


class MemoriaLavoro:
    def __init__(self, c, codici_parole, seme=31, scelte=None):
        from shim.neuroni import DT
        rng = np.random.default_rng(seme)
        self.pool = c.gruppo("pf_sx", "pf_dx")
        self.parola = np.asarray(list(scelte) if scelte is not None else range(min(len(codici_parole), N_MEM)), np.int64)
        self.K = len(self.parola)
        self.slot = {int(w): i for i, w in enumerate(self.parola)}
        _K_COSTRUITE[0] = self.K
        self.ass = np.stack([rng.choice(self.pool, M_PF, replace=False) for _ in range(self.K)]).astype(np.int64)
        J = float(np.median(c.pesi[c.pesi > 0]))
        self.J = J
        pre, post, tipo = [], [], []
        if SELETTIVO:
            from collections import Counter
            molt = Counter(int(x) for cod in codici_parole for x in np.asarray(cod).ravel())
        for k in range(self.K):
            a = self.ass[k]
            lin = np.asarray(codici_parole[int(self.parola[k])], np.int64)
            if SELETTIVO and len(lin):
                m_ = np.array([molt[int(x)] for x in lin])
                ordine_ = np.argsort(m_, kind="stable")
                quanti_ = max(int((m_ == 1).sum()), min(8, len(lin)))
                lin = lin[ordine_[:quanti_]]
            for j in a:
                altri = a[a != j]
                pre.append(rng.choice(altri, min(K_IN, len(altri)), replace=False)); post.append(np.full(min(K_IN, len(altri)), j)); tipo.append(np.full(min(K_IN, len(altri)), 0, np.int8))
                if len(lin):
                    s = rng.choice(lin, min(K_FF, len(lin)), replace=False)
                    pre.append(s); post.append(np.full(len(s), j)); tipo.append(np.full(len(s), 1, np.int8))
            for j in lin:
                s = rng.choice(a, min(K_FB, len(a)), replace=False)
                pre.append(s); post.append(np.full(len(s), j)); tipo.append(np.full(len(s), 2, np.int8))
        pre, post, tipo = np.concatenate(pre), np.concatenate(post), np.concatenate(tipo)
        g = np.array([G_REC, G_FF, G_FB], np.float32)[tipo]
        rit_ms = np.where(tipo == 0, RIT_MS, RIT_LUNGO_MS)
        rit = np.clip(np.round(rng.normal(rit_ms, rit_ms * 0.5) / DT), 1, 250).astype(np.uint8)
        self._unisci(c, pre, post, (g * J).astype(np.float32), rit, tipo)

        self.neuroni = np.unique(self.ass)
        pos = {int(n): i for i, n in enumerate(self.neuroni)}
        self.ass_loc = np.vectorize(pos.get)(self.ass).astype(np.int64)
        self.media = None
        self.var = None
        self.z = np.zeros(self.K, np.float32)
        self.n_fin = 0
        self.svuota_n = 0
        self.g_nmda, self.inib, self.alfa = G_NMDA, INIB_NMDA, ALFA_NMDA
        self.s = np.zeros(self.K, np.float32)
        print("[memoria di lavoro] %d assemblee x %d neuroni nella prefrontale (%d neuroni E), %d sinapsi nuove (ricorrenti %d, "
              "linguaggio->memoria %d, memoria->linguaggio %d), J %.0f pA, G %.2f/%.2f/%.2f" % (
                  self.K, M_PF, len(self.pool), len(pre), int((tipo == 0).sum()), int((tipo == 1).sum()), int((tipo == 2).sum()),
                  J, G_REC, G_FF, G_FB), flush=True)

    def _unisci(self, c, pre_n, post_n, peso_n, rit_n, tipo_n):
        n = c.n
        cont = np.diff(c.indptr)
        pre_v = np.repeat(np.arange(n, dtype=np.int64), cont)
        pre = np.concatenate([pre_v, pre_n.astype(np.int64)])
        del pre_v
        ordine = np.argsort(pre, kind="stable")
        n_v = len(c.indices)
        c.indices = np.concatenate([c.indices, post_n.astype(c.indices.dtype)])[ordine]
        c.pesi = np.concatenate([c.pesi, peso_n.astype(c.pesi.dtype)])[ordine]
        c.ritardi = np.concatenate([c.ritardi, rit_n.astype(c.ritardi.dtype)])[ordine]
        c.indptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))]).astype(c.indptr.dtype)
        c.max_ritardo = int(c.ritardi.max()) + 1
        c.n_sinapsi = len(c.indices)
        nuove = np.flatnonzero(ordine >= n_v)
        t = tipo_n[ordine[nuove] - n_v]
        self.pos = {k: nuove[t == k] for k in (0, 1, 2)}

    def imposta(self, m, g_rec=None, g_ff=None, g_fb=None):
        import torch
        for k, g in ((0, g_rec), (1, g_ff), (2, g_fb)):
            if g is not None:
                p = torch.as_tensor(self.pos[k], device=m.pesi.device)
                m.pesi[p] = float(g) * self.J
        self.g = {"rec": g_rec, "ff": g_ff, "fb": g_fb}

    def leggi(self, conta, dt=0.1):
        x = np.asarray(conta, np.float32)[self.ass_loc].mean(1) / dt
        self.n_fin += 1
        if self.media is None:
            self.media, self.var = x.copy(), np.full_like(x, 1.0)
        z = (x - self.media) / np.sqrt(self.var + 0.25)
        self.aggiorna_lenta(x, self.alfa)
        self.z = 0.5 * self.z + 0.5 * z
        tenute = self.z > Z_MEM
        a = 0.05 if self.n_fin < 60 else 0.005
        quiete = ~tenute
        self.media[quiete] += a * (x[quiete] - self.media[quiete])
        self.var[quiete] += a * ((x[quiete] - self.media[quiete]) ** 2 - self.var[quiete])
        self.svuota_n = max(0, self.svuota_n - 1)
        return self.z


    def corrente_lenta(self, g_nmda, inib, alfa):
        s = getattr(self, "s", None)
        if s is None:
            return None
        I_ass = g_nmda * s - inib * (s.sum() - s)
        out = np.zeros(len(self.neuroni), np.float32)
        np.add.at(out, self.ass_loc.ravel(), np.repeat(I_ass.astype(np.float32), self.ass_loc.shape[1]))
        return out

    def aggiorna_lenta(self, x, alfa, tau_s=0.1, dt=0.1):
        if getattr(self, "s", None) is None:
            self.s = np.zeros(self.K, np.float32)
        soglia = getattr(self, "x_nmda", X_NMDA) + (self.media if (SOGLIA_RELATIVA and self.media is not None) else 0.0)
        sopra = np.clip((x - soglia) / 10.0, 0.0, None)
        self.s = self.s * np.exp(-dt / tau_s) + alfa * sopra * (1.0 - self.s)
        self.s = np.clip(self.s, 0.0, 1.0)
        if self.svuota_n > 0:
            self.s[:] = 0.0

    def tenute(self, parole=None, n=5):
        if os.environ.get("SHIM_MEM_LEGGI_S", "0") == "1":
            ks = np.flatnonzero(self.s > 0.5)
            ks = ks[np.argsort(-self.z[ks])][:n]
            return [(parole[int(self.parola[k])] if parole is not None else int(self.parola[k]), round(float(max(self.z[k], 21.0)), 2)) for k in ks]
        ks = np.flatnonzero(self.z > Z_MEM)
        ks = ks[np.argsort(-self.z[ks])][:n]
        return [(parole[int(self.parola[k])] if parole is not None else int(self.parola[k]), round(float(self.z[k]), 2)) for k in ks]
