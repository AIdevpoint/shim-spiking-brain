from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .corpus_shim import PAROLE as PAROLE_BASE


def _vocabolario():
    import json
    f = (Path(__file__).resolve().parents[1] / "stato") / {"wiki": "shim_vocab_wiki.json", "2000": "shim_vocab_2000.json"}.get(
        os.environ.get("SHIM_CONOSCENZA", ""), "shim_vocab.json")
    if os.environ.get("SHIM_VOCAB", "1") == "1" and f.exists():
        try:
            v = json.loads(f.read_text(encoding="utf-8"))
            if v[:len(PAROLE_BASE)] == list(PAROLE_BASE):
                return v
        except Exception:
            pass
    return list(PAROLE_BASE)


PAROLE = _vocabolario()
from .associazioni_shim import Associazioni

ZONE_PAROLA = ["PSL", "STV", "STSdp", "STSvp", "TPOJ1", "TPOJ2", "A5", "STGa", "TA2"]
SEG = 4
LDA_SHR = 0.5
I_PAROLA_PA = float(os.environ.get("SHIM_I_PAROLA", "60"))
PUSH_FINESTRE = 6
Z_PAROLA = float(os.environ.get("SHIM_Z_PAROLA", "1.5"))
P_MIN = float(os.environ.get("SHIM_P_MIN", "0.6"))


T_SOFTMAX = float(os.environ.get("SHIM_T", "4"))
MAX_FINESTRE = 35
PAUSA_PAROLA = 40


CONFERMA = os.environ.get("SHIM_CONFERMA", "1") == "1"
I_SEME = 1.4
FINESTRE_CONFERMA = 60
K_DENSO = 2000
STRUTTURA = os.environ.get("SHIM_FRASI_SVO", "0") == "1"
SEPARATI_NOMI = os.environ.get("SHIM_SEPARATI_NOMI", "0") == "1"


def parole_separate(parole, sep):
    import json as _json
    fc = (Path(__file__).resolve().parents[1] / "stato") / "shim_classi_2000.json"
    nomi = set()
    if fc.exists():
        dc = _json.loads(fc.read_text(encoding="utf-8"))
        nomi = {w for w, c in zip(dc["parole"], dc["classi"]) if c == "nome"}
    try:
        from .lessico_2000 import SCUOLA, NON_NOMI
        nomi = (nomi | set(SCUOLA)) - NON_NOMI
    except ImportError:
        pass


    primi = [k for k in range(len(parole)) if parole[k] in nomi][:sep]
    resto = [k for k in range(len(parole)) if k not in set(primi)][:max(0, sep - len(primi))]
    return sorted(primi + resto)


LEGATE_STORIE = os.environ.get("SHIM_LEGATE_STORIE", "1") == "1"
PESO_CONTESTO = float(os.environ.get("SHIM_PESO_CONTESTO", "0.5"))
M_CODICE = int(os.environ.get("SHIM_M_CODICE", "40"))
PH, PW = int(os.environ.get("SHIM_PH", "12")), int(os.environ.get("SHIM_PW", "18"))


VISTA_ENERGIA = os.environ.get("SHIM_VISTA_ENERGIA", "1") == "1"
TACE_FINESTRE = 80


PENSA_DA_SOLO = os.environ.get("SHIM_PENSA", "1") == "1"
QUIETE_FINESTRE = int(os.environ.get("SHIM_QUIETE", "450"))
PAUSA_VELOCE = int(os.environ.get("SHIM_PAUSA_VELOCE", "4"))
PUSH_VELOCE = int(os.environ.get("SHIM_PUSH_VELOCE", "3"))
PUSH_PENSIERO = 36
CANDIDATE = 3
RUMORE_PENSIERO = 0.35
RUMORE_RISPOSTA = 0.10
PESO_VERBO = 1.5


NOVITA = os.environ.get("SHIM_NOVITA", "1") == "1"
NOVITA_SD = float(os.environ.get("SHIM_NOVITA_SD", "2.5"))
STRUTTURA_MIN = float(os.environ.get("SHIM_STRUTTURA_MIN", "0.6"))


class _Novita:
    Sinv = None
    d2_n = 0
    d2_m = 0.0
    d2_v = 0.0
    r_m = 0.0
    r_v = 0.0

    Sinv_n = None
    Sinv_a = -10 ** 9

    def _calibra(self, x, k):
        if self.nS < 20 or self.n[k] < 2:
            return
        if self.Sinv_n is None or self.nS - self.Sinv_a >= 30:
            Dn = np.sqrt(np.diag(self.S) + 1e-6)
            R = self.S / (Dn[:, None] * Dn[None, :])
            Sig = (1 - LDA_SHR) * R + LDA_SHR * np.eye(self.dim, dtype=np.float32)
            self.Dn = Dn
            self.Sinv_n = np.linalg.inv(Sig + 1e-6 * np.eye(self.dim, dtype=np.float32)).astype(np.float32)
            self.Sinv_a = self.nS
        z = (x - self.M[k]) / self.Dn
        d2 = float(z @ self.Sinv_n @ z) / self.dim
        zg = (x - self.M[self.n > 0].mean(0)) / self.Dn
        rapporto = d2 / (float(zg @ self.Sinv_n @ zg) / self.dim + 1e-6)
        self.d2_n += 1
        al = max(0.02, 1.0 / self.d2_n)
        dm = d2 - self.d2_m
        self.d2_m += al * dm
        self.d2_v += al * (dm * (d2 - self.d2_m) - self.d2_v)
        dr = rapporto - self.r_m
        self.r_m += al * dr
        self.r_v += al * (dr * (rapporto - self.r_m) - self.r_v)

    def sconosciuta(self, x, s):
        if not NOVITA or self.Sinv is None or self.D is None or self.d2_n < 15:
            return False
        z = x / self.D
        d2min = (float(z @ self.Sinv @ z) - 2.0 * float(np.max(s))) / self.dim
        soglia1 = self.d2_m + NOVITA_SD * float(np.sqrt(max(self.d2_v, 1e-9)))
        noti = self.n > 0
        zg = (x - self.M[noti].mean(0)) / self.D
        rapporto = d2min / (float(zg @ self.Sinv @ zg) / self.dim + 1e-6)
        soglia2 = self.r_m + NOVITA_SD * float(np.sqrt(max(self.r_v, 1e-9)))
        self.ultimo_novita = (d2min, soglia1, rapporto, soglia2)
        if os.environ.get("SHIM_DBG") == "1":
            print("[NOVITA] d2min=%.2f (soglia %.2f)  rapporto=%.2f (soglia %.2f)" % self.ultimo_novita, flush=True)
        return d2min > soglia1 or rapporto > soglia2

    def _azzera_novita(self):
        self.Sinv = None
        self.Sinv_n = None
        self.Sinv_a = -10 ** 9
        self.d2_n, self.d2_m, self.d2_v = 0, 0.0, 0.0
        self.r_m, self.r_v = 0.0, 0.0


class Flusso(_Novita):

    def __init__(self, K, dim):
        self.K, self.dim = K, dim
        self.M = np.zeros((K, dim), np.float32)
        self.n = np.zeros(K, np.int64)
        self.S = np.zeros((dim, dim), np.float32)
        self.nS = 0
        self.P = self.b = self.D = self.noti = None
        self.P_a = -10 ** 9

    def azzera(self):
        self.M[:] = 0
        self.n[:] = 0
        self.S[:] = 0
        self.nS = 0
        self.P = None
        self.P_a = -10 ** 9
        self._azzera_novita()

    def insegna(self, x, k):
        self._calibra(x, k)
        self.n[k] += 1
        if self.n[k] == 1:
            self.P = None
        self.M[k] += (x - self.M[k]) / self.n[k]
        r = x - self.M[k]
        self.nS += 1
        self.S += max(0.002, 1.0 / self.nS) * (np.outer(r, r) - self.S)

    def punteggi(self, x):
        noti = self.n > 0
        if self.nS < 20 or noti.sum() < 2:
            return None
        if self.P is None or self.nS - self.P_a >= 30:
            D = np.sqrt(np.diag(self.S) + 1e-6)
            R = self.S / (D[:, None] * D[None, :])
            Sig = (1 - LDA_SHR) * R + LDA_SHR * np.eye(self.dim, dtype=np.float32)
            Mz = self.M / D
            P = np.linalg.solve(Sig + 1e-6 * np.eye(self.dim, dtype=np.float32), Mz.T).astype(np.float32)
            self.b = 0.5 * (Mz.T * P).sum(0)
            self.Sinv = np.linalg.inv(Sig + 1e-6 * np.eye(self.dim, dtype=np.float32)).astype(np.float32)
            self.D = D
            self.noti = noti.copy()
            self.P_a = self.nS
            self.P = P
        P, b, D, noti_P = self.P, self.b, self.D, self.noti
        if P is None or b is None or D is None or noti_P is None:
            return None
        return np.where(noti_P, (x / D) @ P - b, -1e9)


def _riduci(im):
    if VISTA_ENERGIA:
        im = np.abs(im)
    gh, gw = im.shape
    righe = np.array_split(np.arange(gh), PH)
    colonne = np.array_split(np.arange(gw), PW)
    return np.array([[im[np.ix_(a, b)].mean() for b in colonne] for a in righe], np.float32).ravel()


class Lessico(_Novita):
    n_sconosciute = 0
    n_sconosciute_img = 0
    buf_liv = []
    liv_n = 0
    liv_m = 0.0

    def __init__(self, c, nf, parole=None, seme=17, file=None):
        self.parole = list(parole or PAROLE)
        K = self.K = len(self.parole)
        nn = [c.gruppo("parole")]
        nn = np.concatenate(nn) if nn else np.zeros(0, np.int64)
        if len(nn) < 6 * M_CODICE:
            raise RuntimeError("troppo pochi neuroni per %d parole nelle zone del linguaggio: %d" % (K, len(nn)))
        nn = np.random.default_rng(seme).permutation(nn)
        n = len(nn)
        self.neuroni = nn


        self.sparso = K > K_DENSO
        sep = int(os.environ.get("SHIM_CODICI_SEPARATI", "0"))
        lista_sep = parole_separate(self.parole, sep) if SEPARATI_NOMI and sep > 0 else None
        if self.sparso and lista_sep is not None and len(lista_sep) * M_CODICE < n:
            rg = np.random.default_rng(seme + 1)
            self.codici = np.zeros((K, M_CODICE), np.int32)
            dentro = np.zeros(K, bool)
            dentro[lista_sep] = True
            self.codici[lista_sep] = np.arange(len(lista_sep) * M_CODICE).reshape(len(lista_sep), M_CODICE)
            resto = np.arange(len(lista_sep) * M_CODICE, n)
            for k in np.flatnonzero(~dentro):
                self.codici[k] = resto[rg.choice(len(resto), M_CODICE, replace=False)]
            self.Gm = None
            self.quanti = np.full(K, float(M_CODICE), np.float32)
        elif self.sparso and sep > 0 and sep * M_CODICE < n:
            rg = np.random.default_rng(seme + 1)
            propri = np.arange(sep * M_CODICE).reshape(sep, M_CODICE)
            resto = np.arange(sep * M_CODICE, n)
            altri = [resto[rg.choice(len(resto), M_CODICE, replace=False)] for _ in range(K - sep)]
            self.codici = np.concatenate([propri, np.stack(altri)] if altri else [propri]).astype(np.int32)
            self.Gm = None
            self.quanti = np.full(K, float(M_CODICE), np.float32)
        elif self.sparso:
            rg = np.random.default_rng(seme + 1)
            self.codici = np.stack([rg.choice(n, M_CODICE, replace=False) for _ in range(K)]).astype(np.int32)
            self.Gm = None
            self.quanti = np.full(K, float(M_CODICE), np.float32)
        else:
            Gm = np.zeros((K, n), np.float32)
            if n // K >= M_CODICE:
                for k, g in enumerate(np.array_split(np.arange(n), K)):
                    Gm[k, g] = 1.0
            else:
                rg = np.random.default_rng(seme + 1)
                for k in range(K):
                    Gm[k, rg.choice(n, M_CODICE, replace=False)] = 1.0
            self.Gm = Gm
            self.quanti = Gm.sum(1).astype(np.float32)
        self.n_pool = n
        self.gruppi = None
        self.nf = nf
        self.dim = nf * SEG
        self.M = np.zeros((K, self.dim), np.float32)
        self.n = np.zeros(K, np.int64)
        self.S = np.zeros((self.dim, self.dim), np.float32)
        self.nS = 0
        self.P = None
        self.b = None
        self.D = None
        self.P_a = -10 ** 9
        self.buf = []
        self.silenzio = 0
        self.insegnando = None
        self.impara = True
        self.finalizzati = 0
        self.ultima = None
        self.push = np.zeros(K, np.float32)
        self.push_n = 0
        self.tace = 0
        self.dopo_push = 0
        self.k_riconosciuta = -1
        self.conferma_k = set()
        self.riconosciuta_n = 0
        self.noti_P = None
        self.media = None
        self.var = None
        self.n_fin = 0
        self.z_liscio = np.zeros(K, np.float32)
        self.attesa = 0
        self.z_ultimo = np.zeros(K, np.float32)
        self.file = file
        self.assoc = None
        if os.environ.get("SHIM_CONOSCENZA") == "wiki" and file is not None:
            fw = Path(str(file)).with_name("shim_assoc_wiki.npz")
            if fw.exists():
                import scipy.sparse as _sp
                from .associazioni_grandi import AssociazioniGrandi
                d_ = np.load(fw)
                N_ = _sp.csr_matrix((d_["data"], d_["indices"], d_["indptr"]), shape=tuple(d_["shape"]))
                if N_.shape[0] == K:
                    self.assoc = AssociazioniGrandi(self.parole, N_, nk=d_["nk"], file_extra=Path(str(file)).with_name("shim_fatti_wiki.npz"),
                                                    file_vettori=Path(str(file)).with_name("shim_vettori_wiki.npy"))
        if self.assoc is None and K > 5000:
            import scipy.sparse as _sp
            from .associazioni_grandi import AssociazioniGrandi
            self.assoc = AssociazioniGrandi(self.parole, _sp.csr_matrix((K, K), dtype=np.float32), nk=np.zeros(K, np.float32))
        if self.assoc is None:
            nome_a = "shim_assoc_2000.npz" if os.environ.get("SHIM_CONOSCENZA") == "2000" else "shim_assoc.npz"
            self.assoc = Associazioni(self.parole, file=(Path(str(file)).with_name(nome_a) if file is not None else None))
        self.pensiero = None
        self.proximo_push_n = 0
        self.quiete = 0
        self.rng = np.random.default_rng(23)
        self.vista = Flusso(K, PH * PW)
        self.buf_img = []
        self.buf_liv = []
        self.insegnando_img = None
        self.finalizzati_img = 0
        self.ultima_img = None
        if file is not None and Path(file).exists():
            d = np.load(file)
            k0 = d["M"].shape[0]
            if k0 <= K and d["M"].shape[1] == self.M.shape[1]:
                self.M[:k0], self.n[:k0] = d["M"].astype(np.float32), d["n"]
                self.S, self.nS = d["S"].astype(np.float32), int(d["nS"])
            if "Mv" in d.files and d["Mv"].shape[0] <= K and d["Mv"].shape[1] == self.vista.M.shape[1]:
                k1 = d["Mv"].shape[0]
                self.vista.M[:k1], self.vista.n[:k1] = d["Mv"].astype(np.float32), d["nv"]
                self.vista.S, self.vista.nS = d["Sv"].astype(np.float32), int(d["nSv"])


    def salva(self):
        if self.file is not None:
            Path(self.file).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.file, M=self.M, n=self.n, S=self.S, nS=self.nS, Mv=self.vista.M, nv=self.vista.n,
                     Sv=self.vista.S, nSv=self.vista.nS)
            self.assoc.salva()

    def azzera(self):
        self.M[:] = 0
        self.n[:] = 0
        self.S[:] = 0
        self.nS = 0
        self.P = None
        self.P_a = -10 ** 9
        self._azzera_novita()
        self.ultima = None
        self.buf = []
        self.buf_img = []
        self.buf_liv = []
        self.liv_n, self.liv_m = 0, 0.0
        self.vista.azzera()
        self.assoc.azzera()
        self._assoc_ok = False
        self.pensiero = None
        self.proximo_push_n = 0
        self.conferma_k = set()
        self.seme_pendente = None
        self.ultima_img = None
        self.silenzio = 0
        self.push[:] = 0
        self.push_n = 0
        self.dopo_push = 0
        self.tace = 0
        self.attesa = 0
        self.z_liscio[:] = 0


    def ascolta(self, a, sente, propria):
        if propria:
            self.buf = []
            self.silenzio = 0
            return
        if sente and a is not None:
            self.buf.append(np.asarray(a, np.float32))
            self.silenzio = 0
        elif self.buf:
            self.silenzio += 1
            if self.silenzio >= 1:
                self._finalizza()

    def _caratteristica(self):
        if len(self.buf) < 2 or len(self.buf) > MAX_FINESTRE:
            return None
        A = np.stack(self.buf)
        if len(A) < SEG:
            A = np.repeat(A, int(np.ceil(SEG / len(A))), axis=0)
        pezzi = np.array_split(np.arange(len(A)), SEG)
        return np.concatenate([A[p].mean(0) for p in pezzi]).astype(np.float32)

    def _finalizza(self):
        x = self._caratteristica()
        self.ultimo_x = x
        self.buf = []
        self.silenzio = 0
        if x is not None:
            if self.insegnando is not None:
                if self.impara:
                    self._insegna(x, self.insegnando)
            else:
                self._riconosci(x)
        self.finalizzati += 1

    def _insegna(self, x, k):
        self._calibra(x, k)
        self.n[k] += 1
        if self.n[k] == 1:
            self.P = None
        self.M[k] += (x - self.M[k]) / self.n[k]
        r = x - self.M[k]
        self.nS += 1
        self.S += max(0.002, 1.0 / self.nS) * (np.outer(r, r) - self.S)

    def punteggi(self, x):
        noti = self.n > 0
        if self.nS < 30 or noti.sum() < 2:
            return None
        if self.P is None or self.nS - self.P_a >= 50:
            D = np.sqrt(np.diag(self.S) + 1e-6)
            R = self.S / (D[:, None] * D[None, :])
            Sig = (1 - LDA_SHR) * R + LDA_SHR * np.eye(self.dim, dtype=np.float32)
            Mz = self.M / D
            P = np.linalg.solve(Sig + 1e-6 * np.eye(self.dim, dtype=np.float32), Mz.T).astype(np.float32)
            self.b = 0.5 * (Mz.T * P).sum(0)
            self.Sinv = np.linalg.inv(Sig + 1e-6 * np.eye(self.dim, dtype=np.float32)).astype(np.float32)
            self.D = D
            self.noti_P = noti.copy()
            self.P_a = self.nS
            self.P = P
        P, b, D, noti_P = self.P, self.b, self.D, self.noti_P
        if P is None or b is None or D is None or noti_P is None:
            return None
        s = (x / D) @ P - b
        s = np.where(noti_P, s, -1e9)
        return s

    def guarda(self, im):
        if im is not None:
            self.buf_img.append(_riduci(np.asarray(im, np.float32)))
            if len(self.buf_img) < 8:
                return
        if not self.buf_img:
            return
        B = self.buf_img[1:] or self.buf_img
        x = np.mean(B, 0).astype(np.float32)
        self.buf_img = []


        struttura = float(x.max() / (np.median(x) + 1e-9)) if VISTA_ENERGIA else None
        if self.insegnando_img is not None:
            if self.impara:
                self.vista.insegna(x, self.insegnando_img)
                if struttura is not None:
                    self.liv_n += 1
                    self.liv_m += max(0.02, 1.0 / self.liv_n) * (struttura - self.liv_m)
        else:
            s = self.vista.punteggi(x)
            if s is not None:
                if os.environ.get("SHIM_DBG") == "1" and struttura is not None:
                    print("[STRUTTURA] %.2f (insegnate: %.2f, n=%d)" % (struttura, self.liv_m, self.liv_n), flush=True)
                if struttura is not None and self.liv_n >= 10 and (struttura - 1.0) < STRUTTURA_MIN * (self.liv_m - 1.0):
                    self.n_sconosciute_img += 1
                    self.ultima_img = None
                elif self.vista.sconosciuta(x, s):
                    self.n_sconosciute_img += 1
                    self.ultima_img = None
                else:
                    self._spingi(s, "img")
        self.finalizzati_img += 1

    def _riconosci(self, x):
        s = self.punteggi(x)
        if s is None:
            return
        if self.sconosciuta(x, s):
            self.n_sconosciute += 1
            self.ultima = None
            return
        self._spingi(s, "udito")

    def _spingi(self, s, quale):
        T = T_SOFTMAX or max(1.0, np.sqrt(self.dim) / 2.0)
        e = np.exp((s - s.max()) / T)
        p = e / e.sum()
        k = int(p.argmax())
        if quale == "img":
            self.ultima_img = (k, float(p[k]))
        else:
            self.ultima = (k, float(p[k]))
        if p[k] >= P_MIN:
            self.push[:] = 0
            self.push[k] = I_PAROLA_PA
            self.push_n = PUSH_FINESTRE
            self.k_riconosciuta = k
            self.conferma_k = {k}
            self.riconosciuta_n = FINESTRE_CONFERMA
            self.tace = TACE_FINESTRE


    def avvia_pensiero(self, k, lunghezza=4, veloce=False):
        self.pensiero = {"seme": k, "dette": [], "rimasti": int(lunghezza) - 1, "veloce": bool(veloce)}
        if STRUTTURA and getattr(self, "_classe", None) is None:
            self._struttura(np.zeros(self.K, np.float32), {"svo": True, "dette": [k], "seme": k})
        if STRUTTURA and self._classe[k] == 1:
            if os.environ.get("SHIM_CONOSCENZA") == "2000" and self.parole[k] not in self._agenti_scuola():
                self.pensiero.update(legate=True, rimasti=2)
            else:
                self.pensiero.update(svo=True, rimasti=2)
        self.proximo_push_n = 0
        self.quiete = 0
        self.tace = TACE_FINESTRE
        if self.attesa > 0:
            self.seme_pendente = k
            self.push[:] = 0
            return
        self._spingi_seme(k)

    seme_pendente = None

    def _agenti_scuola(self):
        if getattr(self, "_agenti", None) is None:
            from .corpus_shim import CATEGORIE
            self._agenti = set(CATEGORIE["animali"]) | set(CATEGORIE["persone"])
        return self._agenti

    def _spingi_seme(self, k):
        self.seme_pendente = None
        self.push[:] = 0
        self.push[k] = I_SEME * I_PAROLA_PA
        self.push_n = PUSH_FINESTRE
        self.k_riconosciuta = k
        self.conferma_k = {k}
        self.riconosciuta_n = FINESTRE_CONFERMA

    def avvia_risposta(self, cues, bugia=False):
        ks = [int(k) for k in cues]
        self.pensiero = {"seme": ks[0], "cues": ks, "dette": [], "rimasti": 0, "bugia": bool(bugia)}
        self.proximo_push_n = max(2, int(self.attesa) + 1)
        self.quiete = 0
        self.tace = TACE_FINESTRE

    def avvia_analogia(self, a, b, c):
        V = self.assoc.vettori()
        r = V @ (V[b] - V[a] + V[c])
        for k in (a, b, c):
            r[k] = -1e9
        top = float(np.max(r))
        r = np.exp((r - top) / 0.05).astype(np.float32)
        r[[a, b, c]] = 0.0
        self.pensiero = {"seme": int(a), "dette": [], "rimasti": 0, "r": r}
        self.proximo_push_n = max(2, int(self.attesa) + 1)
        self.quiete = 0
        self.tace = TACE_FINESTRE

    def avvia_relazione(self, rel, x):
        from .relazioni_scuola import RELAZIONI
        esempi = [(a, b) for a, b in RELAZIONI.get(rel, []) if a != self.parole[x] and a in self._idx_parole() and b in self._idx_parole()]
        if not esempi:
            return False
        ix = self._idx_parole()
        V = self.assoc.vettori()
        rv = np.mean([V[ix[b]] - V[ix[a]] for a, b in esempi], 0)
        r = V @ (V[x] + rv)
        r[x] = -1e9
        if rel in ix:
            r[ix[rel]] = -1e9
        top = float(np.max(r))
        r = np.exp((r - top) / 0.05).astype(np.float32)
        r[x] = 0.0
        self.pensiero = {"seme": int(x), "dette": [], "rimasti": 0, "r": r}
        self.proximo_push_n = max(2, int(self.attesa) + 1)
        self.quiete = 0
        self.tace = TACE_FINESTRE
        return True

    def _idx_parole(self):
        if getattr(self, "_ix_parole", None) is None:
            self._ix_parole = {w: i for i, w in enumerate(self.parole)}
        return self._ix_parole

    def _struttura(self, r, pz):
        if not STRUTTURA or not (pz.get("svo") or pz.get("legate")):
            return r
        if getattr(self, "_ordine", None) is None:
            from .frasi_shim import VERBI, QUI as _Q
            from .corpus_shim import CATEGORIE
            f = Path(_Q) / ("shim_ordine_2000.npy" if os.environ.get("SHIM_CONOSCENZA") == "2000" else "shim_ordine.npy")
            O = np.load(f) if f.exists() else np.zeros((self.K, self.K), np.float32)
            if O.shape[0] != self.K:
                O = np.zeros((self.K, self.K), np.float32)
            self._ordine = O / (O.sum(1, keepdims=True) + 1e-9)
            nomi = {w for c in ("animali", "cibo", "natura", "casa", "persone") for w in CATEGORIE[c]} - {"ciao"}
            self._nomi_scuola = nomi
            self._forte = np.array([w in nomi for w in self.parole])
            self._classe = np.array([2 if w in VERBI else (1 if w in nomi else 0) for w in self.parole])
            fc = Path(_Q) / "shim_classi_2000.json"
            if os.environ.get("SHIM_CONOSCENZA") == "2000" and fc.exists():
                import json as _json
                dc = _json.loads(fc.read_text(encoding="utf-8"))
                cl = dict(zip(dc["parole"], dc["classi"]))
                scuola = set(VERBI) | nomi
                self._classe = np.array([self._classe[i] if w in scuola else {"verbo": 2, "nome": 1}.get(cl.get(w), 0)
                                         for i, w in enumerate(self.parole)])
                Pc = np.asarray(dc["conteggi"], np.float64)
                art = Pc[:, 1] / (Pc.sum(1) + 1e-9)
                ix_c = {w: i for i, w in enumerate(dc["parole"])}
                self._forte = np.array([(w in nomi) or (cl.get(w) == "nome" and art[ix_c[w]] >= 0.5) for w in self.parole])
                from .lessico_2000 import NON_NOMI
                no_ = np.array([w in NON_NOMI for w in self.parole])
                self._forte &= ~no_
                self._classe = np.where(no_ & (self._classe == 1), 0, self._classe)
                fs_ = Path(_Q) / "shim_assoc_storie.npz"
                if LEGATE_STORIE and fs_.exists():
                    from .associazioni_shim import Associazioni
                    self._assoc_storie = Associazioni(self.parole, fs_)
        from .associazioni_shim import TRACCIA
        if pz.get("legate"):
            Pp = self.assoc.ppmi()
            k0 = pz["seme"]
            r2 = Pp[k0] / (Pp[k0].max() + 1e-9) + self.assoc.forza(k0)
            S_ = getattr(self, "_assoc_storie", None)
            if S_ is not None and S_.nk[k0] > 0:
                Ps = S_.ppmi()
                r2 = np.sqrt(r2 * (Ps[k0] / (Ps[k0].max() + 1e-9) + S_.forza(k0)))
            r2 = r2 * self._forte
            return r2.astype(np.float32)
        O, cl = self._ordine, self._classe
        if getattr(self, "_scuola_mask", None) is None:
            from .frasi_shim import VERBI as _V
            from .corpus_shim import CATEGORIE as _C
            vs = set(_C["azioni"])
            self._scuola_mask = (np.array([w in vs for w in self.parole]), np.array([w in self._nomi_scuola for w in self.parole]))
            self._ordine_scuola = None
            from .frasi_shim import QUI as _Q2
            fo = Path(_Q2) / "shim_ordine.npy"
            if fo.exists():
                o_ = np.load(fo)
                Os = np.zeros((self.K, self.K), np.float32)
                k_ = min(o_.shape[0], self.K)
                Os[:k_, :k_] = o_[:k_, :k_]
                self._ordine_scuola = Os / (Os.sum(1, keepdims=True) + 1e-9)
        n = len(pz["dette"])
        scuola = os.environ.get("SHIM_CONOSCENZA") == "2000" and self.parole[pz["seme"]] in self._nomi_scuola
        if scuola and self._ordine_scuola is not None:
            O = self._ordine_scuola
        av = O[pz["dette"][-1]] / (O[pz["dette"][-1]].max() + 1e-9)
        if n == 1:
            m_ = self._scuola_mask[0] if scuola else (cl == 2)
            r = (0.3 * r + av) * np.where(m_, 3.0, 0.05)
        else:
            s0 = O[pz["seme"]] / (O[pz["seme"]].max() + 1e-9)
            m_ = self._scuola_mask[1] if scuola else (cl == 1)
            r = (0.3 * r + av + TRACCIA * s0) * np.where(m_, 3.0, 0.05)
        return r.astype(np.float32)

    FILASTROCCA = ["uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci"]

    def avvia_conteggio(self, k, passi, verso=1):
        self.pensiero = {"seme": int(k), "dette": [], "rimasti": int(passi), "veloce": True, "conta": int(verso)}
        self.proximo_push_n = 0
        self.quiete = 0
        self.tace = TACE_FINESTRE
        if self.attesa > 0:
            self.seme_pendente = k
            self.push[:] = 0
            return
        self._spingi_seme(k)

    def _spinta_pensiero(self):
        pz = self.pensiero
        if pz is None:
            return
        from .associazioni_shim import TRACCIA
        if pz.get("conta"):
            ult = self.parole[pz["dette"][-1]] if pz["dette"] else None
            r = np.zeros(self.K, np.float32)
            if ult in self.FILASTROCCA:
                j = self.FILASTROCCA.index(ult) + pz["conta"]
                if 0 <= j < len(self.FILASTROCCA) and self.FILASTROCCA[j] in self._idx_parole():
                    r[self._idx_parole()[self.FILASTROCCA[j]]] = 1.0
            if not r.any():
                self.pensiero = None
                return
            self.push[:] = 0
            k = int(np.argmax(r))
            self.push[k] = I_PAROLA_PA * 1.2
            self.push_n = PUSH_FINESTRE
            self.tace = TACE_FINESTRE
            self.k_riconosciuta = k
            self.conferma_k = {k}
            self.riconosciuta_n = FINESTRE_CONFERMA
            return
        if "r" in pz:
            r = pz["r"].copy()
        elif "cues" in pz:
            c0, resto = pz["cues"][0], pz["cues"][1:]
            r = self.assoc.forza(c0) + PESO_VERBO * sum((self.assoc.forza(k) for k in resto), np.zeros(self.K, np.float32))
            for k in pz["cues"]:
                r[k] = 0
        elif pz["dette"]:
            ultimo, seme = pz["dette"][-1], pz["seme"]
            r = self.assoc.forza(ultimo) + TRACCIA * self.assoc.forza(seme)
            r = self._struttura(r, pz)
        else:
            return
        ctx = [k for k in getattr(self, "contesto", []) if k not in pz["dette"] and k not in pz.get("cues", ())]
        if ctx and "r" not in pz:
            r = r + PESO_CONTESTO * sum((self.assoc.forza(k) for k in ctx), np.zeros(self.K, np.float32))
        for k in pz["dette"]:
            r[k] = 0
        if pz.get("bugia") and np.any(r > 0):
            r[int(np.argmax(r))] = 0
        piu = np.flatnonzero(r > 0)
        if len(piu) == 0:
            self.pensiero = None
            return
        top = piu[np.argsort(-r[piu])[:CANDIDATE]]
        self.push[:] = 0
        for k in top:
            rum = RUMORE_RISPOSTA if ("cues" in pz or "r" in pz) else RUMORE_PENSIERO
            self.push[k] = I_PAROLA_PA * float(r[k] / r[top[0]]) * float(1.0 + rum * self.rng.standard_normal())
        self.push = np.clip(self.push, 0.0, 1.6 * I_PAROLA_PA)
        self.push_n = PUSH_FINESTRE
        self.tace = TACE_FINESTRE
        self.k_riconosciuta = int(top[0])
        self.conferma_k = {int(x) for x in top}
        self.riconosciuta_n = FINESTRE_CONFERMA

    def _parola_uscita(self, k):
        self.quiete = 0
        pz = self.pensiero
        if pz is None:
            self.catena = [self.parole[k]]
            return
        pz["dette"].append(int(k))
        self.catena = [self.parole[j] for j in pz["dette"]]
        self.catena_veloce = bool(pz.get("veloce")) and not pz.get("conta") and not pz.get("concetto")
        self.catena_forma = "legate" if pz.get("legate") else ""
        if pz["rimasti"] > 0:
            pz["rimasti"] -= 1
            self.proximo_push_n = PUSH_VELOCE if pz.get("veloce") else PUSH_PENSIERO
        else:
            self.pensiero = None


    def corrente(self):
        if self.push_n <= 0:
            return None
        self.push_n -= 1
        if self.push_n <= 0:
            self.dopo_push = 4
        if self.sparso:
            att = np.flatnonzero(self.push)
            out = np.zeros(self.n_pool, np.float32)
            if len(att):
                np.add.at(out, self.codici[att].ravel(), np.repeat(self.push[att], self.codici.shape[1]))
            return np.minimum(out, 1.6 * I_SEME * I_PAROLA_PA)
        return np.minimum(self.push @ self.Gm, 1.6 * I_SEME * I_PAROLA_PA)

    def _assoc_ha_dati(self):
        if getattr(self, "_assoc_ok", False):
            return True
        self._assoc_ok = bool(self.assoc.N.sum() > 0)
        return self._assoc_ok

    def leggi(self, conta, sente, propria, dt=0.1):
        conta = np.asarray(conta, np.float32)
        x = (conta[self.codici].sum(1) if self.sparso else (self.Gm @ conta)) / self.quanti / dt
        if self.media is None:
            self.media, self.var = x.copy(), np.full_like(x, 1.0)
        self.n_fin += 1
        if self.tace > 0 and self.insegnando is None:
            self.tace -= 1
        z = (x - self.media) / np.sqrt(self.var + 0.25)
        self.z_ultimo = z
        self.z_liscio = 0.5 * self.z_liscio + 0.5 * z
        self.dopo_push = max(0, self.dopo_push - 1)
        if not sente and self.push_n <= 0 and self.dopo_push <= 0 and self.attesa <= 0:
            a = 0.05 if self.n_fin < 60 else 0.01
            self.media += a * (x - self.media)
            self.var += a * ((x - self.media) ** 2 - self.var)
        if self.n_fin < 30:
            return None
        self.attesa = max(0, self.attesa - 1)
        if self.seme_pendente is not None and self.attesa <= 0 and not propria:
            self._spingi_seme(self.seme_pendente)
        self.riconosciuta_n = max(0, self.riconosciuta_n - 1)
        if self.proximo_push_n > 0:
            self.proximo_push_n -= 1
            if self.proximo_push_n == 0:
                self._spinta_pensiero()
        if sente or self.buf or self.buf_img:
            self.quiete = 0
        elif self.pensiero is None and PENSA_DA_SOLO and self._assoc_ha_dati() and self.insegnando is None \
                and self.insegnando_img is None:
            self.quiete += 1
            if self.quiete >= QUIETE_FINESTRE:
                p = (self.assoc.nk + 0.1).astype(np.float64)
                p /= p.sum()
                self.avvia_pensiero(int(self.rng.choice(self.K, p=p)), int(self.rng.integers(3, 6)))
        if propria:
            self.attesa = max(self.attesa, 8)
        if self.attesa > 0 or self.insegnando is not None or self.insegnando_img is not None:
            return None
        k = int(self.z_liscio.argmax())
        if CONFERMA:
            if self.riconosciuta_n <= 0 or not self.conferma_k:
                return None
            cand = np.fromiter(self.conferma_k, int)
            k = int(cand[np.argmax(self.z_liscio[cand])])
            if os.environ.get("SHIM_DBG") == "1":
                print("[DBG] k=%d %s z=%.2f media=%.2f n_conf=%d rec=%d attesa=%d" % (k, self.parole[k], self.z_liscio[k], float(np.mean(self.z_liscio)),
                                                                                   len(cand), self.riconosciuta_n, self.attesa), flush=True)
            if self.z_liscio[k] - float(np.mean(self.z_liscio)) > Z_PAROLA:
                self.attesa = PAUSA_VELOCE if (self.pensiero or {}).get("veloce") else PAUSA_PAROLA
                self.tace = TACE_FINESTRE
                self._parola_uscita(k)
                return self.parole[k]
            return None
        altra = float(np.partition(self.z_liscio, -2)[-2])
        if self.z_liscio[k] > Z_PAROLA and self.z_liscio[k] - altra > 0.5:
            self.attesa = PAUSA_VELOCE if (self.pensiero or {}).get("veloce") else PAUSA_PAROLA
            self.tace = TACE_FINESTRE
            self._parola_uscita(k)
            return self.parole[k]
        return None
