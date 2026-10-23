from __future__ import annotations

import base64
import hashlib
import http.server
import io
import collections
import json
import math
import os
import random
import re
import socketserver
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
RADICE = QUI.parent
LIB = RADICE / "shim" / "lib"
NOME = os.environ.get("SHIM_NOME", "Billy")
MODELLO_LLM = os.environ.get("SHIM_LLM", "qwen2.5:7b")


VOCE = os.environ.get("SHIM_VOCE", "piper:it_IT-riccardo-x_low")
PAROLACCE = os.environ.get("SHIM_PAROLACCE", "1") == "1"

MODELLO_ORECCHIE = os.environ.get("SHIM_ORECCHIE", "small")


SOLO_CERVELLO = os.environ.get("SHIM_SOLO_CERVELLO", "1") == "1"

IMBARAZZO_PA, IMBARAZZO_S = float(os.environ.get("SHIM_IMB_PA", "20")), 10.0
LEGGI_FORZA = float(os.environ.get("SHIM_LEGGI_FORZA", "2.0"))
def _parole_struttura():
    from .parole_shim import STRUTTURA
    return STRUTTURA


PAGINA = os.environ.get("SHIM_PAGINA", "")
CONCETTI = os.environ.get("SHIM_CONCETTI", "1") == "1"
RIPASSO = int(os.environ.get("SHIM_RIPASSO", "0"))
Z_IN_TESTA = float(os.environ.get("SHIM_Z_IN_TESTA", "20"))
PENSA_IN_SILENZIO = os.environ.get("SHIM_PENSA_IN_SILENZIO", "0") == "1"
SVUOTA_PA = float(os.environ.get("SHIM_MEM_SVUOTA_PA", "300"))
IMBARAZZO_SCALA = float(os.environ.get("SHIM_IMB_SCALA", "1.0"))
S_IMBARAZZO = float(os.environ.get("SHIM_IMB_SOGLIA", "0.5"))
VOCE_TU = "piper:it_IT-paola-medium"
I_SPINTA_PA = 40.0
UDITO = ["udito_%d" % k for k in range(1, 11)] + ["udito"]
LINGUA_ASCOLTO = ["parole"]
PAROLA = ["bocca"]
VISTA = ["v1_sx", "v1_dx"]
AMIGDALA, ACCUMBENS, TALAMO = ["amigdala"], ["accumbens"], ["talamo"]


I_UDITO, I_ASCOLTO, I_PAROLA, I_VISTA, I_INSULTO, I_AFFETTO = 25.0, 12.0, 30.0, 15.0, 90.0, 70.0

CALO_ORA, CALO_PARLA, ENERGIA_INIZIO = 0.25, 0.01, 0.8

NOIA_S = 45.0

PAROLE_INSULTO = ("stupido", "scemo", "idiota", "cretino", "zitto", "fai schifo", "inutile", "deficiente", "vaffanculo",
                  "stronzo", "brutto")
PAROLE_AFFETTO = ("bravo", "grazie", "ti voglio bene", "simpatico", "bello", "grande", "amico", "complimenti", "fantastico")
PAROLE_RICHIESTA = ("prendi", "prendimi", "mi aiuti", "aiutami", "puoi", "potresti", "fai", "fammi", "porta", "portami",
                    "vai", "dammi", "canta", "balla", "mi passi", "accendi", "spegni", "vieni")


RIFIUTO = re.compile(r"\b(no|non|dopo|lascia|fallo tu|neanche|manco)\b")


def zone_indici(c, nomi_corti):
    idx = {n: i for i, n in enumerate(c.nomi)}
    zone = [idx[z] for nome in nomi_corti for z in ([nome] if nome in idx else ["L_" + nome, "R_" + nome]) if z in idx]
    return np.flatnonzero(np.isin(c.area, zone) & c.eccitatorio), zone


class Cervello(threading.Thread):

    def __init__(self, N):
        super().__init__(daemon=True)
        import torch
        from .kernel_shim import MotoreSHIM
        from .cervello_shim import CervelloUmano, tara
        self.torch = torch
        self.c = CervelloUmano(N)


        from .voce_cervello import Coclea, Bocca, Arcuato
        from . import memoria_lavoro
        self.coclea, self.bocca = Coclea(self.c), Bocca(self.c)
        self.arcuato = Arcuato(self.coclea, self.bocca, Path(os.environ.get("SHIM_ARCUATO") or ((Path(__file__).resolve().parents[1] / "stato") / "shim_arcuato.npz")))
        self.lessico = None
        if SOLO_CERVELLO and os.environ.get("SHIM_PAROLE", "1") == "1":
            try:
                from .parole_shim import Lessico
                self.lessico = Lessico(self.c, self.arcuato.nf, file=Path(os.environ.get("SHIM_LESSICO") or (
                    (Path(__file__).resolve().parents[1] / "stato") / "shim_lessico.npz")))
            except RuntimeError as e:
                print("[SHIM] niente lessico: %s" % e, flush=True)
                self.lessico = None
        self.memoria = None
        if memoria_lavoro.ACCESA and self.lessico is not None:
            l_ = self.lessico
            cod = [l_.neuroni[l_.codici[k]] if l_.sparso else l_.neuroni[l_.Gm[k] > 0] for k in range(l_.K)]
            scelte = None
            if os.environ.get("SHIM_CONOSCENZA") == "2000" and int(os.environ.get("SHIM_CODICI_SEPARATI", "0")) > 0:
                scelte = list(range(min(int(os.environ["SHIM_CODICI_SEPARATI"]), l_.K)))
                from .parole_shim import SEPARATI_NOMI, parole_separate
                if SEPARATI_NOMI:
                    scelte = parole_separate(l_.parole, int(os.environ["SHIM_CODICI_SEPARATI"]))
            elif os.environ.get("SHIM_CONOSCENZA") == "2000":
                import json as _json
                fc = (Path(__file__).resolve().parents[1] / "stato") / "shim_classi_2000.json"
                scelte = list(range(min(762, l_.K)))
                if fc.exists():
                    dc = _json.loads(fc.read_text(encoding="utf-8"))
                    Pc = np.asarray(dc["conteggi"], np.float64)
                    art = Pc[:, 1] / (Pc.sum(1) + 1e-9)
                    ix_ = {w: i for i, w in enumerate(l_.parole)}
                    forti = [ix_[w] for j, (w, c) in enumerate(zip(dc["parole"], dc["classi"])) if c == "nome" and art[j] >= 0.5 and w in ix_]
                    scelte += [k for k in forti if k >= 762][:max(0, memoria_lavoro.N_MEM - len(scelte))]
            if os.environ.get("SHIM_CONOSCENZA") == "wiki":
                from .relazioni_scuola import RELAZIONI
                ix_ = {w: i for i, w in enumerate(l_.parole)}
                extra_ = sorted({ix_[w] for rel, cp in RELAZIONI.items() for w in [rel] + [x for c_ in cp for x in c_] if w in ix_} - set(range(memoria_lavoro.N_MEM)))
                scelte = list(range(min(l_.K, memoria_lavoro.N_MEM))) + extra_
            self.memoria = memoria_lavoro.MemoriaLavoro(self.c, cod, scelte=scelte)
        self.numeri = None
        from . import numeri_shim
        if numeri_shim.ACCESO:
            self.numeri = numeri_shim.Numeri(self.c)
        self.m = MotoreSHIM(self.c)
        cache = (Path(__file__).resolve().parents[1] / "stato") / "tarature"
        impronta = hashlib.sha1((QUI / "cervello_shim.py").read_bytes() + (str(self.c.n) + "shim"
                                                                             + memoria_lavoro.impronta() + numeri_shim.impronta()).encode()).hexdigest()[:12]
        f = cache / ("shim_%s.npy" % impronta)
        if f.exists():
            self.m.dc.copy_(torch.as_tensor(np.load(f), device=self.m.dc.device))
            print("[SHIM] taratura ricaricata", flush=True)
        else:
            f_t = tara(self.m, self.c)
            for _ in range(6):
                e_ = float(np.mean(f_t[0::2]))
                if 2.0 <= e_ <= 4.5:
                    break
                print("[SHIM] taratura non arrivata (E %.2f Hz): altri 15 giri" % e_, flush=True)
                f_t = tara(self.m, self.c, giri=15)
            cache.mkdir(parents=True, exist_ok=True)
            if not 2.0 <= float(np.mean(f_t[0::2])) <= 4.5:
                print("[SHIM] ATTENZIONE: taratura fuori dall'intervallo (E %.2f Hz): le prove su questo cervello non valgono" % float(np.mean(f_t[0::2])), flush=True)
                f = cache / ("shim_%s_SBAGLIATA.npy" % impronta)
            np.save(f, self.m.dc.cpu().numpy())
        self.dc_base = self.m.dc.clone()
        dev = self.m.dc.device
        self.gruppi = {}
        for nome, zone in (("udito", UDITO), ("ascolto", LINGUA_ASCOLTO), ("parola", PAROLA), ("vista", VISTA),
                           ("amigdala", AMIGDALA), ("accumbens", ACCUMBENS), ("talamo", TALAMO)):
            ii, zz = zone_indici(self.c, zone)
            self.gruppi[nome] = (torch.as_tensor(ii, device=dev), zz)
        self.pop = torch.as_tensor(self.c.pop, device=dev)
        self.quanti = torch.bincount(self.pop, minlength=2 * self.c.na).float()
        self.lock = threading.Lock()
        self.stimoli = []
        self.t = 0.0
        self.x_reale = 0.0
        self.zone_hz = np.zeros(self.c.na)
        self.riposo = None
        self.lettura = {}
        self.energia = ENERGIA_INIZIO
        self.umore = 0.0
        self.parla_fino = 0.0
        self.ultimo_ascolto = 0.0
        self.storia_zone = []
        self.guardato = 0.0
        self.spari = (0, b"", b"")
        self.finestra = 0

        from .occhio import Occhio
        self.occhio = Occhio(self.c)
        self.occhio_idx = torch.as_tensor(self.occhio.neuroni, device=dev)
        self.occhio_I = torch.zeros(len(self.occhio.neuroni), device=dev)
        self.occhio_t = 0.0
        self.occhio_guardato = 0.0
        self.visto = None

        import collections
        self.coclea_idx = torch.as_tensor(self.coclea.neuroni, device=dev)
        self.bocca_idx = torch.as_tensor(self.bocca.neuroni, device=dev)
        self.coclea_coda = collections.deque()
        self.alla_bocca = None

        self.bocca_I = np.zeros(len(self.bocca.unita), np.float32)

        self.alla_parola = None
        if self.lessico is not None:
            self.parola_idx = torch.as_tensor(self.lessico.neuroni, device=dev)
        if self.memoria is not None:
            self.mem_idx = torch.as_tensor(self.memoria.neuroni, device=dev)
        if self.numeri is not None:
            self.num_idx = torch.as_tensor(self.numeri.neuroni, device=dev)
            self.numeri.m = self.m
            self.numeri.riposo()
        self.etichetta_ora = None
        self.spinta_fino = 0.0

    def stimola(self, gruppo, pA, secondi):
        with self.lock:
            self.stimoli.append((gruppo, pA, self.t + secondi))

    def gruppo_hz(self, nome):
        ii, zz = self.gruppi[nome]
        return float(np.mean(self.zone_hz[zz])) if len(zz) else 0.0

    def run(self):
        from shim.neuroni import DT
        torch = self.torch
        passi = int(100.0 / DT)
        riposi = []
        prof = os.environ.get("SHIM_PROFILA") == "1"
        acc, n_acc, p_prev = {}, 0, None

        def _sync():
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
        while True:
            t0 = time.time()
            if prof:
                _sync()
                _pa = time.perf_counter()
                if p_prev is not None:
                    acc["6 resto del ciclo (zone, 3D, storia)"] = acc.get("6 resto del ciclo (zone, 3D, storia)", 0.0) + _pa - p_prev
            extra = torch.zeros_like(self.dc_base)
            with self.lock:
                self.stimoli = [s for s in self.stimoli if s[2] > self.t]
                for g, pA, _ in self.stimoli:
                    extra[self.gruppi[g][0]] += pA
                if time.time() - self.occhio_t < 0.5:
                    extra[self.occhio_idx] += self.occhio_I
                sente = bool(self.coclea_coda)
                self.etichetta_ora = None
                if sente:
                    I_c, self.etichetta_ora = self.coclea_coda.popleft()
                    extra[self.coclea_idx] += torch.as_tensor(I_c, device=extra.device)
                if self.bocca_I.any():
                    extra[self.bocca_idx] += torch.as_tensor(self.bocca_I[self.bocca.dove], device=extra.device)
                if self.lessico is not None:
                    Ip = self.lessico.corrente()
                    if Ip is not None:
                        extra[self.parola_idx] += torch.as_tensor(Ip, device=extra.device)
                spinta = self.t < self.spinta_fino
                if spinta:
                    extra[self.bocca_idx] += I_SPINTA_PA
                if self.memoria is not None and self.memoria.svuota_n > 0:
                    extra[self.mem_idx] -= SVUOTA_PA
                if self.memoria is not None and self.memoria.g_nmda > 0:
                    I_l = self.memoria.corrente_lenta(self.memoria.g_nmda, self.memoria.inib, self.memoria.alfa)
                    if I_l is not None:
                        extra[self.mem_idx] += torch.as_tensor(I_l, device=extra.device)
                if self.numeri is not None:
                    I_n = self.numeri.corrente(self.t)
                    if I_n is not None:
                        extra[self.num_idx] += torch.as_tensor(I_n, device=extra.device)
                sp_ = getattr(self.memoria, "spinta", None) if self.memoria is not None else None
                if sp_ is not None and sp_[2] > self.t:
                    extra[self.mem_idx[torch.as_tensor(self.memoria.ass_loc[sp_[0]], device=extra.device)]] += sp_[1]
            if prof:
                _sync()
                _pb = time.perf_counter()
            self.m.dc.copy_(self.dc_base + extra)
            self.m.conta.zero_()
            self.m.passi(passi)
            if prof:
                _sync()
                _pc = time.perf_counter()
            conta = self.m.conta.float()
            per_pop = torch.zeros(2 * self.c.na, device=conta.device).index_add_(0, self.pop, conta) / self.quanti / 0.1
            zone = per_pop[0::2].cpu().numpy()
            if prof:
                _pd = time.perf_counter()
            self.finestra += 1
            if SOLO_CERVELLO:
                frase = self.bocca.finestra(self.m.conta[self.bocca_idx].cpu().numpy(), sente or spinta,
                                            propria=self.etichetta_ora is not None)
                self.bocca_I = self.arcuato.finestra(self.m.conta[self.coclea_idx].cpu().numpy(), sente, self.etichetta_ora)
                parola = None
                if self.lessico is not None:
                    try:
                        propria = self.etichetta_ora is not None
                        self.lessico.ascolta(self.arcuato.a_ora, sente, propria)
                        im_l = None
                        if time.time() - self.occhio_t < 0.5:
                            im_l = self.occhio.leggi(self.m.conta[self.occhio_idx].cpu().numpy(), 0.1)
                        self.lessico.guarda(im_l)
                        self.ultima_conta_parola = self.m.conta[self.parola_idx].cpu().numpy()
                        parola = self.lessico.leggi(self.ultima_conta_parola, sente, propria)
                        if self.lessico.tace > 0 or self.lessico.insegnando is not None or self.lessico.insegnando_img is not None:
                            frase = None
                    except Exception as e_l:
                        self.errori_lessico = getattr(self, "errori_lessico", 0) + 1
                        if self.errori_lessico <= 3:
                            print("[SHIM] errore del lessico: %r" % (e_l,), flush=True)
                if self.numeri is not None:
                    self.numeri._t = self.t
                    self.numeri.leggi(self.m.conta[self.num_idx].cpu().numpy())
                if self.memoria is not None:
                    z_m = self.memoria.leggi(self.m.conta[self.mem_idx].cpu().numpy())
                    if self.lessico is not None:
                        from .memoria_lavoro import Z_MEM
                        tn = np.flatnonzero(z_m > Z_MEM)
                        self.lessico.contesto = [int(self.memoria.parola[k]) for k in tn[np.argsort(-z_m[tn])][:3]]
                if parola and self.alla_parola:
                    threading.Thread(target=self.alla_parola, args=(parola,), daemon=True).start()
                if frase and self.alla_bocca:
                    threading.Thread(target=self.alla_bocca, args=(frase,), daemon=True).start()
            if prof:
                _pe = time.perf_counter()
            if time.time() - self.occhio_guardato < 3.0:
                self.visto = self.occhio.leggi(self.m.conta[self.occhio_idx].cpu().numpy(), 0.1)
            if time.time() - self.guardato < 3.0:
                bit = np.packbits((self.m.conta > 0).cpu().numpy(), bitorder="little").tobytes()
                tutte = (per_pop[0::2] * 0.8 + per_pop[1::2] * 0.2).cpu().numpy().astype(np.float32).tobytes()
                self.spari = (self.finestra, tutte, bit)
            if prof:
                p_prev = time.perf_counter()
                for k_, v_ in (("1 preparare le correnti (Python, trasferimenti)", _pb - _pa), ("2 MOTORE: 1000 passi da 0,1 ms sulla scheda", _pc - _pb),
                               ("3 popolazioni e zone (scheda -> CPU)", _pd - _pc), ("4 bocca, arcuato, lessico, vista (CPU e Python)", _pe - _pd),
                               ("5 cervello 3D e occhio", p_prev - _pe)):
                    acc[k_] = acc.get(k_, 0.0) + v_
                n_acc += 1
                if n_acc >= 40:
                    tot = sum(acc.values())
                    print("[PROFILO] %d finestre, %.1f ms l'una (%.2fx tempo reale):" % (n_acc, 1000 * tot / n_acc, 0.1 * n_acc / tot), flush=True)
                    for k_ in sorted(acc):
                        print("   %-52s %6.1f ms  %4.1f%%" % (k_, 1000 * acc[k_] / n_acc, 100 * acc[k_] / tot), flush=True)
                    acc, n_acc = {}, 0
            self.t += 0.1
            self.x_reale = 0.9 * self.x_reale + 0.1 * (0.1 / max(time.time() - t0, 1e-6))
            self.zone_hz = 0.7 * self.zone_hz + 0.3 * zone
            self.storia_zone.append(zone.mean())
            self.storia_zone = self.storia_zone[-600:]
            if self.riposo is None:
                riposi.append({g: self.gruppo_hz(g) for g in self.gruppi})
                riposi[-1]["_tutto"] = float(self.zone_hz.mean())
                if len(riposi) >= 30:
                    self._riposo_tutto = max(np.mean([r["_tutto"] for r in riposi]), 0.3)
                    self.riposo = {g: max(np.mean([r[g] for r in riposi]), 0.3) for g in self.gruppi}
                continue

            self.energia = max(0.0, self.energia - 0.1 / 3600.0 * CALO_ORA - (CALO_PARLA * 0.1 if self.t < self.parla_fino else 0.0))

            glob = float(self.zone_hz.mean()) / self._riposo_tutto - 1.0
            rel = {g: self.gruppo_hz(g) / self.riposo[g] - 1.0 for g in self.gruppi}
            self.globale = glob
            self.umore = 0.98 * self.umore + 0.02 * float(np.clip(rel["accumbens"] - rel["amigdala"], -2, 2))
            self.lettura = {k: round(v, 3) for k, v in rel.items()}

    def stato(self):
        rel = self.lettura or {g: 0.0 for g in self.gruppi}
        glob = getattr(self, "globale", 0.0)
        amig, piacere, ecc = (rel.get("amigdala", 0.0) - glob, rel.get("accumbens", 0.0) - glob, rel.get("talamo", 0.0) - glob)
        noia = min(1.0, (self.t - self.ultimo_ascolto) / (2 * NOIA_S))
        if amig > 0.5 and ecc > 0.1:
            emozione = "arrabbiato"
        elif amig > 0.5:
            emozione = "spaventato"
        elif piacere > 0.4:
            emozione = "contento"
        elif self.energia < 0.3:
            emozione = "stanco"
        elif self.umore < -0.15:
            emozione = "triste"
        elif noia > 0.8:
            emozione = "annoiato"
        else:
            emozione = "tranquillo"
        return {"t": round(self.t, 1), "x_reale": round(self.x_reale, 3), "energia": round(self.energia, 3),
                "umore": round(self.umore, 3), "emozione": emozione, "noia": round(noia, 2), "zone": rel,
                "corteccia_hz": round(float(self.zone_hz.mean()), 2), "neuroni": int(self.c.n),
                "sinapsi": int(self.c.n_sinapsi), "zone_n": int(self.c.na), "parla": self.t < self.parla_fino,
                "ingressi": sorted({g for g, _, fino in list(self.stimoli) if fino > self.t}),
                **({"memoria": self.memoria.tenute(self.lessico.parole)} if self.memoria is not None else {})}

    def voglia(self):
        rel = self.lettura or {}
        z = (3.0 * (self.energia - 0.45) + 1.5 * self.umore + 1.0 * rel.get("accumbens", 0.0) - 1.5 * max(0.0, rel.get("amigdala", 0.0)))
        idx = {n: i for i, n in enumerate(self.c.nomi)}
        a, b = self.zone_hz[idx["pf_sx"]], self.zone_hz[idx["pf_dx"]]
        rumore = 2.0 * (a - b) / max(a + b, 0.1)
        p = 1.0 / (1.0 + math.exp(-(z + rumore)))
        return p, rumore


_WHISPER = None


RIPETI_PENALITA = float(os.environ.get("SHIM_RIPETI", "1.05"))
ORECCHIE_NPU = "http://127.0.0.1:8801/"


def orecchie_npu_accese():
    try:
        urllib.request.urlopen(urllib.request.Request(ORECCHIE_NPU, data=np.zeros(1600, np.float32).tobytes()), timeout=3).read()
        return True
    except Exception:
        return False


ALLUCINAZIONI = re.compile(r"grazie per (la visione|l'attenzione|aver guardato|la sua cura)|grazie a tutti e a tutte|"
                           r"sottotitol|iscrivet|iscriviti|amara\.org|qtss|al prossimo video", re.I)


def voce_vera_s(pcm):
    n = len(pcm) // 320
    if n == 0:
        return 0.0
    rms = np.sqrt((pcm[:n * 320].reshape(n, 320) ** 2).mean(1))
    soglia = max(0.01, 3 * float(np.percentile(rms, 20)))
    return float((rms > soglia).sum()) * 0.02


def ascolta_pcm(pcm):
    global _WHISPER
    dur = len(pcm) / 16000.0
    if voce_vera_s(pcm) < 0.25:
        print("[SHIM] orecchie: niente voce (%.1f s di audio)" % dur, flush=True)
        return "", dur
    testo = None
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(ORECCHIE_NPU, data=pcm.astype(np.float32).tobytes()),
                                              timeout=15).read())
        testo = r["testo"].strip()
    except Exception as e:
        print("[SHIM] orecchie NPU spente (%s): Whisper %s sul processore" % (type(e).__name__, MODELLO_ORECCHIE), flush=True)
    if testo is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            WhisperModel = None
        if _WHISPER is None and WhisperModel is not None:
            _WHISPER = WhisperModel(MODELLO_ORECCHIE, device="cpu", compute_type="int8")
        seg, _ = _WHISPER.transcribe(pcm.astype(np.float32), language="it", vad_filter=True, beam_size=1)
        testo = " ".join(x.text.strip() for x in seg).strip()
    if ALLUCINAZIONI.search(testo):
        print("[SHIM] orecchie: scartato '%s' (Whisper lo inventa sul silenzio)" % testo, flush=True)
        return "", dur
    return testo, dur


def ascolta(dati_audio):
    from faster_whisper import decode_audio
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(dati_audio)
        nome = f.name
    try:
        return ascolta_pcm(decode_audio(nome, sampling_rate=16000).astype(np.float32))
    finally:
        os.unlink(nome)


def _chatml(messaggi, apri=True):
    return "".join("<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"]) for m in messaggi) + (
        "<|im_start|>assistant\n" if apri else "")


def prepara(storia):
    corpo = _corpo([], False, 1)
    corpo["prompt"] = _chatml([{"role": "system", "content": SISTEMA}] + _finestra(storia), apri=False)
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


def _corpo(messaggi, stream, num_predict):
    return {"model": MODELLO_LLM, "stream": stream, "raw": True, "prompt": _chatml(messaggi), "keep_alive": "60m",
            "options": {"temperature": 0.9, "num_predict": num_predict, "seed": random.randrange(1 << 30),


                        "repeat_penalty": RIPETI_PENALITA, "repeat_last_n": 256,
                        "stop": ["<|im_end|>", "<|im_start|>"]}}


def scrivi_risposta(storia, frase, st, decisione):
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", headers={"Content-Type": "application/json"},
                                 data=json.dumps(_corpo(_messaggi(storia, frase, st, decisione), False, 32)).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        testo = json.loads(r.read())["response"]
    fuori = FUORI.search(testo)
    return pulisci(testo[:fuori.start()] if fuori else testo)


FINE_FRASE = re.compile(r"[.!?…]+[\"')]*\s")
VIRGOLA = re.compile(r"[,;:]\s")


PAROLA_INTERA = re.compile(r"\S+\s")


def _fine_pezzo(buf, detto):
    for m in FINE_FRASE.finditer(buf):
        if m.end() >= (3 if not detto else 8):
            return m.end()
    if not detto:
        m = VIRGOLA.search(buf, 3)
        if m:
            return m.end()
        parole = list(PAROLA_INTERA.finditer(buf))
        if len(parole) >= 5:
            return parole[4].end()
    return None


FUORI = re.compile(r"[\[\(<{]|[^\x00-ɏ‘-…\s]")


def scrivi_a_pezzi(storia, frase, st, decisione, tempi=None):
    corpo = _corpo(_messaggi(storia, frase, st, decisione), True, 40)
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    buf, detto, grezzo = "", False, ""
    with urllib.request.urlopen(req, timeout=120) as r:
        for riga in r:
            if not riga.strip():
                continue
            d = json.loads(riga)
            if tempi is not None and "primo_token" not in tempi:
                tempi["primo_token"] = time.time()
            buf += d.get("response", "")
            grezzo += d.get("response", "")
            fuori = FUORI.search(buf)
            if fuori:
                grezzo = grezzo[:len(grezzo) - (len(buf) - fuori.start())]
                buf = buf[:fuori.start()]
                if tempi is not None:
                    tempi["tagliato"] = True
                break
            fine = _fine_pezzo(buf, detto)
            while fine:
                detto = True
                yield pulisci(buf[:fine])
                buf = buf[fine:]
                fine = _fine_pezzo(buf, True)
            if tempi is not None:
                tempi["grezzo"] = grezzo
            if d.get("done"):
                if tempi is not None:
                    tempi["letti"] = (d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 0) / 1e9)
                break
    if len(buf.strip()) >= 3:
        yield pulisci(buf)


def pulisci(testo):
    testo = testo.strip()
    testo = re.sub(r"^\W*%s\s*:\s*" % NOME, "", testo).strip().strip('"')
    for en, it in (("thanks", "grazie"), ("Thanks", "Grazie"), ("sorry", "scusa"), ("Sorry", "Scusa")):
        testo = re.sub(r"\b%s\b" % en, it, testo)
    return testo


SISTEMA = (
    "Sei %s, un ragazzo che parla SOLO in italiano colloquiale, a voce: al massimo due frasi brevi e naturali. NON sei "
    "un assistente: hai i tuoi umori. Non parlare mai di numeri, di cervelli o di modelli. Niente argomenti sessuali. "
    "Dopo ogni frase che ti dicono, una nota (Ora: ...) dice come ti senti adesso e che cosa hai deciso: rispondi "
    "COSI', ma non ripetere mai la nota. Rispondi solo con quello che dici ad alta voce, senza virgolette, senza parentesi e "
    "senza descrivere gesti."
) % NOME


def _utente(frase, st, decisione):
    emo = st["emozione"]

    tono = {"arrabbiato": "arrabbiato, brusco" + (", parolacce ok" if PAROLACCE else ""),
            "spaventato": "spaventato, agitato", "contento": "contento, scherzi, caloroso",
            "stanco": "stanco, svogliato, sbuffi", "triste": "giu' di morale, parli poco",
            "annoiato": "annoiato, distratto"}.get(emo, "tranquillo, gentile")
    if emo != "arrabbiato":
        tono += ", niente parolacce"
    energia = "alta" if st["energia"] > 0.6 else "media" if st["energia"] > 0.3 else "bassa"
    nota = "Ora: %s; energia %s.%s" % (tono, energia, (" " + decisione) if decisione else "")
    return [{"role": "user", "content": frase}, {"role": "system", "content": nota}]


def _finestra(storia):
    inizio = max(0, (len(storia) - 12) // 78 * 78)
    return storia[inizio:]


def _messaggi(storia, frase, st, decisione):
    return [{"role": "system", "content": SISTEMA}] + _finestra(storia) + _utente(frase, st, decisione)


_PIPER = {}


VERSI = {"arrabbiato": ["Uff!", "Bah.", "Tsk."], "spaventato": ["Ehm...", "Oh!"], "contento": ["Ah!", "Oh!", "Eh eh."],
         "stanco": ["Mmh...", "Eh...", "Uff..."], "triste": ["Mah...", "Eh..."], "annoiato": ["Boh.", "Mah."],
         "tranquillo": ["Mh.", "Allora.", "Eh.", "Dunque."]}
_VERSI_AUDIO = {}
VERSO_SUBITO = os.environ.get("SHIM_VERSO", "1") != "0"


def voce_verso(testo):
    if testo not in _VERSI_AUDIO:
        _VERSI_AUDIO[testo] = voce(testo)
    return _VERSI_AUDIO[testo]


def in_16k(wav):
    import wave
    with wave.open(io.BytesIO(wav)) as w:
        fr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    t = np.arange(0, len(x) / fr, 1 / 16000.0)
    return np.interp(t, np.arange(len(x)) / fr, x).astype(np.float32)


def taglia_silenzio(pcm, margine_s=0.02):
    if len(pcm) == 0:
        return pcm
    forte = np.flatnonzero(np.abs(pcm) > 0.03 * (np.abs(pcm).max() + 1e-9))
    if len(forte) == 0:
        return pcm[:0]
    m = int(margine_s * 16000)
    return pcm[max(0, forte[0] - m):forte[-1] + m]


def voce(testo, chi=None, var=None):
    VOCE_ = chi or VOCE
    if VOCE_.startswith("piper:"):
        try:
            import wave
            from piper import PiperVoice
            nome = VOCE_.split(":", 1)[1]
            if nome not in _PIPER:
                _PIPER[nome] = PiperVoice.load(str(QUI / "voci" / (nome + ".onnx")))
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                if var is not None:
                    from piper import SynthesisConfig
                    rg = np.random.default_rng(int(var))
                    cfg = SynthesisConfig(length_scale=float(rg.uniform(0.85, 1.25)), noise_scale=float(rg.uniform(0.3, 1.0)),
                                          noise_w_scale=float(rg.uniform(0.3, 1.0)))
                    _PIPER[nome].synthesize_wav(testo, w, syn_config=cfg)
                else:
                    _PIPER[nome].synthesize_wav(testo, w)
            return buf.getvalue(), "audio/wav"
        except Exception as e:
            print("[SHIM] Piper non va (%s): voce di Windows" % e, flush=True)
    try:
        if VOCE.startswith("piper:"):
            raise RuntimeError("piper")
        import asyncio
        import edge_tts

        async def _f():
            buf = io.BytesIO()
            async for pezzo in edge_tts.Communicate(testo, VOCE).stream():
                if pezzo["type"] == "audio":
                    buf.write(pezzo["data"])
            return buf.getvalue()
        return asyncio.run(_f()), "audio/mpeg"
    except Exception as e:
        print("[SHIM] edge-tts non va (%s): voce di Windows" % e, flush=True)
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            nome = f.name
        ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$s.SelectVoice('Microsoft Elsa Desktop'); $s.SetOutputToWaveFile('%s'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()" % nome)
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=testo, text=True, encoding="utf-8")
        dati = Path(nome).read_bytes()
        os.unlink(nome)
        return dati, "audio/wav"


class SHIM:

    def __init__(self, N):
        self.cervello = Cervello(N)
        self.storia = []
        self.lock = threading.Lock()
        self.diario = []

        self.spontanei = []
        self.ascolta_fino = (0, 0.0)
        self.cervello.alla_bocca = self.dalla_bocca
        self.cervello.alla_parola = self.dalla_parola
        self._lock_insegna = threading.Lock()
        self.segreti = set()


    def sente_suono(self, pcm, sillabe=None, propria_vuota=False):
        b = self.cervello
        fin = b.coclea.correnti(pcm)
        et = [None] * len(fin)
        if propria_vuota:
            et = [[] for _ in fin]
        if sillabe:
            u = b.bocca.unita
            for i in range(len(fin)):
                if fin[i].mean() < 3.0:
                    et[i] = []
                    continue
                c, v = sillabe[min(len(sillabe) - 1, i * len(sillabe) // max(len(fin), 1))]
                et[i] = [u.index(v)] + ([u.index(c)] if c else [])
        with b.lock:
            b.coclea_coda.extend(zip(fin, et))

    def di_frase(self, testo, parole=None, n=None):
        b = self.cervello
        self._parlando = True
        try:
            audio, tipo = voce(testo)
            pcm = taglia_silenzio(in_16k(audio)) if tipo == "audio/wav" else np.zeros(1600, np.float32)
            dur = len(pcm) / 16000.0
            b.parla_fino = max(b.parla_fino, b.t) + dur
            self.sente_suono(np.concatenate([pcm, np.zeros(800, np.float32)]), propria_vuota=True)
            pezzo = {"testo": testo, "frase": testo, "audio": base64.b64encode(audio).decode(), "tipo_audio": tipo, "verso": False,
                     "parole": list(parole or []), "vocali": []}
            self.turni = getattr(self, "turni", {})
            if n is None:
                n_, fino = self.ascolta_fino
                n = n_ if (time.time() <= fino and n_ in self.turni) else None
            if n is not None and n in self.turni:
                self.turni[n]["pezzi"].append(pezzo)
                self.turni[n]["detto"] = True
            else:
                k = self._prossimo = getattr(self, "_prossimo", 0) + 1
                self.turni[k] = {"pezzi": [pezzo], "fine": True, "t0": time.time(),
                                 "meta": {"gesto": "Idle", "stato": b.stato(), "accetta": None, "pronto_s": 0.0}}
                self.spontanei.append(k)
            self.annota("dice: %s" % testo)
        finally:
            self._parlando = False

    def dalla_parola(self, testo):
        b = self.cervello
        l_ = b.lessico
        cat_ = getattr(l_, "catena", None) if l_ is not None else None
        if cat_ and cat_[-1] == testo and getattr(l_, "catena_veloce", False):
            if l_.pensiero is not None:
                self.annota("pensa: %s" % " ".join(cat_))
                return
            l_.catena_veloce = False
            from .frasi_shim import componi, componi_legate
            testo_f = componi_legate(cat_) if getattr(l_, "catena_forma", "") == "legate" else componi(cat_)
            return self.di_frase(testo_f, list(cat_), getattr(self, "turno_frase", None))
        if PENSA_IN_SILENZIO:
            n_, fino_ = self.ascolta_fino
            if not (time.time() <= fino_ and n_ in getattr(self, "turni", {})):
                self._pensiero_muto(testo)
                return
        self._parlando = True
        try:
            audio, tipo = voce(testo)
            pcm = taglia_silenzio(in_16k(audio)) if tipo == "audio/wav" else np.zeros(1600, np.float32)
            dur = len(pcm) / 16000.0
            b.parla_fino = max(b.parla_fino, b.t) + dur
            self.sente_suono(np.concatenate([pcm, np.zeros(800, np.float32)]), propria_vuota=True)
            pezzo = {"testo": testo, "audio": base64.b64encode(audio).decode(), "tipo_audio": tipo, "verso": False,
                     "parole": [testo], "vocali": []}
            cat = getattr(b.lessico, "catena", None) if b.lessico is not None else None
            if cat and cat[-1] == testo:
                from .frasi_shim import componi
                pezzo["frase"] = componi(cat)
                if len(cat) > 1:
                    self.annota("frase: %s" % pezzo["frase"])
            n, fino = self.ascolta_fino
            if time.time() <= fino and n in getattr(self, "turni", {}):
                self.turni[n]["pezzi"].append(pezzo)
                self.ascolta_fino = (n, min(max(fino, time.time() + dur + 1.0), self.turni[n]["t0"] + 15.0 / min(1.0, max(0.3, b.x_reale or 1.0))))
                self.annota("dice la parola (dal cervello): %s" % testo)
            else:
                k = self._prossimo = getattr(self, "_prossimo", 0) + 1
                self.turni = getattr(self, "turni", {})
                st = b.stato()
                self.turni[k] = {"pezzi": [pezzo], "fine": True, "t0": time.time(),
                                 "meta": {"gesto": "Idle", "stato": st, "accetta": None, "pronto_s": 0.0}}
                self.spontanei.append(k)
                self.annota("dice da solo una parola (dal cervello): %s" % testo)
        finally:
            self._parlando = False

    def pensa(self, parola, lunghezza=4, veloce=False):
        l = self.cervello.lessico
        k = l.parole.index(parola)
        n = self._prossimo = getattr(self, "_prossimo", 0) + 1
        self.turni = getattr(self, "turni", {})
        st = self.cervello.stato()
        self.turni[n] = {"pezzi": [], "fine": False, "t0": time.time(), "capito": parola,
                         "meta": {"gesto": "Idle", "stato": st, "accetta": None, "pronto_s": 0.0}}
        self.ascolta_fino = (n, time.time() + (6.0 * lunghezza + 8.0) / min(1.0, max(0.3, self.cervello.x_reale or 1.0)))
        l.avvia_pensiero(k, lunghezza, veloce=veloce)
        self.turno_frase = n if veloce else None

        def chiudi():
            while time.time() < self.ascolta_fino[1] or getattr(self, "_parlando", False):
                time.sleep(0.1)
            self.turni[n]["fine"] = True
        threading.Thread(target=chiudi, daemon=True).start()
        return n


    FORMULE = [
        ("saluto", ("ciao", "buongiorno", "buonasera", "salve", "ehi", "hey")),
        ("come_stai", ("come stai", "come va", "tutto bene", "come ti senti", "stai bene")),
        ("nome", ("come ti chiami", "chi sei", "il tuo nome", "tuo nome")),
        ("grazie", ("grazie",)),
        ("sai_fare", ("cosa sai fare", "cosa sai", "che sai fare", "cosa fai")),
        ("addio", ("arrivederci", "a dopo", "buonanotte", "a presto", "addio")),
        ("bravo", ("bravo", "bravissimo", "giusto", "esatto", "bene cosi")),
        ("sbagliato", ("sbagliato", "non e' vero", "non è vero", "falso")),
        ("sentimento", ("sono stanco", "sono stanca", "sono triste", "sono contento", "sono contenta", "sono felice", "sono arrabbiato",
                        "sono arrabbiata", "sono malato", "sono malata", "sono annoiato", "sono annoiata", "mi annoio", "ho paura",
                        "sto male", "sto bene")),
        ("come_mi_chiamo", ("come mi chiamo", "chi sono io", "ti ricordi il mio nome", "sai il mio nome", "il mio nome lo sai")),
        ("mi_chiamo", ("mi chiamo", "il mio nome e'", "il mio nome è")),
    ]
    AMICI_FILE = (Path(__file__).resolve().parents[1] / "stato") / "shim_amici.json"
    STATO_FRASE = {"contento": "Sto bene, sono contento.", "tranquillo": "Sto bene, sono tranquillo.", "stanco": "Sono un po' stanco.",
                   "annoiato": "Mi annoio un po'.", "triste": "Non tanto bene, sono un po' triste.", "spaventato": "Ho un po' paura.",
                   "arrabbiato": "Sono arrabbiato."}

    def _formule(self, testo):
        t = " " + " ".join(re.findall(r"[a-zàèéìòóù']+", testo.lower())) + " "
        trovate = []
        for nome, pezzi in self.FORMULE:
            if any((" " + p + " ") in t for p in pezzi):
                trovate.append(nome)
        if "come_mi_chiamo" in trovate and "mi_chiamo" in trovate:
            trovate.remove("mi_chiamo")
        if "mi_chiamo" in trovate and not re.search(r"(?:mi chiamo|il mio nome [eè]'?)\s+([a-zàèéìòóù]+)", testo.lower()):
            trovate.remove("mi_chiamo")
        return trovate

    def _risposta_formule(self, trovate):
        b = self.cervello
        out = []
        for f in trovate:
            if f == "saluto":
                out.append("Ciao!")
            elif f == "come_stai":
                out.append(self.STATO_FRASE.get(b.stato()["emozione"], "Sto bene."))
            elif f == "nome":
                out.append("Mi chiamo SHIM.")
            elif f == "grazie":
                out.append("Prego!")
            elif f == "sai_fare":
                out.append("Conosco circa duemilacinquecento parole. Puoi insegnarmi le cose come a scuola, per esempio: il gatto è un mammifero. Poi chiedimi cos'è, cosa ha, dove vive." if os.environ.get("SHIM_CONOSCENZA") == "2000"
                           else "So dire frasi sulle cose che conosco: gli animali, il cibo, la casa, la natura.")
            elif f == "addio":
                out.append("Ciao, a presto!")
            elif f == "bravo":
                b.stimola("accumbens", 15.0, 4.0)
                out.append("Grazie!")
            elif f == "sbagliato":
                self.feedback(False)
                out.append("Ops, scusa.")
            elif f == "mi_chiamo":
                m = re.search(r"(?:mi chiamo|il mio nome [eè]'?)\s+([a-zàèéìòóù]+)", getattr(self, "_testo_formule", "").lower())
                nome_ = m.group(1).capitalize()
                try:
                    self.AMICI_FILE.write_text(json.dumps({"nome": nome_}, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
                self.nome_amico = nome_
                b.stimola("accumbens", 10.0, 3.0)
                out.append("Piacere, %s!" % nome_)
            elif f == "sentimento":
                t_ = getattr(self, "_testo_formule", "").lower()
                if re.search(r"stanc|malat|male|trist|paura|arrabbiat|annoi", t_):
                    b.stimola("amigdala", 6.0, 3.0)
                    out.append("Mi dispiace. Riposati un po'." if re.search(r"stanc|malat|male", t_) else
                               "Mi dispiace. Vuoi parlare un po' con me?" if re.search(r"trist|paura|annoi", t_) else "Mi dispiace. Calmati un po'.")
                else:
                    b.stimola("accumbens", 10.0, 3.0)
                    out.append("Che bello! Sono contento anch'io.")
            elif f == "come_mi_chiamo":
                nome_ = getattr(self, "nome_amico", None)
                if nome_ is None and self.AMICI_FILE.exists():
                    nome_ = json.loads(self.AMICI_FILE.read_text(encoding="utf-8")).get("nome")
                out.append(("Ti chiami %s." % nome_) if nome_ else "Non me l'hai ancora detto.")
        return " ".join(out)

    def dialogo(self, testo):
        b, l = self.cervello, self.cervello.lessico
        ix = {w: i for i, w in enumerate(l.parole)}
        trovate = self._formule(testo)
        self._testo_formule = testo
        with b.lock:
            if any(e is not None for _, e in b.coclea_coda):
                b.coclea_coda = collections.deque((f, e) for f, e in b.coclea_coda if e is None)
                b.parla_fino = min(b.parla_fino, b.t)
                if l is not None:
                    l.attesa = 0
        formula_parole = {"ciao", "grazie", "bravo", "giusto", "esatto", "sbagliato", "falso", "bene", "salve", "nome"}
        for nome_f, pezzi_f in self.FORMULE:
            if nome_f in trovate:
                formula_parole.update(w for p_ in pezzi_f for w in p_.split())
        lette = [w for w in re.findall(r"[a-zàèéìòóù]+", testo.lower()) if w in ix and len(w) >= 3 and w not in formula_parole]
        lette = list(dict.fromkeys(lette))[:4]
        n = self._prossimo = getattr(self, "_prossimo", 0) + 1
        self.turni = getattr(self, "turni", {})
        self.turni[n] = {"pezzi": [], "fine": False, "t0": time.time(), "capito": testo,
                         "meta": {"gesto": "Idle", "stato": b.stato(), "accetta": None, "pronto_s": 0.0}}
        info = {"lette": lette, "formule": trovate, "in_testa": [], "modo": None}
        self.turni[n]["info"] = info
        threading.Thread(target=self._dialogo_filo, args=(n, trovate, lette, info), daemon=True).start()
        return n, info

    def _somma(self, testo):
        from .numeri_shim import PAROLE_NUMERO
        t = testo.lower().replace("+", " più ").replace(" - ", " meno ")
        toks = re.findall(r"[a-zàèéìòóù]+|\d+", t)
        nums = []
        for w in toks:
            if w.isdigit() and 1 <= int(w) <= 10:
                nums.append(int(w))
            elif w in PAROLE_NUMERO:
                nums.append(PAROLE_NUMERO.index(w) + 1)
        if ("più" in toks or "piu" in toks) and len(nums) >= 2 and nums[0] + nums[1] <= 10:
            return nums[0], nums[1], 1
        if "meno" in toks and len(nums) >= 2 and nums[0] - nums[1] >= 1:
            return nums[0], nums[1], -1
        return None

    def _conta_avanti(self, a, b, n, verso=1):
        from .numeri_shim import PAROLE_NUMERO
        b_ = self.cervello
        nm = b_.numeri
        with b_.lock:
            nm.svuota()
            nm.spinta = (float(a), 300.0, b_.t + 0.3)
        t0 = b_.t
        while b_.t - t0 < 0.8:
            time.sleep(0.05)
        with b_.lock:
            nm.avvia_conteggio(b_.t, b, verso)
        t0 = b_.t
        while b_.t - t0 < b * 1.0 + 2.0 and not (nm.conteggio or {}).get("finito"):
            time.sleep(0.05)
        contati = list((nm.conteggio or {}).get("lista") or [])
        ris = contati[-1] if contati else None
        with b_.lock:
            nm.svuota()
        frase = PAROLE_NUMERO[a - 1].capitalize() + "... " + ", ".join(PAROLE_NUMERO[x - 1] if x else "ehm" for x in contati)
        frase += (". Fa %s." % PAROLE_NUMERO[ris - 1]) if ris else ". Ehm, mi sono perso."
        self.di_frase(frase, [], n)
        return ris

    def _confronto(self, testo):
        from .numeri_shim import PAROLE_NUMERO
        t = testo.lower()
        nums = []
        for w in re.findall(r"[a-zàèéìòóù]+|\d+", t):
            if w.isdigit() and 1 <= int(w) <= 10:
                nums.append(int(w))
            elif w in PAROLE_NUMERO:
                nums.append(PAROLE_NUMERO.index(w) + 1)
        if len(nums) >= 2 and nums[0] != nums[1]:
            if any(x in t for x in ("più grande", "piu grande", "maggiore", "più alto", "di più", "il più")):
                return nums[0], nums[1], "grande"
            if any(x in t for x in ("più piccolo", "piu piccolo", "minore", "di meno", "più basso")):
                return nums[0], nums[1], "piccolo"
        return None

    def _confronta(self, a, b, verso, n):
        from .numeri_shim import PAROLE_NUMERO
        b_ = self.cervello
        nm = b_.numeri
        letti = []
        for x in (a, b):
            with b_.lock:
                nm.svuota()
            t0 = b_.t
            while b_.t - t0 < 0.6:
                time.sleep(0.03)
            with b_.lock:
                base = nm.z.copy()
                nm.spinta = (float(x), 150.0, b_.t + 0.3, 0.25)
            t0 = b_.t
            vals = []
            while b_.t - t0 < 0.3:
                g = nm.grandezza(base)
                if g is not None:
                    vals.append(g)
                time.sleep(0.03)
            letti.append(float(np.mean(vals)) if vals else None)
        with b_.lock:
            nm.svuota()
        if None in letti:
            self.di_frase("Ehm, non lo so.", [], n)
            return None
        scelto = a if ((letti[0] > letti[1]) == (verso == "grande")) else b
        self.di_frase("Il %s." % PAROLE_NUMERO[scelto - 1], [], n)
        return scelto, letti

    QUANTI_FILE = (Path(__file__).resolve().parents[1] / "stato") / "shim_quanti.json"

    def _quanti_carica(self):
        if getattr(self, "quanti_esempi", None) is None:
            try:
                self.quanti_esempi = [tuple(e) for e in json.loads(self.QUANTI_FILE.read_text(encoding="utf-8"))]
            except Exception:
                self.quanti_esempi = []
        return list(self.quanti_esempi)

    def _quanti_salva(self):
        try:
            self.QUANTI_FILE.write_text(json.dumps(self.quanti_esempi[-500:]), encoding="utf-8")
        except Exception as e:
            print("[SHIM] non salvo gli esempi di quanti: %r" % (e,), flush=True)

    def quanti_sono(self, n, contrasto, contesto="puntini"):
        from .numeri_shim import PAROLE_NUMERO, SIGMA_LOG
        b_ = self.cervello
        nm = b_.numeri
        try:
            es = self._quanti_carica()
            if len(es) < 6:
                self.di_frase("Non ho ancora imparato a contare le cose.", [], n)
                return
            X = np.array([[1.0, e[1]] for e in es])
            y = np.array([e[0] for e in es])
            a, bb = np.linalg.lstsq(X, y, rcond=None)[0]


            ctx = [e for e in es if (e[2] if len(e) > 2 else "puntini") == contesto]
            if contesto != "puntini" and len({e[1] for e in ctx}) >= 2:
                a, bb = np.linalg.lstsq(np.array([[1.0, e[1]] for e in ctx]), np.array([e[0] for e in ctx]), rcond=None)[0]
            elif contesto != "puntini" and ctx:
                a = float(np.mean([e[0] - bb * e[1] for e in ctx]))
            stima = float(np.clip((contrasto - a) / max(bb, 1e-6), 0.6, 10.0))
            with b_.lock:
                nm.svuota()
            t0 = b_.t
            while b_.t - t0 < 0.6:
                time.sleep(0.03)
            with b_.lock:
                base = nm.z.copy()
                nm.spinta = (stima, 150.0, b_.t + 0.3, SIGMA_LOG)
            t0 = b_.t
            vals = []
            while b_.t - t0 < 0.3:
                g = nm.grandezza(base)
                if g is not None:
                    vals.append(g)
                time.sleep(0.03)
            with b_.lock:
                nm.svuota()
            self.turni[n]["info"] = {"stima_somma": round(stima, 2), "letto_neuroni": round(float(np.mean(vals)), 2) if vals else None}
            if not vals:
                self.di_frase("Ehm, non lo so.", [], n)
                return
            k = int(np.clip(round(float(np.mean(vals))), 1, 10))
            self.di_frase("Sono %s." % PAROLE_NUMERO[k - 1] if k > 1 else "È uno.", [PAROLE_NUMERO[k - 1]], n)
        finally:
            self.turni[n]["fine"] = True

    def _conta_parole(self, a, b, n, verso=1):
        from .numeri_shim import PAROLE_NUMERO
        b_, l = self.cervello, self.cervello.lessico
        ix = {w: i for i, w in enumerate(l.parole)}
        if PAROLE_NUMERO[a - 1] not in ix:
            self.di_frase("Non conosco i numeri.", [], n)
            return None
        with b_.lock:
            l.catena, l.catena_veloce = [], False
            l.avvia_conteggio(ix[PAROLE_NUMERO[a - 1]], b, verso)
        t0 = b_.t
        while b_.t - t0 < 4.5 + 1.2 * b and l.pensiero is not None:
            time.sleep(0.05)
        with b_.lock:
            cat = list(getattr(l, "catena", []) or [])
            l.pensiero = None
        passi = [w for w in cat[1:]]
        ris = passi[-1] if len(passi) == b else None
        frase = PAROLE_NUMERO[a - 1].capitalize() + "... " + ", ".join(passi)
        frase += (". Fa %s." % ris) if ris else ". Ehm, mi sono perso."
        self.di_frase(frase, cat, n)
        return ris


    def _concetti(self):
        if getattr(self, "concetti", None) is None:
            from .concetti_shim import Concetti
            from .frasi_shim import QUI as _Q
            self.concetti = Concetti(Path(_Q) / ("shim_concetti_2000.json" if os.environ.get("SHIM_CONOSCENZA") == "2000" else "shim_concetti.json"))
            from .scuola_concetti import frasi
            if self.concetti.n() > 0 and not self.concetti.studiate:
                self.concetti.studiate = [f for f in frasi() if not f.startswith(("Il cane abbaia", "Il gatto miagola"))][:325]
            nuove_f = [f for f in frasi() if f not in set(self.concetti.studiate)]
            if nuove_f:
                capite_ = [f for f in nuove_f if self._capisci_concetto(f, impara=True)]
                n_ = len(capite_)
                self.concetti.studiate += capite_
                self.concetti.salva()
                self.annota("ha studiato il libro di scuola: %d frasi nuove (%d capite), %d concetti" % (len(nuove_f), n_, self.concetti.n()))
            fd_ = Path(_Q) / "shim_definizioni_2000.json"
            if os.environ.get("SHIM_CONOSCENZA") == "2000" and fd_.exists():
                d_ = json.loads(fd_.read_text(encoding="utf-8"))
                nuove_ = 0
                for w_, c_ in d_.items():
                    if w_ not in self.concetti.f["è"] and w_ != c_:
                        self.concetti.f["è"][w_] = {c_: 0.5}
                        nuove_ += 1
                self.annota("ha letto le definizioni della Wikipedia: %d" % nuove_)
            fw_ = Path(_Q) / "shim_wikidata_2000.json"
            if os.environ.get("SHIM_CONOSCENZA") == "2000" and fw_.exists():
                d_ = json.loads(fw_.read_text(encoding="utf-8"))
                n_e = n_h = 0
                for w_, c_ in d_.get("è", {}).items():
                    if w_ not in self.concetti.f["è"] and w_ != c_:
                        self.concetti.f["è"][w_] = {c_: 0.5}
                        n_e += 1
                for a_, parti_ in d_.get("ha", {}).items():
                    if a_ not in self.concetti.f["ha"]:
                        self.concetti.f["ha"][a_] = {p_: 0.5 for p_ in parti_[:6] if p_ != a_}
                        n_h += 1
                self.annota("ha letto Wikidata: %d 'e' un', %d 'ha'" % (n_e, n_h))
        return self.concetti

    def _capisci_concetto(self, testo, impara=False):
        from .concetti_shim import capisci
        from .frasi_shim import VERBI, _NOMI_2000
        l = self.cervello.lessico
        if getattr(self, "_noti_set", None) is None or len(self._noti_set) != len(l.parole):
            self._noti_set = set(l.parole)
        if getattr(self, "_nomi_concetti", None) is None:
            from .lessico_2000 import SCUOLA
            from .concetti_shim import QUALITA, VERBI_BASE
            self._nomi_concetti = (set(_NOMI_2000()) | set(SCUOLA)) - QUALITA - VERBI_BASE
        r = capisci(testo, self._noti_set, self._nomi_concetti, set(VERBI))
        if impara and r and r[0] == "impara":
            self.concetti.impara(r[1], r[2], r[3], r[4])
            return True
        if impara and r and r[0] == "impara_no":
            self.concetti.impara_no(r[1], r[2], r[3])
            return True
        return False if impara else r


    VEDI_INSEGNA = re.compile(r"^\s*(?:questo|questa|qui c'?\s?è|ecco)\s+(?:è\s+)?(?:uno\s|una\s|un'|un\s|il\s|lo\s|la\s|l')\s*([a-zàèéìòóù]+)", re.I)
    VEDI_CHIEDI = re.compile(r"(?:cos'?\s?è|che cos'?\s?è|che cosa è|cosa è)\s+(?:questo|questa|qui)|cosa vedi|che cosa vedi|cosa c'?\s?è qui", re.I)

    def _guarda_di_nuovo(self, b, l, volte=1):
        from .occhio import Occhio
        img = getattr(b, "ultima_immagine", None)
        for _ in range(volte):
            n0 = l.finalizzati_img
            corr = b.occhio.correnti(img)
            with b.lock:
                b.occhio_I.copy_(b.torch.as_tensor(corr, device=b.occhio_I.device))
                b.occhio_t = time.time()
            t0 = b.t
            while l.finalizzati_img == n0 and b.t - t0 < 4.0:
                time.sleep(0.05)

    def _vista_nel_dialogo(self, testo, n, ix):
        b, l = self.cervello, self.cervello.lessico
        recente = getattr(b, "ultima_immagine", None) is not None and time.time() - getattr(b, "ultima_immagine_t", 0.0) < 300.0
        m = self.VEDI_INSEGNA.match(testo) if "?" not in testo else None
        if m and recente:
            w = m.group(1).lower()
            from .concetti_shim import singolare, indeterminativo
            w = w if w in ix else singolare(w, ix)
            if w not in ix:
                self.di_frase("Non conosco la parola %s." % m.group(1), [], n)
                return True
            impara0 = l.impara
            l.impara, l.insegnando_img = True, ix[w]
            try:
                self._guarda_di_nuovo(b, l, 3)
            finally:
                l.insegnando_img, l.impara = None, impara0
            l.salva()
            self.annota("ha imparato a vedere: %s (%d esempi)" % (w, int(l.vista.n[ix[w]])))
            self.di_frase("Ho capito: questo è %s." % indeterminativo(w), [w], n)
            return True
        if self.VEDI_CHIEDI.search(testo):
            if not recente:
                self.di_frase("Non vedo niente: fammi vedere qualcosa.", [], n)
                return True
            l.ultima_img = None
            self._guarda_di_nuovo(b, l, 1)
            u = l.ultima_img
            if u is None:
                self.di_frase("Non lo so, non l'ho mai visto.", [], n)
                return True
            visti = self._dai_neuroni({l.parole[u[0]]: 1.0}, None, 1)
            if not visti:
                self.di_frase("Ehm, non mi viene.", [], n)
            else:
                from .concetti_shim import indeterminativo
                self.di_frase("È %s." % indeterminativo(visti[0]), visti[:1], n)
            return True
        return False

    def _di_seguito(self, testo, ix):
        u = getattr(self, "_ultima_domanda", None)
        if u is None or time.time() - getattr(self, "_ultima_domanda_t", 0.0) > 180.0:
            return None
        from .concetti_shim import _parole, singolare
        ps = _parole(testo)
        if not ps or "?" not in testo:
            return None
        if ps[0] == "e" and 1 <= len(ps) <= 3 and ps[-1] not in ("perché", "perche"):
            w = singolare(ps[-1], ix, ps[-2] if len(ps) > 2 else None)
            if w in ix and w != u[1]:
                if u[0] in ("domanda", "quanti"):
                    return (u[0], w, u[2]) + tuple(u[3:4]) if u[0] == "quanti" else ("domanda", w, u[2])
                return (u[0], w, u[2], u[3], u[4] if len(u) > 4 else None)
            return None
        s = " ".join(ps)
        rel = {"cosa è": "è", "che cosa è": "è", "cosa ha": "ha", "che cosa ha": "ha", "cosa mangia": "mangia", "che cosa mangia": "mangia",
               "cosa fa": "fa", "che cosa fa": "fa", "dove vive": "dove", "dove sta": "dove", "come è": "come"}.get(s)
        if rel is not None:
            return ("domanda", u[1], rel)
        if s in ("e perché", "perché", "e perche", "perche") and u[0] == "verifica":
            return ("perche",) + tuple(u[1:])
        return None

    def _dai_neuroni(self, sc, soggetto, quanti):
        b, l = self.cervello, self.cervello.lessico
        ix = {w: i for i, w in enumerate(l.parole)}
        sc = {w: v for w, v in sc.items() if w in ix}
        if not sc:
            return []
        r = np.zeros(l.K, np.float32)
        top_ = max(sc.values())
        for w, v in sc.items():
            r[ix[w]] = np.exp((v - top_) / 0.15)
        quanti = min(quanti, len(sc))
        t_ = t_inizio_ = b.t
        while b.t - t_ < 6.0 and (l.attesa > 0 or b.coclea_coda or b.t < b.parla_fino):
            time.sleep(0.05)
        usciti = []
        for prova_ in range(2):
            with b.lock:
                l.catena, l.catena_veloce = [], False
                l.pensiero = {"seme": int(ix[soggetto]) if soggetto in ix else int(np.argmax(r)), "dette": [], "rimasti": quanti - 1,
                              "r": r, "veloce": True, "concetto": True}
                l.proximo_push_n = max(2, int(l.attesa) + 1)
                l.quiete = 0
            t0 = b.t
            while b.t - t0 < (2.0 + 1.5 * quanti if prova_ == 0 else 3.0 + 2.0 * quanti) and l.pensiero is not None:
                time.sleep(0.05)
            with b.lock:
                usciti = [w for w in (getattr(l, "catena", []) or []) if w in sc]
                l.pensiero = None
                if usciti:
                    l.push[:] = 0
                    l.push_n = 0
                    l.riconosciuta_n = 0
                    l.conferma_k = set()
                    from .parole_shim import PAUSA_VELOCE
                    l.attesa = min(l.attesa, PAUSA_VELOCE)
            if usciti:
                break
        usciti = usciti[:quanti]
        self.annota("dai neuroni: %s, bocca libera dopo %.2f s, parola dopo %.2f s di cervello (tentativo %d)" % (
            usciti, t0 - t_inizio_ if prova_ == 0 else -1.0, b.t - t0, prova_ + 1))
        return usciti

    CONTRARI = {"grande": "piccolo", "alto": "basso", "veloce": "lento", "pesante": "leggero", "caldo": "freddo", "lungo": "corto",
                "forte": "debole", "duro": "morbido", "vecchio": "giovane", "dolce": "amaro"}

    def _frase_no(self, a, rel, x, forma):
        from .concetti_shim import frase, con_articolo
        f = frase(a, rel, [x], {x: forma} if forma else None)
        S = con_articolo(a)
        return (S + " non " + f[len(S) + 1:]).capitalize() if f.lower().startswith(S.lower()) else f

    def _confronta_concetti(self, C, cosa, n):
        from .concetti_shim import con_articolo
        _, q, a, b, verso = cosa
        contr = dict(self.CONTRARI, **{v: k for k, v in self.CONTRARI.items()})
        opposto = contr.get(q)

        def punto(w):
            sc = C.punteggi(w, "come")[0]
            return (1 if q in sc else 0) - (1 if opposto and opposto in sc else 0), sc
        pa, sa = punto(a)
        pb, sb = punto(b)
        if pa == pb:
            self.di_frase("Non lo so.", [], n)
            return True
        vince, perde = (a, b) if (pa > pb) == (verso == "più") else (b, a)
        usciti = self._dai_neuroni({vince: 1.0}, None, 1)
        if not usciti:
            self.di_frase("Ehm, non mi viene.", [], n)
            return True

        def qual(w, sc):
            for x in (q, opposto):
                if x and x in sc:
                    from .concetti_shim import accorda
                    return "%s è %s" % (con_articolo(w), accorda(x, w))
            return None
        spiega = [t for t in (qual(a, sa), qual(b, sb)) if t]
        t = con_articolo(usciti[0]).capitalize() + "." + ((" " + " e ".join(spiega)[:1].upper() + " e ".join(spiega)[1:] + ".") if spiega else "")
        self.di_frase(t, [usciti[0]], n)
        return True

    def _riassunto(self, C, a, n):
        from .concetti_shim import frase
        ix = {w: i for i, w in enumerate(self.cervello.lessico.parole)}
        piano = [("è", 1), ("ha", 2), ("mangia", 1), ("fa", 1), ("dove", 1)]
        punti = [(rel, q, C.punteggi(a, rel)) for rel, q in piano]
        punti = [(rel, q, sv) for rel, q, sv in punti if any(w in ix and w != a for w in sv[0])]
        if len(punti) < 2:
            return False
        frasi_, dette = [], []
        for rel, q, (sc, via) in punti:
            usciti = self._dai_neuroni({w: v for w, v in sc.items() if w != a}, a, q)
            if not usciti:
                continue
            forme = {w: C.forma(a, rel, w, via.get(w)) for w in usciti if C.forma(a, rel, w, via.get(w))}
            f = frase(a, rel, usciti, forme)
            if frasi_:
                S = f.split(" ")
                art = 2 if S[0].lower() in ("il", "lo", "la", "i", "gli", "le") else 1
                f = " ".join(S[art:])
                f = f[:1].upper() + f[1:]
            frasi_.append(f)
            dette += usciti
        if not frasi_:
            self.di_frase("Ehm, non mi viene.", [], n)
        else:
            self.di_frase(" ".join(frasi_), [a] + dette, n)
        return True

    def _in_comune(self, C, a, b, n):
        from .concetti_shim import plurale, con_articolo
        ca, cb = C.categorie(a, 3), C.categorie(b, 3)
        comuni = {c: ca[c] + cb[c] for c in set(ca) & set(cb)}
        ha_a, ha_b = C.punteggi(a, "ha")[0], C.punteggi(b, "ha")[0]
        parti = sorted(set(ha_a) & set(ha_b), key=lambda w: -(ha_a[w] + ha_b[w]))
        if not comuni and not parti:
            self.di_frase("Non so cosa hanno in comune.", [], n)
            return True
        out, dette = [], []
        if comuni:
            usciti = self._dai_neuroni(comuni, None, 1)
            if usciti:
                out.append("Sono tutti e due %s." % plurale(usciti[0]))
                dette += usciti
        if parti:
            usciti = self._dai_neuroni({w: ha_a[w] + ha_b[w] for w in parti}, None, min(2, len(parti)))
            if usciti:
                va = C.punteggi(a, "ha")[1]
                out.append("Hanno %s." % " e ".join(C.forma(a, "ha", w, va.get(w)) or con_articolo(w) for w in usciti))
                dette += usciti
        self.di_frase(" ".join(out) if out else "Ehm, non mi viene.", dette, n)
        return True

    def _risposta_concetto(self, cosa, n):
        from .concetti_shim import frase, con_articolo
        C = self._concetti()
        b, l = self.cervello, self.cervello.lessico
        ix = {w: i for i, w in enumerate(l.parole)}
        tipo = cosa[0]
        if tipo == "sconosciuta":
            self.di_frase("Non conosco la parola %s." % cosa[1], [], n)
            return True
        if tipo == "impara_no":
            _, a, rel, x, forma = cosa
            C.impara_no(a, rel, x)
            C.salva()
            self.annota("ha imparato che no: %s %s %s" % (a, rel, x))
            f_ = self._frase_no(a, rel, x, forma)
            self.di_frase("Ho capito: %s" % (f_[:1].lower() + f_[1:]), [a, x], n)
            return True
        if tipo == "confronto":
            return self._confronta_concetti(C, cosa, n)
        if tipo == "riassunto":
            return self._riassunto(C, cosa[1], n)
        if tipo == "comune":
            return self._in_comune(C, cosa[1], cosa[2], n)
        if tipo == "impara":
            _, a, rel, x, forma = cosa
            C.impara(a, rel, x, forma)
            C.salva()
            self.annota("ha imparato: %s %s %s" % (a, rel, x))
            self.di_frase("Ho capito: %s" % frase(a, rel, [x], {x: forma}).lower(), [a, x], n)
            return True
        if tipo in ("perche", "quanti"):
            from .concetti_shim import indeterminativo, NUMERI_PAROLA
            soggetto, rel, x = cosa[1], cosa[2], cosa[3]
            sc, via = C.punteggi(soggetto, rel)
            if x not in sc:
                if rel == "è" and sc:
                    self.di_frase("Ma non è vero: " + frase(soggetto, rel, [max(sc, key=sc.get)]).lower(), [soggetto], n)
                else:
                    self.di_frase("Non lo so.", [], n)
                return True
            v = via.get(x)
            if rel == "è" and x not in C.f["è"].get(soggetto, {}):
                v = next((c for c in C.f["è"].get(soggetto, {}) if x in C.categorie(c, 2)), None)
            forma_ = C.forma(soggetto, rel, x, v)
            if tipo == "quanti":
                num = next((q for q in (forma_ or "").split() if q in NUMERI_PAROLA), None)
                if num is None:
                    self.di_frase("Non lo so quanti.", [], n)
                    return True
                usciti = self._dai_neuroni({num: 1.0}, soggetto, 1)
                if not usciti:
                    self.di_frase("Ehm, non mi viene.", [], n)
                    return True
                self.di_frase(usciti[0].capitalize() + ". " + frase(soggetto, rel, [x], {x: forma_} if forma_ else None), [soggetto, usciti[0]], n)
                return True
            if v is None:
                self.di_frase("Perché me l'hai insegnato: " + frase(soggetto, rel, [x], {x: forma_} if forma_ else None).lower(), [soggetto, x], n)
                return True
            cand = {c: w_ for c, w_ in C.categorie(soggetto, 2).items() if c == v or (rel != "è" and x in C.punteggi(c, rel)[0])}
            usciti = self._dai_neuroni(cand or {v: 1.0}, soggetto, 1)
            if not usciti:
                self.di_frase("Ehm, non mi viene.", [], n)
                return True
            c = usciti[0]
            fc = C.forma(c, rel, x) or forma_
            self.di_frase("Perché è %s: %s" % (indeterminativo(c), frase(c, rel, [x], {x: fc} if fc else None).lower()), [soggetto, c, x], n)
            return True
        if tipo == "chi":
            _, rel, x = cosa
            sc, via = C.chi(rel, x), {}
            soggetto, quanti = None, 2
        else:
            soggetto, rel = cosa[1], cosa[2]
            sc, via = C.punteggi(soggetto, rel)
            quanti = 1 if rel == "è" or tipo == "verifica" else 2
        sc = {w: v for w, v in sc.items() if w in ix and w != soggetto}
        if tipo == "domanda" and rel == "fa" and sc and all(via.get(w) for w in sc):
            return False
        if tipo == "verifica" and cosa[3] in sc:
            x = cosa[3]
            sc = {x: sc[x]}
            if rel == "è" and x not in C.f["è"].get(soggetto, {}):
                via = {x: next((c for c in C.f["è"].get(soggetto, {}) if x in C.categorie(c, 2)), None)}
        elif tipo == "verifica" and C.negato(soggetto, rel, cosa[3]):
            self.di_frase("No. " + self._frase_no(soggetto, rel, cosa[3], C.forma(soggetto, rel, cosa[3])), [soggetto], n)
            return True
        elif tipo == "verifica" and rel != "è":
            x = cosa[3]
            if True:
                chi_ = [w for w, v in sorted(C.chi(rel, x).items(), key=lambda kv: -kv[1]) if w != soggetto]
                if chi_:
                    self.di_frase("Credo di no. " + frase(chi_[0], rel, [x], {x: C.forma(chi_[0], rel, x)} if C.forma(chi_[0], rel, x) else None), [x], n)
                else:
                    self.di_frase("Non lo so.", [], n)
                return True
        if not sc:
            if tipo == "verifica" or tipo == "chi" or rel in ("è", "ha", "mangia", "dove", "come"):
                self.di_frase("Non lo so.", [], n)
                return True
            return False
        usciti = self._dai_neuroni(sc, soggetto, quanti)
        if not usciti:
            self.di_frase("Ehm, non mi viene.", [], n)
            return True
        forme = {w: C.forma(soggetto, rel, w, via.get(w)) for w in usciti if soggetto and C.forma(soggetto, rel, w, via.get(w))}
        if tipo == "chi":
            t = " e ".join(con_articolo(w) for w in usciti)
            self.di_frase(t[:1].upper() + t[1:] + ".", usciti, n)
        elif tipo == "verifica":
            giusto = cosa[3] in usciti
            spiega = via.get(cosa[3]) if giusto else None
            t = ("Sì. " if giusto else "No. ") + frase(soggetto, rel, usciti, forme)
            if spiega:
                from .concetti_shim import indeterminativo
                t = t[:-1] + ", perché è %s." % indeterminativo(spiega)
            self.di_frase(t, [soggetto] + usciti, n)
        else:
            self.di_frase(frase(soggetto, rel, usciti, forme), [soggetto] + usciti, n)
        return True

    def _dialogo_filo(self, n, trovate, lette, info):
        from .relazioni_scuola import RELAZIONI
        from .parole_shim import I_SEME, I_PAROLA_PA
        from .frasi_shim import VERBI
        b, l = self.cervello, self.cervello.lessico
        ix = {w: i for i, w in enumerate(l.parole)}

        def x():
            return max(0.25, b.x_reale or 0.5)
        try:
            conf = self._confronto(self.turni[n].get("capito", "")) if b.numeri is not None else None
            if conf is not None:
                info["modo"] = "confronto"
                info["risultato"] = self._confronta(conf[0], conf[1], conf[2], n)
                return
            if self._vista_nel_dialogo(self.turni[n].get("capito", ""), n, ix):
                info["modo"] = "vista"
                return
            if CONCETTI:
                cosa_ = self._capisci_concetto(self.turni[n].get("capito", ""))
                if cosa_ is None:
                    cosa_ = self._di_seguito(self.turni[n].get("capito", ""), ix)
                if cosa_ is not None and cosa_[0] in ("domanda", "verifica", "perche", "quanti") and len(cosa_) >= 3:
                    self._ultima_domanda, self._ultima_domanda_t = cosa_, time.time()
                if cosa_ is not None:
                    info["modo"] = "concetto"
                    info["concetto"] = list(cosa_)
                    if self._risposta_concetto(cosa_, n):
                        return
            somma = self._somma(self.turni[n].get("capito", ""))
            if somma is not None:
                info["modo"] = "somma" if somma[2] > 0 else "sottrazione"
                info["risultato"] = self._conta_parole(somma[0], somma[1], n, somma[2])
                return
            if trovate:
                self.di_frase(self._risposta_formule(trovate), [], n)
                cl_ = getattr(l, "_classe", None)
                if cl_ is None or not any(cl_[ix[w]] == 1 for w in lette):
                    return
            if not lette:
                if not trovate:
                    info["modo"] = "non capito"
                    self.di_frase("Non ho capito.", [], n)
                return
            mem = b.memoria
            with b.lock:
                l.pensiero, l.proximo_push_n, l.seme_pendente, l.quiete = None, 0, None, 0
                if mem is not None:
                    mem.svuota_n = 3
            time.sleep(0.35 / x())
            for w in lette:
                with b.lock:
                    l.pensiero, l.proximo_push_n = None, 0
                    l.push[:] = 0
                    l.push[ix[w]] = LEGGI_FORZA * I_SEME * I_PAROLA_PA
                    l.push_n = 5
                time.sleep(0.5 / x())
            da_tenere = [w for w in lette if mem is not None and ix[w] in set(mem.parola.tolist())] if RIPASSO else []
            for _ in range(RIPASSO):
                for w in da_tenere:
                    with b.lock:
                        l.pensiero, l.proximo_push_n = None, 0
                        l.push[:] = 0
                        l.push[ix[w]] = LEGGI_FORZA * I_SEME * I_PAROLA_PA
                        l.push_n = 5
                    time.sleep(0.4 / x())
            time.sleep(0.5 / x())
            in_testa = []
            if mem is not None:
                in_testa = [w for w, z in mem.tenute(l.parole, 4) if z > Z_IN_TESTA]
                if len(in_testa) < min(2, len(lette)):
                    time.sleep(0.8 / x())
                    in_testa = [w for w, z in mem.tenute(l.parole, 4) if z > Z_IN_TESTA]
            info["intrusi"] = [w for w in in_testa if w not in lette]
            in_testa = [w for w in in_testa if w in lette]
            info["in_testa"] = in_testa
            info["da_memoria"] = bool(in_testa)
            usa = in_testa or list(reversed(lette))
            rel = [w for w in usa if w in RELAZIONI]
            altre = [w for w in usa if w not in RELAZIONI]
            if rel and altre:
                info["modo"] = "relazione"
                self.turno_frase = None
                self.ascolta_fino = (n, time.time() + 12.0 / x())
                with b.lock:
                    ok = l.avvia_relazione(rel[0], ix[altre[0]])
                if ok:
                    self._aspetta_detto(n, 10.0, None)
                    return
            classe = getattr(l, "_classe", None)
            if classe is None and _parole_struttura():
                with b.lock:
                    l._struttura(np.zeros(l.K, np.float32), {"svo": True, "dette": [0], "seme": 0})
                classe = getattr(l, "_classe", None)
            nomi = [w for w in altre if classe is not None and classe[ix[w]] == 1]
            if not nomi and classe is not None:
                from .concetti_shim import singolare
                nomi = [s_ for s_ in (singolare(w, ix, "i") for w in altre) if s_ in ix and classe[ix[s_]] == 1]
            if not nomi:
                info["modo"] = "niente"
                if not trovate:
                    self.di_frase("Non ho capito bene.", [], n)
                return
            info["modo"] = "frase"
            k = ix[nomi[0]]
            with b.lock:
                l.catena, l.catena_veloce = [], True
                l.avvia_pensiero(k, 3, veloce=True)
            self.turno_frase = n
            self.ascolta_fino = (n, time.time() + 12.0 / x())
            self._aspetta_detto(n, 8.0, nomi[0])
        except Exception as e:
            print("[SHIM] errore del dialogo: %r" % (e,), flush=True)
        finally:
            self.turni[n]["fine"] = True

    def _aspetta_detto(self, n, secondi_cervello, nome):
        b, l = self.cervello, self.cervello.lessico
        t0 = b.t
        npz = len(self.turni[n]["pezzi"])
        while b.t - t0 < secondi_cervello:
            if len(self.turni[n]["pezzi"]) > npz:
                time.sleep(0.2)
                return
            time.sleep(0.1)
        from .frasi_shim import componi, componi_legate
        with b.lock:
            cat = list(getattr(l, "catena", []) or [])
            l.pensiero, l.catena_veloce = None, False
        if cat:
            self.di_frase(componi_legate(cat) if getattr(l, "catena_forma", "") == "legate" else componi(cat), cat, n)
        elif nome:
            self.di_frase(componi([nome]).rstrip(".") + "... ehm.", [nome], n)
        else:
            self.di_frase("Ehm, non lo so.", [], n)

    def imbarazzo(self):
        b = self.cervello
        a = float(b.lettura.get("amigdala", 0.0) - getattr(b, "globale", 0.0)) if getattr(b, "lettura", None) else 0.0
        return float(np.clip(a / IMBARAZZO_SCALA, 0.0, 1.5))

    def feedback(self, giusto):
        if not giusto:
            self.cervello.stimola("amigdala", IMBARAZZO_PA, 6.0)
            self.cervello.stimola("amigdala", IMBARAZZO_PA * 0.4, IMBARAZZO_S + 14.0)
            self.annota("imbarazzato: gli hanno detto che era sbagliato")
        return {"giusto": bool(giusto), "imbarazzo": round(self.imbarazzo(), 3)}

    def chiedi(self, parole, analogia=False, relazione=None):
        l = self.cervello.lessico
        ks = [l.parole.index(p) for p in parole]
        n = self._prossimo = getattr(self, "_prossimo", 0) + 1
        self.turni = getattr(self, "turni", {})
        st = self.cervello.stato()
        self.turni[n] = {"pezzi": [], "fine": False, "t0": time.time(), "capito": " ".join(parole),
                         "meta": {"gesto": "Idle", "stato": st, "accetta": None, "pronto_s": 0.0}}
        self.ascolta_fino = (n, time.time() + 18.0 / min(1.0, max(0.3, self.cervello.x_reale or 1.0)))
        emb = self.imbarazzo()
        imbarazzato = emb > S_IMBARAZZO
        bugia = bool(imbarazzato and not analogia and any(p in self.segreti for p in parole))
        if imbarazzato:
            self.turni[n]["meta"]["imbarazzo"] = round(emb, 3)
            l.attesa = max(int(l.attesa), 30)
            threading.Thread(target=self.dalla_parola, args=("ehm",), daemon=True).start()
        self.turni[n]["meta"]["bugia"] = bugia
        if relazione:
            l.avvia_relazione(relazione, ks[0])
        elif analogia:
            l.avvia_analogia(ks[0], ks[1], ks[2])
        else:
            l.avvia_risposta(ks, bugia=bugia)

        def chiudi():
            while time.time() < self.ascolta_fino[1] or getattr(self, "_parlando", False):
                time.sleep(0.1)
            self.turni[n]["fine"] = True
        threading.Thread(target=chiudi, daemon=True).start()
        return n

    def insegna_parola(self, parola, chi, var):
        l = self.cervello.lessico
        k = l.parole.index(parola)
        audio, tipo = voce(parola, VOCE if chi == "lui" else VOCE_TU, var)
        pcm = taglia_silenzio(in_16k(audio))
        with self._lock_insegna:
            n0, nk0 = l.finalizzati, int(l.n[k])
            l.insegnando = k
            try:
                self.sente_suono(pcm)
                t_fine = time.time() + len(pcm) / 16000.0 * 2.0 + 30.0
                while l.finalizzati == n0 and time.time() < t_fine:
                    time.sleep(0.05)
                time.sleep(0.15)
            finally:
                l.insegnando = None
        return {"ok": int(l.n[k]) > nk0, "esempi": int(l.n[k])}

    def insegna_pcm(self, parola, pcm):
        l = self.cervello.lessico
        if parola not in l.parole:
            return {"ok": False, "errore": "parola sconosciuta"}
        k = l.parole.index(parola)
        pcm = taglia_silenzio(np.asarray(pcm, np.float32))
        if len(pcm) < 1600 or voce_vera_s(pcm) < 0.15:
            return {"ok": False, "errore": "non ho sentito niente"}
        with self._lock_insegna:
            n0, nk0 = l.finalizzati, int(l.n[k])
            l.insegnando = k
            try:
                self.sente_suono(pcm)
                t_fine = time.time() + len(pcm) / 16000.0 * 2.0 + 30.0
                while l.finalizzati == n0 and time.time() < t_fine:
                    time.sleep(0.05)
                time.sleep(0.15)
            finally:
                l.insegnando = None
        return {"ok": int(l.n[k]) > nk0, "esempi": int(l.n[k])}

    def lallazione(self, minuti):
        b = self.cervello
        t_fine = time.time() + 60 * minuti
        self.annota("lallazione per %.1f minuti" % minuti)
        while time.time() < t_fine:
            if not getattr(self, "_parlando", False) and b.t > b.parla_fino + 0.5 and not b.coclea_coda:
                b.spinta_fino = b.t + 0.3
            time.sleep(2.5)
        b.arcuato.salva()
        self.annota("lallazione finita: esempi per suono %s" % dict(zip(b.bocca.unita, b.arcuato.esempi.tolist())))

    def turno_cervello(self, pcm, sottotitolo=""):
        n = self._prossimo = getattr(self, "_prossimo", 0) + 1
        self.turni = getattr(self, "turni", {})
        st = self.cervello.stato()
        gesto = {"arrabbiato": "No", "contento": "ThumbsUp", "stanco": "Sitting", "spaventato": "Jump"}.get(st["emozione"], "Idle")
        self.turni[n] = {"pezzi": [], "fine": False, "t0": time.time(), "capito": sottotitolo,
                         "meta": {"gesto": gesto, "stato": st, "accetta": None, "pronto_s": 0.0}}
        dur = len(pcm) / 16000.0
        self.ascolta_fino = (n, time.time() + (dur + 4.0) / min(1.0, max(0.3, self.cervello.x_reale or 1.0)))
        self.sente_suono(pcm)
        self.cervello.ultimo_ascolto = self.cervello.t
        self.annota("sente %.1f s di suono%s" % (dur, (" ('%s')" % sottotitolo) if sottotitolo else ""))

        def chiudi():
            while time.time() < self.ascolta_fino[1] or getattr(self, "_parlando", False):
                time.sleep(0.1)
            self.turni[n]["fine"] = True
        threading.Thread(target=chiudi, daemon=True).start()
        return n

    def _pensiero_muto(self, testo):
        self.pensieri = getattr(self, "pensieri", [])
        self.pensieri.append({"t": round(time.time(), 1), "testo": testo})
        self.pensieri = self.pensieri[-60:]
        self.annota("pensa (in silenzio): %s" % testo)

    def dalla_bocca(self, sillabe):
        from .voce_cervello import scrivi
        b = self.cervello
        testo = scrivi(sillabe)
        if PENSA_IN_SILENZIO:
            self._pensiero_muto("(balbetta) " + testo)
            return
        self._parlando = True
        try:


            pezzi_pcm = []
            for c, v in sillabe:
                a_s, t_s = voce(scrivi([(c, v)]))
                pezzi_pcm.append(taglia_silenzio(in_16k(a_s)) if t_s == "audio/wav" else np.zeros(1600, np.float32))
            pausa = np.zeros(800, np.float32)
            pcm = np.concatenate([x for p in pezzi_pcm for x in (p, pausa)])
            buf = io.BytesIO()
            import wave
            with wave.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
            audio, tipo = buf.getvalue(), "audio/wav"
            dur = len(pcm) / 16000.0
            b.parla_fino = max(b.parla_fino, b.t) + dur
            for p_s, s_s in zip(pezzi_pcm, sillabe):
                self.sente_suono(np.concatenate([p_s, pausa]), [s_s])
            pezzo = {"testo": testo, "audio": base64.b64encode(audio).decode(), "tipo_audio": tipo, "verso": False,
                     "vocali": [v for _, v in sillabe], "consonanti": [c for c, _ in sillabe]}
            n, fino = self.ascolta_fino
            if time.time() <= fino and n in getattr(self, "turni", {}):
                self.turni[n]["pezzi"].append(pezzo)
                self.ascolta_fino = (n, min(max(fino, time.time() + dur + 1.0), self.turni[n]["t0"] + 15.0 / min(1.0, max(0.3, b.x_reale or 1.0))))
                self.annota("dice (dal cervello, z %.2f): %s" % (b.bocca.z_medio, testo))
            else:
                k = self._prossimo = getattr(self, "_prossimo", 0) + 1
                self.turni = getattr(self, "turni", {})
                st = b.stato()
                self.turni[k] = {"pezzi": [pezzo], "fine": True, "t0": time.time(),
                                 "meta": {"gesto": "Idle", "stato": st, "accetta": None, "pronto_s": 0.0}}
                self.spontanei.append(k)
                self.annota("dice da solo (dal cervello): %s" % testo)
        finally:
            self._parlando = False

    def annota(self, cosa):
        self.diario.append({"t": round(self.cervello.t, 1), "cosa": cosa})
        self.diario = self.diario[-30:]
        print("[SHIM] %s" % cosa, flush=True)

    def senti(self, testo, durata):
        b = self.cervello
        b.ultimo_ascolto = b.t
        b.stimola("udito", I_UDITO, max(durata, 0.8))
        b.stimola("ascolto", I_ASCOLTO, max(durata, 0.8) + 0.5)
        basso = testo.lower()
        if any(p in basso for p in PAROLE_INSULTO):
            b.stimola("amigdala", I_INSULTO, 3.0)
            self.annota("si sente insultato")
        if any(p in basso for p in PAROLE_AFFETTO):
            b.stimola("accumbens", I_AFFETTO, 3.0)
            self.annota("gli fa piacere")

    def _decidi(self, testo, durata, aspetta_s):
        b = self.cervello
        self.senti(testo, durata)
        t_fine = b.t + aspetta_s
        while b.t < t_fine:
            time.sleep(0.02)
        st = b.stato()
        richiesta = any(re.search(r"\b%s\b" % re.escape(p), testo.lower()) for p in PAROLE_RICHIESTA)
        decisione, gesto = None, None
        if richiesta:
            p, rumore = b.voglia()
            accetta = random.random() < p
            decisione = ("Ti chiedono una cosa: HAI DECISO DI SI', la fai (a modo tuo)."
                         if accetta else "Ti chiedono una cosa: HAI DECISO DI NO, non ne hai voglia: rifiuta.")
            gesto = "Yes" if accetta else "No"
            self.annota("richiesta: voglia %.2f (rumore dei neuroni %+.2f) -> %s" % (p, rumore, "si'" if accetta else "no"))
        return st, decisione, gesto, richiesta


    def turno(self, testo, durata=1.5, spontaneo=False):
        n = self._prossimo = getattr(self, "_prossimo", 0) + 1
        self.turni = getattr(self, "turni", {})
        self.turni[n] = {"pezzi": [], "fine": False, "meta": None, "t0": time.time()}
        for vecchio in [k for k in self.turni if k < n - 20]:
            del self.turni[vecchio]
        threading.Thread(target=self._turno, args=(n, testo, durata, spontaneo), daemon=True).start()
        return n

    def _turno(self, n, testo, durata, spontaneo):
        import queue
        b, T = self.cervello, self.turni[n]
        try:
            if spontaneo:
                st, decisione, gesto, richiesta = b.stato(), None, None, False
                spunti = ["Ti annoi: di' qualcosa a chi e' nella stanza, oppure canticchia due righe di una canzone inventata.",
                          "Ti e' venuto in mente un pensiero strano: dillo ad alta voce.",
                          "Fai una domanda curiosa a chi e' con te."]
                frase = "(nessuno parla da un po'. %s)" % random.choice(spunti)
                gesto = "Dance" if st["energia"] > 0.5 and st["umore"] >= 0 else "Idle"
                b.ultimo_ascolto = b.t
            else:
                st, decisione, gesto, richiesta = self._decidi(testo, durata, 0.15)
                frase = testo
            if gesto is None:
                gesto = {"arrabbiato": "No", "contento": "ThumbsUp", "stanco": "Sitting", "annoiato": "Idle",
                         "spaventato": "Jump", "triste": "Idle"}.get(st["emozione"], "Wave" if len(self.storia) <= 2 else "Idle")
            T["meta"] = {"gesto": gesto, "stato": st, "accetta": (gesto == "Yes") if richiesta else None,
                         "pronto_s": round(time.time() - T["t0"], 2)}
            frasi = queue.Queue()

            def voci():
                while True:
                    f = frasi.get()
                    if f is None:
                        break
                    verso = isinstance(f, tuple)
                    if verso:
                        f = f[1]
                    audio, tipo = voce_verso(f) if verso else voce(f)
                    dur = max(0.6, 0.075 * len(f))
                    b.parla_fino = max(b.parla_fino, b.t) + dur
                    b.stimola("parola", I_PAROLA, dur)
                    b.stimola("udito", I_UDITO * 0.4, dur)
                    T["pezzi"].append({"testo": f, "audio": base64.b64encode(audio).decode(), "tipo_audio": tipo,
                                       "pronto_s": round(time.time() - T["t0"], 2), "verso": verso})
            tv = threading.Thread(target=voci, daemon=True)
            tv.start()
            if not spontaneo and VERSO_SUBITO:
                frasi.put(("verso", random.choice(VERSI.get(st["emozione"], VERSI["tranquillo"]))))
            dette = []
            with self.lock:
                tempi = {"deciso": time.time()}
                for k, f in enumerate(scrivi_a_pezzi(self.storia, frase, st, decisione, tempi)):
                    if k == 0:
                        tempi["primo_pezzo"] = time.time()
                    if not f:
                        continue
                    if k == 0 and richiesta:
                        sbaglia = (gesto == "No" and not RIFIUTO.search(f.lower())) or (
                            gesto == "Yes" and re.search(r"\b(fallo tu|prendilo tu|scordatelo|non lo faccio)\b", f.lower()))
                        if sbaglia:
                            f = scrivi_risposta(self.storia, frase, st, decisione + (" Dillo chiaramente: non lo fai."
                                                if gesto == "No" else " Dillo chiaramente: lo fai."))
                            self.annota("riscritto: il cervello aveva detto %s" % ("no" if gesto == "No" else "si'"))
                            tempi["grezzo"] = None
                            dette.append(f)
                            frasi.put(f)
                            break
                    dette.append(f)
                    frasi.put(f)
                parole = " ".join(dette)
                self.storia += _utente(frase, st, decisione)
                self.storia.append({"role": "assistant", "content": tempi.get("grezzo") or parole})
            threading.Thread(target=self._prepara, daemon=True).start()
            frasi.put(None)
            tv.join()
            self.annota("dice (%s): %s" % (st["emozione"], parole))
            t0 = T["t0"]
            print("[SHIM] tempi: deciso %.2f s, primo token %.2f, primo pezzo %.2f, prima voce %.2f; riletti %s%s" % (
                tempi["deciso"] - t0, tempi.get("primo_token", t0) - t0, tempi.get("primo_pezzo", t0) - t0,
                next((p["pronto_s"] for p in T["pezzi"] if not p["verso"]), -1), tempi.get("letti"),
                " (tagliato)" if tempi.get("tagliato") else ""), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            T["errore"] = str(e)
        finally:
            T["fine"] = True

    def _prepara(self):
        try:
            with self.lock:
                prepara(self.storia)
        except Exception as e:
            print("[SHIM] preparare non va: %s" % e, flush=True)

    def rispondi(self, testo, durata=1.5, aspetta_s=0.3):
        b = self.cervello
        st, decisione, gesto, richiesta = self._decidi(testo, durata, aspetta_s)
        with self.lock:
            parole = scrivi_risposta(self.storia, testo, st, decisione)
            if richiesta and gesto == "No" and not RIFIUTO.search(parole.lower()):
                parole = scrivi_risposta(self.storia, testo, st, decisione + " Dillo chiaramente: non lo fai.")
                self.annota("riscritto: il cervello aveva detto no")
            elif richiesta and gesto == "Yes" and re.search(r"\b(fallo tu|prendilo tu|scordatelo|non lo faccio)\b", parole.lower()):
                parole = scrivi_risposta(self.storia, testo, st, decisione + " Dillo chiaramente: lo fai.")
                self.annota("riscritto: il cervello aveva detto si'")
            self.storia += _utente(testo, st, decisione) + [{"role": "assistant", "content": parole}]
        return self._parla(parole, st, gesto, accetta=(gesto == "Yes") if richiesta else None)

    def spontaneo(self):
        b = self.cervello
        st = b.stato()
        spunti = ["Ti annoi: di' qualcosa a chi e' nella stanza, oppure canticchia due righe di una canzone inventata.",
                  "Ti e' venuto in mente un pensiero strano: dillo ad alta voce.",
                  "Fai una domanda curiosa a chi e' con te."]
        with self.lock:
            spunto = "(nessuno parla da un po'. %s)" % random.choice(spunti)
            parole = scrivi_risposta(self.storia, spunto, st, None)
            self.storia += _utente(spunto, st, None) + [{"role": "assistant", "content": parole}]
        b.ultimo_ascolto = b.t
        return self._parla(parole, st, "Dance" if st["energia"] > 0.5 and st["umore"] >= 0 else "Idle", accetta=None)

    def _parla(self, parole, st, gesto, accetta):
        b = self.cervello
        audio, tipo = voce(parole)
        durata = max(1.0, 0.075 * len(parole))
        b.parla_fino = b.t + durata
        b.stimola("parola", I_PAROLA, durata)
        b.stimola("udito", I_UDITO * 0.4, durata)
        if gesto is None:
            gesto = {"arrabbiato": "No", "contento": "ThumbsUp", "stanco": "Sitting", "annoiato": "Idle",
                     "spaventato": "Jump", "triste": "Idle"}.get(st["emozione"], "Wave" if len(self.storia) <= 2 else "Idle")
        self.annota("dice (%s): %s" % (st["emozione"], parole))
        return {"testo": parole, "audio": base64.b64encode(audio).decode(), "tipo_audio": tipo, "gesto": gesto,
                "accetta": accetta, "stato": st}


LUI: SHIM = None


class Gestore(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _invia(self, dati, tipo, codice=200):
        self.send_response(codice)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(dati)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(dati)

    def _json(self, o):
        self._invia(json.dumps(o, ensure_ascii=False).encode("utf-8"), "application/json")

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p in ("/", "/chat", "/chat.html", "/shim"):
                self._invia((QUI / "chat.html").read_bytes(), "text/html; charset=utf-8")
            elif p in ("/mondo", "/mondo.html") and False:
                self._invia((QUI / "mondo_shim.html").read_bytes(), "text/html; charset=utf-8")
            elif p in ("/", "/vecchia.html"):
                self._invia((QUI / "chat.html").read_bytes(), "text/html; charset=utf-8")
            elif p == "/parole_da_insegnare":
                f_ = QUI / "parole_ascolto_300.json"
                l_ = LUI.cervello.lessico
                ps_ = [w for w in json.loads(f_.read_text(encoding="utf-8")) if l_ is not None and w in l_.parole] if f_.exists() else []
                self._json({"parole": ps_, "esempi": {w: int(l_.n[l_.parole.index(w)]) for w in ps_} if l_ is not None else {}})
            elif p in ("/chat", "/chat.html"):
                self._invia((QUI / "chat.html").read_bytes(), "text/html; charset=utf-8")
            elif p == "/corpo_umano.js":
                self._invia((QUI / "corpo_umano.js").read_bytes(), "text/javascript; charset=utf-8")
            elif p == "/visto":
                b = LUI.cervello
                b.occhio_guardato = time.time()
                gw, gh = b.occhio.risoluzione()
                im = b.visto if b.visto is not None else np.zeros((gh, gw))
                self._json({"w": int(im.shape[1]), "h": int(im.shape[0]), "neuroni_v1": int(len(b.occhio.neuroni)),
                            "gw_sx": int(b.occhio.lati["sx"]["gw"]),
                            "pixel": np.round(im * 127).astype(int).reshape(-1).tolist(),
                            "telecamera": time.time() - b.occhio_t < 0.5, "energia": round(float(getattr(b.occhio, "energia", 0.0)), 1)})
            elif p == "/spari":
                b = LUI.cervello
                b.guardato = time.time()
                f, hz, bit = b.spari
                self._invia(np.array([f], np.uint32).tobytes() + hz + bit, "application/octet-stream")
            elif p == "/stato":
                st = LUI.cervello.stato()
                st["diario"] = LUI.diario[-8:]
                if time.time() - getattr(LUI, "_npu_t", 0) > 20:
                    LUI._npu_t, LUI._npu = time.time(), orecchie_npu_accese()
                st["orecchie"] = "npu" if LUI._npu else "processore"
                st["solo_cervello"] = SOLO_CERVELLO
                st["imbarazzo"] = round(LUI.imbarazzo(), 3)
                st["bocca_z"] = round(LUI.cervello.bocca.z_medio, 2)
                st["fascio_sente"] = LUI.cervello.arcuato.percepito[-12:]
                bb = LUI.cervello.bocca
                if getattr(bb, "z_ultimo", None) is not None:
                    st["bocca_vocali_z"] = [round(float(v), 1) for v in bb.z_ultimo[:5]]
                    st["bocca_vocali_hz"] = [round(float(v), 1) for v in bb.x_ultimo[:5]]
                    st["bocca_media_hz"] = [round(float(v), 1) for v in bb.media[:5]]
                self._json(st)
            elif p == "/spontaneo":
                self._json(LUI.spontaneo())
            elif p == "/pensieri":
                from urllib.parse import parse_qs, urlparse
                q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                da = float(q.get("da", "0"))
                self._json({"pensieri": [x for x in getattr(LUI, "pensieri", []) if x["t"] > da]})
            elif p == "/spontaneo_turno":
                if SOLO_CERVELLO:
                    self._json({"id": LUI.spontanei.pop(0) if LUI.spontanei else 0})
                else:
                    self._json({"id": LUI.turno("", spontaneo=True)})
            elif p == "/pezzi":
                from urllib.parse import parse_qs, urlparse
                q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                T = getattr(LUI, "turni", {}).get(int(q.get("id", "0")))
                if T is None:
                    self._json({"errore": "turno sconosciuto", "fine": True}); return
                da = int(q.get("da", "0"))
                self._json({"pezzi": T["pezzi"][da:], "fine": T["fine"], "meta": T["meta"], "errore": T.get("errore"), "info": T.get("info")})
            elif p.startswith("/lib/") or p.startswith("/modelli/"):
                base = LIB if p.startswith("/lib/") else QUI / "modelli"
                f = (base / p.split("/", 2)[2]).resolve()
                if base.resolve() not in f.parents or not f.is_file():
                    self.send_error(404); return
                tipo = {".js": "text/javascript", ".glb": "model/gltf-binary"}.get(f.suffix, "application/octet-stream")
                self._invia(f.read_bytes(), tipo)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        p = self.path.split("?")[0]
        dati = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        try:
            if p == "/lallazione":
                d = json.loads(dati or b"{}")
                threading.Thread(target=LUI.lallazione, args=(float(d.get("minuti", 3)),), daemon=True).start()
                self._json({"ok": True}); return
            if p == "/prova_bocca":
                d = json.loads(dati or b"{}")
                b = LUI.cervello
                b.arcuato.prove[:] = 0
                b.arcuato.prove[b.bocca.unita.index(d.get("vocale", "a"))] = float(d.get("prove", 5.0))
                b.bocca_I = b.arcuato._spinta()
                b.spinta_fino = b.t + 0.3
                self._json({"ok": True}); return
            if p == "/arcuato":
                d = json.loads(dati or b"{}")
                a = LUI.cervello.arcuato
                if d.get("azzera"):
                    a.azzera()
                if "riadatta" in d:
                    if d["riadatta"]:
                        a.mu[:] = 0; a.var[:] = 1; a.n_norm = 0; a.forza = 0.0
                    a.solo_adatta = bool(d["riadatta"])
                if d.get("salva"):
                    a.salva()
                if "impara" in d:
                    a.impara = bool(d["impara"])
                if d.get("registra") is True:
                    a.registra = []
                elif d.get("registra"):
                    reg = a.registra or []
                    np.savez(d["registra"], f=np.array([x[0] for x in reg], np.float32),
                             vocale=np.array([(x[1][0] if x[1] else -1) for x in reg]),
                             consonante=np.array([(x[1][1] if x[1] and len(x[1]) > 1 else -1) for x in reg]))
                    a.registra = None
                self._json({"esempi": a.esempi.tolist(), "impara": a.impara}); return
            if p == "/lessico":
                d = json.loads(dati or b"{}")
                l = LUI.cervello.lessico
                if l is None:
                    self._json({"errore": "niente lessico"}); return
                if d.get("azzera"):
                    l.azzera()
                if "impara" in d:
                    l.impara = bool(d["impara"])
                if "insegnando_img" in d:
                    l.insegnando_img = d["insegnando_img"]
                if d.get("salva"):
                    l.salva()
                self._json({"parole": l.parole, "esempi": l.n.tolist(), "impara": l.impara, "ultima": l.ultima, "nS": l.nS,
                            "finalizzati": l.finalizzati, "esempi_img": l.vista.n.tolist(), "finalizzati_img": l.finalizzati_img,
                            "ultima_img": l.ultima_img, "sconosciute": l.n_sconosciute,
                            "sconosciute_img": l.n_sconosciute_img}); return
            if p == "/insegna_frasi":
                d = json.loads(dati or b"{}")
                l = LUI.cervello.lessico
                for f in d.get("frasi", []):
                    l.assoc.impara(f)
                if d.get("salva"):
                    l.assoc.salva()
                self._json({"frasi": len(d.get("frasi", [])), "somma": float(l.assoc.N.sum())}); return
            if p == "/feedback":
                d = json.loads(dati or b"{}")
                self._json(LUI.feedback(bool(d.get("giusto", True)))); return
            if p == "/segreto":
                d = json.loads(dati or b"{}")
                if d.get("azzera"):
                    LUI.segreti = set()
                LUI.segreti |= set(d.get("parole", []))
                self._json({"segreti": sorted(LUI.segreti)}); return
            if p == "/chiedi":
                d = json.loads(dati or b"{}")
                if d.get("da_memoria"):
                    b_ = LUI.cervello
                    if b_.memoria is None:
                        self._json({"errore": "memoria di lavoro spenta (SHIM_MEMORIA=1)"}); return
                    d["parole"] = [w for w, _ in b_.memoria.tenute(b_.lessico.parole, int(d.get("quante", 3)))]
                    if not d["parole"]:
                        self._json({"id": None, "in_testa": []}); return
                    from .relazioni_scuola import RELAZIONI
                    rel_ = [w for w in d["parole"] if w in RELAZIONI]
                    altre_ = [w for w in d["parole"] if w not in RELAZIONI]
                    if rel_ and altre_:
                        d["relazione"], d["parole"] = rel_[0], altre_[:1]
                self._json({"id": LUI.chiedi(d["parole"], bool(d.get("analogia")), d.get("relazione")), "in_testa": d["parole"]}); return
            if p == "/numeri":
                d = json.loads(dati or b"{}")
                b_ = LUI.cervello
                nm = b_.numeri
                if nm is None:
                    self._json({"errore": "numeri spenti (SHIM_NUMERI=1)"}); return
                with b_.lock:
                    if "pensa" in d:
                        nm.spinta = (float(d["pensa"]), float(d.get("pa", 300.0)), b_.t + float(d.get("s", 0.3)))
                    if "via" in d:
                        nm.via_fino = b_.t + float(d["via"])
                    if "via_pa" in d:
                        nm.via_pa = float(d["via_pa"])
                    if "g_succ" in d or "g_pred" in d:
                        nm.imposta(b_.m, d.get("g_succ"), d.get("g_pred"))
                    for k_ in ("g_tieni", "x_tieni", "inib"):
                        if k_ in d:
                            setattr(nm, k_, float(d[k_]))
                    if d.get("svuota"):
                        nm.svuota()
                    if "g_lavoro" in d:
                        nm.g_lavoro = float(d["g_lavoro"])
                    if "quantita" in d:
                        from .numeri_shim import SIGMA_LOG
                        nm.spinta = (float(d["quantita"]), float(d.get("pa", 150.0)), b_.t + float(d.get("s", 0.3)), SIGMA_LOG)
                    if "conta" in d:
                        nm.avvia_conteggio(b_.t, int(d["conta"]), int(d.get("verso", 1)))
                cn_ = nm.conteggio or {}
                gr_ = nm.grandezza()
                self._json({"in_mente": nm.in_mente(), "grandezza": None if gr_ is None else round(gr_, 2),
                            "z": [round(float(v_), 2) for v_ in nm.z], "s": [round(float(v_), 2) for v_ in nm.s],
                            "contati": cn_.get("lista"), "finito": bool(cn_.get("finito")), "t": round(b_.t, 2)}); return
            if p == "/memoria":
                d = json.loads(dati or b"{}")
                b_ = LUI.cervello
                if b_.memoria is None:
                    self._json({"errore": "memoria di lavoro spenta (SHIM_MEMORIA=1)"}); return
                mem, l_ = b_.memoria, b_.lessico
                if "ripasso" in d:
                    global RIPASSO
                    RIPASSO = int(d["ripasso"])
                with b_.lock:
                    if any(k in d for k in ("g_rec", "g_ff", "g_fb")):
                        mem.imposta(b_.m, d.get("g_rec"), d.get("g_ff"), d.get("g_fb"))
                    for k_ in ("g_nmda", "inib", "alfa", "x_nmda"):
                        if k_ in d:
                            setattr(mem, k_, float(d[k_]))
                    if d.get("svuota"):
                        mem.svuota_n = int(d.get("finestre", 3))
                    if d.get("spingi"):
                        mem.spinta = (mem.slot[l_.parole.index(d["spingi"])], float(d.get("pa", 300.0)), b_.t + float(d.get("s", 0.3)))
                    if d.get("senti"):
                        k_ = l_.parole.index(d["senti"])
                        l_.pensiero, l_.proximo_push_n, l_.seme_pendente, l_.quiete = None, 0, None, 0
                        l_.push[:] = 0
                        from .parole_shim import I_SEME, I_PAROLA_PA
                        l_.push[k_] = float(d.get("forza", 1.0)) * I_SEME * I_PAROLA_PA
                        l_.push_n = int(d.get("finestre", 6))
                z_ = mem.z
                sonda_ = None
                if d.get("sonda"):
                    k_ = l_.parole.index(d["sonda"])
                    cod_ = l_.codici[k_] if l_.sparso else np.flatnonzero(l_.Gm[k_] > 0)
                    cnt_ = getattr(b_, "ultima_conta_parola", None)
                    tutti_ = float(cnt_.mean() / 0.1) if cnt_ is not None else None
                    sonda_ = {"codice_hz": round(float(cnt_[cod_].mean() / 0.1), 2) if cnt_ is not None else None, "linguaggio_hz": tutti_,
                              "assemblea_z": round(float(mem.z[mem.slot[k_]]), 2) if k_ in mem.slot else None,
                              "s": round(float(mem.s[mem.slot[k_]]), 3) if k_ in mem.slot else None,
                              "push": float(l_.push[k_]), "push_n": int(l_.push_n), "sparso": bool(l_.sparso), "n_codice": int(len(cod_))}
                self._json({"sonda": sonda_, "in_testa": mem.tenute(l_.parole, 8), "z_max": round(float(z_.max()), 2), "z_media": round(float(z_.mean()), 3),
                            "t": round(b_.t, 1), "g": getattr(mem, "g", None),
                            "nmda": {"g_nmda": mem.g_nmda, "inib": mem.inib, "alfa": mem.alfa, "s_max": round(float(mem.s.max()), 3),
                                     "s_accese": int((mem.s > 0.5).sum())}}); return
            if p == "/quanti":
                d = json.loads(dati or b"{}")
                b_ = LUI.cervello
                if b_.numeri is None:
                    self._json({"errore": "numeri spenti (SHIM_NUMERI=1)"}); return
                cg = float(getattr(b_.occhio, "contrasto_grezzo", 0.0))
                if "insegna" in d:
                    LUI.quanti_esempi = LUI._quanti_carica() + [(cg, int(d["insegna"]), str(d.get("contesto", "puntini")))]
                    LUI._quanti_salva()
                    self._json({"esempi": len(LUI.quanti_esempi), "contrasto": round(cg, 1)}); return
                n_ = LUI._prossimo = getattr(LUI, "_prossimo", 0) + 1
                LUI.turni = getattr(LUI, "turni", {})
                LUI.turni[n_] = {"pezzi": [], "fine": False, "t0": time.time(), "capito": "(quanti sono?)",
                                 "meta": {"gesto": "Idle", "stato": b_.stato(), "accetta": None, "pronto_s": 0.0}}
                threading.Thread(target=LUI.quanti_sono, args=(n_, cg, str(d.get("contesto", "puntini"))), daemon=True).start()
                self._json({"id": n_, "contrasto": round(cg, 1)}); return
            if p == "/guarda_e_di":
                d = json.loads(dati or b"{}")
                b_ = LUI.cervello
                n_ = LUI._prossimo = getattr(LUI, "_prossimo", 0) + 1
                LUI.turni = getattr(LUI, "turni", {})
                LUI.turni[n_] = {"pezzi": [], "fine": False, "t0": time.time(), "capito": "(guarda)",
                                 "meta": {"gesto": "Idle", "stato": b_.stato(), "accetta": None, "pronto_s": 0.0}}
                sec_ = float(d.get("s", 8.0)) / min(1.0, max(0.25, b_.x_reale or 0.5))
                LUI.ascolta_fino = (n_, time.time() + sec_)

                def _chiudi(n__=n_, fino=time.time() + sec_):
                    while time.time() < fino:
                        time.sleep(0.1)
                    LUI.turni[n__]["fine"] = True
                threading.Thread(target=_chiudi, daemon=True).start()
                self._json({"id": n_}); return
            if p == "/dialogo":
                d = json.loads(dati or b"{}")
                n_, info_ = LUI.dialogo(d.get("testo", ""))
                self._json(dict(info_, id=n_)); return
            if p == "/pensa":
                d = json.loads(dati or b"{}")
                self._json({"id": LUI.pensa(d["parola"], int(d.get("lunghezza", 4)))}); return
            if p == "/insegna_parola":
                d = json.loads(dati or b"{}")
                self._json(LUI.insegna_parola(d["parola"], d.get("voce", "lui"), d.get("var"))); return
            if p == "/turno" and SOLO_CERVELLO:
                d = json.loads(dati or b"{}")
                audio, tipo = voce(d.get("testo", ""), VOCE if d.get("voce") == "lui" else VOCE_TU, d.get("var"))
                pcm_t = in_16k(audio)
                if d.get("taglia"):
                    pcm_t = taglia_silenzio(pcm_t)
                self._json({"id": LUI.turno_cervello(pcm_t, d.get("testo", "")), "capito": d.get("testo", "")})
                return
            if p == "/turno":
                d = json.loads(dati or b"{}")
                self._json({"id": LUI.turno(d.get("testo", ""), float(d.get("durata", 1.5))), "capito": d.get("testo", "")})
                return
            if p == "/insegna_pcm":
                from urllib.parse import parse_qs, urlparse
                q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                r_ = LUI.insegna_pcm(q.get("parola", ""), np.frombuffer(dati, np.float32).copy())
                if r_.get("ok") and int(q.get("salva", "0")):
                    LUI.cervello.lessico.salva()
                self._json(r_); return
            if p == "/dialogo_pcm":
                pcm = np.frombuffer(dati, np.float32).copy()
                testo, _ = ascolta_pcm(pcm)
                if not testo or len(testo) < 2:
                    self._json({"capito": ""}); return
                if re.search(r"\bquant[ie]\b", testo, re.I) and not re.search(r"\bfa\b|più|meno|\bha\b|\bhanno\b", testo, re.I):
                    self._json({"capito": testo, "quanti": True}); return
                n_, info_ = LUI.dialogo(testo)
                self._json(dict(info_, id=n_, capito=testo)); return
            if p == "/senti_pcm":
                pcm = np.frombuffer(dati, np.float32).copy()
                if voce_vera_s(pcm) < 0.15:
                    self._json({"errore": "non ho sentito niente"}); return
                self._json({"id": LUI.turno_cervello(taglia_silenzio(pcm), "")}); return
            if p == "/ascolta_pcm_turno":
                t_o = time.time()
                pcm = np.frombuffer(dati, np.float32).copy()
                if SOLO_CERVELLO:
                    if voce_vera_s(pcm) < 0.25:
                        self._json({"capito": ""}); return
                    testo, _ = ascolta_pcm(pcm)
                    self._json({"id": LUI.turno_cervello(pcm, testo), "capito": testo or "(un suono)"})
                    return
                testo, durata = ascolta_pcm(pcm)
                print("[SHIM] orecchie: %.2f s -> '%s'" % (time.time() - t_o, testo), flush=True)
                if not testo or len(testo) < 2:
                    self._json({"capito": ""}); return
                self._json({"id": LUI.turno(testo, durata), "capito": testo})
                return
            if p == "/ascolta_turno":
                t_o = time.time()
                testo, durata = ascolta(dati)
                print("[SHIM] orecchie: %.2f s -> '%s'" % (time.time() - t_o, testo), flush=True)
                if not testo or len(testo) < 2:
                    self._json({"capito": ""}); return
                self._json({"id": LUI.turno(testo, durata), "capito": testo})
                return
            if p == "/ascolta":
                t0 = time.time()
                testo, durata = ascolta(dati)
                if not testo or len(testo) < 2:
                    self._json({"capito": ""}); return
                r = LUI.rispondi(testo, durata)
                r["capito"], r["secondi"] = testo, round(time.time() - t0, 2)
                self._json(r)
            elif p == "/occhio":
                from urllib.parse import parse_qs, urlparse
                q = {k: int(v[0]) for k, v in parse_qs(urlparse(self.path).query).items()}
                img = np.frombuffer(dati, np.uint8).reshape(q["h"], q["w"])
                b = LUI.cervello
                corr = b.occhio.correnti(img)
                with b.lock:
                    b.occhio_I.copy_(b.torch.as_tensor(corr, device=b.occhio_I.device))
                    b.occhio_t = time.time()
                b.ultima_immagine, b.ultima_immagine_t = img.copy(), time.time()
                self._json({"ok": True})
            elif p == "/scrivi":
                d = json.loads(dati or b"{}")
                r = LUI.rispondi(d.get("testo", ""), float(d.get("durata", 1.5)))
                r["capito"] = d.get("testo", "")
                self._json(r)
            elif p == "/corpo":
                d = json.loads(dati or b"{}")
                if d.get("ti_vede"):
                    LUI.cervello.stimola("vista", I_VISTA * float(d.get("vicinanza", 1.0)), 0.3)
                self._json({"ok": True})
            elif p == "/energia":
                LUI.cervello.energia = float(json.loads(dati)["energia"])
                self._json({"ok": True})
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"errore": str(e)})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global LUI
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    N = int(args[0]) if args else 250000
    porta = int(sys.argv[sys.argv.index("--porta") + 1]) if "--porta" in sys.argv else 8800
    LUI = SHIM(N)
    LUI.cervello.start()
    while LUI.cervello.riposo is None:
        time.sleep(0.2)
    def _carica_orecchie():
        global _WHISPER
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            WhisperModel = None
        if _WHISPER is None and WhisperModel is not None:
            _WHISPER = WhisperModel(MODELLO_ORECCHIE, device="cpu", compute_type="int8")
        print("[SHIM] orecchie pronte (Whisper %s)" % MODELLO_ORECCHIE, flush=True)
        voce("ciao")
        for v in sum(VERSI.values(), []):
            voce_verso(v)
        print("[SHIM] voce pronta (%s)" % VOCE, flush=True)
        if SOLO_CERVELLO:
            return
        try:
            t0 = time.time()
            LUI._prepara()
            print("[SHIM] parole pronte (%s, %.1f s)" % (MODELLO_LLM, time.time() - t0), flush=True)
        except Exception as e:
            print("[SHIM] il modello delle parole non risponde: %s" % e, flush=True)
    threading.Thread(target=_carica_orecchie, daemon=True).start()
    print("SHIM e' sveglio: http://127.0.0.1:%d/  (%d neuroni, %d sinapsi, %d gruppi)" % (
        porta, LUI.cervello.c.n, LUI.cervello.c.n_sinapsi, LUI.cervello.c.na), flush=True)
    Server(("127.0.0.1", porta), Gestore).serve_forever()


if __name__ == "__main__":
    main()
