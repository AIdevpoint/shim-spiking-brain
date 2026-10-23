
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

from .neuroni import DT, TAU_M, TAU_S, C_M, V_SOGLIA, V_RESET, E_L, T_REF
from triton.runtime import driver

GRUPPO = 64


BLOCCO_NEURONI = {"xpu": 64, "cuda": 128}


SCALA_INTERI = 4096.0


PASSI_PER_LANCIO = int(os.environ.get("SHIM_PASSI", "1"))


BLOCCO_SINAPSI = {"xpu": 512, "cuda": 256}


_SEM = tl.constexpr(os.environ.get("SHIM_SEM", "relaxed"))


@triton.jit
def _in_interi(w, SCALA: tl.constexpr):
    x = w * SCALA
    return (x + tl.where(x >= 0, 0.5, -0.5)).to(tl.int32)


@triton.jit
def _aggiorna(V, I, refr, buf, lam, dc, spara_f, conta, passo, seme,
              n, D, w_est, dm, ds, kv, dt, v_soglia, v_reset, e_l, t_ref,
              BLOCCO: tl.constexpr, INTERI: tl.constexpr = False, SCALA: tl.constexpr = 4096.0):
    pid = tl.program_id(0)
    i = pid * BLOCCO + tl.arange(0, BLOCCO)
    m = i < n
    slot = passo % D
    cella = slot.to(tl.int64) * n + i
    grezzo = tl.load(buf + cella, mask=m, other=0)
    tl.debug_barrier()
    tl.store(buf + cella, grezzo * 0, mask=m)
    if INTERI:
        arrivo = grezzo.to(tl.float32) * (1.0 / SCALA)
    else:
        arrivo = grezzo


    la = tl.load(lam + i, mask=m, other=0.0)
    u = tl.rand(seme + passo, i)
    pk = tl.exp(-la)
    F = pk
    k = tl.zeros([BLOCCO], tl.float32)
    for j in tl.static_range(1, 17):
        oltre = u > F
        k += tl.where(oltre, 1.0, 0.0)
        pk = pk * la / j
        F = F + pk
    corr = tl.load(I + i, mask=m, other=0.0) * ds + arrivo + w_est * k
    tl.store(I + i, corr, mask=m)
    r = tl.load(refr + i, mask=m, other=0.0)
    v = tl.load(V + i, mask=m, other=0.0)
    attivo = r <= 0.0
    d = tl.load(dc + i, mask=m, other=0.0)
    v_nuovo = e_l + (v - e_l) * dm + (corr + d) * kv
    v = tl.where(attivo, v_nuovo, v)
    r = tl.where(attivo, r, r - dt)
    spara = attivo & (v >= v_soglia) & m
    v = tl.where(spara, v_reset, v)
    r = tl.where(spara, t_ref, r)
    tl.store(V + i, v, mask=m)
    tl.store(refr + i, r, mask=m)
    uno = tl.where(spara, 1, 0)
    tl.store(spara_f + i, uno.to(tl.int8), mask=m)
    c = tl.load(conta + i, mask=m, other=0)
    tl.store(conta + i, c + uno, mask=m)


@triton.jit
def _propaga(spara_f, indptr, indices, pesi, ritardi, buf, passo, n, D,
             GR: tl.constexpr, BLOCCO: tl.constexpr, INTERI: tl.constexpr = False, SCALA: tl.constexpr = 4096.0):
    pid = tl.program_id(0)
    jj = pid * GR + tl.arange(0, GR)
    if tl.max(tl.load(spara_f + jj, mask=jj < n, other=0).to(tl.int32), axis=0) == 0:
        return
    for q in range(GR):
        j = pid * GR + q
        if j < n:
            if tl.load(spara_f + j) != 0:
                ini = tl.load(indptr + j)
                fine = tl.load(indptr + j + 1)
                for off in range(ini, fine, BLOCCO):
                    o = off + tl.arange(0, BLOCCO)
                    mm = o < fine
                    post = tl.load(indices + o, mask=mm, other=0)
                    w = tl.load(pesi + o, mask=mm, other=0.0).to(tl.float32)
                    d = tl.load(ritardi + o, mask=mm, other=1).to(tl.int64)
                    cella = ((passo + d) % D) * n + post
                    if INTERI:
                        w = _in_interi(w, SCALA)
                    tl.atomic_add(buf + cella, w, mask=mm, sem=_SEM)


@triton.jit
def _passo(V, I, refr, buf, lam, dc, spara_f, conta, indptr, indices, pesi, ritardi, passo, seme,
           n, D, w_est, dm, ds, kv, dt, v_soglia, v_reset, e_l, t_ref,
           BLOCCO: tl.constexpr, BS: tl.constexpr, INTERI: tl.constexpr = False, SCALA: tl.constexpr = 4096.0,
           NPASSI: tl.constexpr = 1):
    pid = tl.program_id(0)
    i = pid * BLOCCO + tl.arange(0, BLOCCO)
    m = i < n
    la = tl.load(lam + i, mask=m, other=0.0)
    corr = tl.load(I + i, mask=m, other=0.0)
    r = tl.load(refr + i, mask=m, other=0.0)
    v = tl.load(V + i, mask=m, other=0.0)
    d = tl.load(dc + i, mask=m, other=0.0)
    c = tl.load(conta + i, mask=m, other=0)
    e0 = tl.exp(-la)


    q1 = e0 * la / 1
    F1 = e0 + q1
    q2 = q1 * la / 2
    F2 = F1 + q2
    q3 = q2 * la / 3
    F3 = F2 + q3
    for s in range(NPASSI):
        p = passo + s
        cella = (p % D).to(tl.int64) * n + i
        grezzo = tl.load(buf + cella, mask=m, other=0)
        tl.debug_barrier()
        tl.store(buf + cella, grezzo * 0, mask=m)
        if INTERI:
            arrivo = grezzo.to(tl.float32) * (1.0 / SCALA)
        else:
            arrivo = grezzo
        u = tl.rand(seme + p, i)
        k = (tl.where(u > e0, 1.0, 0.0) + tl.where(u > F1, 1.0, 0.0) + tl.where(u > F2, 1.0, 0.0)
             + tl.where(u > F3, 1.0, 0.0))


        pk = q3 * la / 4
        F = F3 + pk
        j = 5
        while (j < 17) & (tl.max(tl.where(u > F, 1, 0), axis=0) > 0):
            oltre = u > F
            k += tl.where(oltre, 1.0, 0.0)
            pk = pk * la / j
            F = F + pk
            j += 1
        corr = corr * ds + arrivo + w_est * k
        attivo = r <= 0.0
        v_nuovo = e_l + (v - e_l) * dm + (corr + d) * kv
        v = tl.where(attivo, v_nuovo, v)
        r = tl.where(attivo, r, r - dt)
        spara = attivo & (v >= v_soglia) & m
        v = tl.where(spara, v_reset, v)
        r = tl.where(spara, t_ref, r)
        uno = tl.where(spara, 1, 0)
        tl.store(spara_f + i, uno.to(tl.int8), mask=m)
        c = c + uno
        tl.debug_barrier()


        if tl.max(uno, axis=0) > 0:
            for q in range(BLOCCO):
                jn = pid * BLOCCO + q
                if jn < n:
                    if tl.load(spara_f + jn) != 0:
                        ini = tl.load(indptr + jn)
                        fine = tl.load(indptr + jn + 1)
                        for off in range(ini, fine, BS):
                            o = off + tl.arange(0, BS)
                            mm = o < fine
                            post = tl.load(indices + o, mask=mm, other=0)
                            w = tl.load(pesi + o, mask=mm, other=0.0).to(tl.float32)
                            dd = tl.load(ritardi + o, mask=mm, other=1).to(tl.int64)
                            cc = ((p + dd) % D) * n + post
                            if INTERI:
                                w = _in_interi(w, SCALA)
                            tl.atomic_add(buf + cc, w, mask=mm, sem=_SEM)
        tl.debug_barrier()
    tl.store(I + i, corr, mask=m)
    tl.store(V + i, v, mask=m)
    tl.store(refr + i, r, mask=m)
    tl.store(conta + i, c, mask=m)


LISTA = os.environ.get("SHIM_LISTA", "1") == "1"
BLOCCO_LISTA = 512


@triton.jit(do_not_specialize=["par", "passo", "seme"])
def _aggiorna_lista(V, I, refr, buf, lam, dc, spara_f, conta, lista, lista_p, contatore, par, passo, seme,
                    n, D, w_est, dm, ds, kv, dt, v_soglia, v_reset, e_l, t_ref, Ad, Bd, da, Dm, Vs, Tr, Kv,
                    BLOCCO: tl.constexpr, INTERI: tl.constexpr = False, SCALA: tl.constexpr = 4096.0,
                    NPASSI: tl.constexpr = 1, ADATTA: tl.constexpr = False, PARAMETRI: tl.constexpr = False):
    pid = tl.program_id(0)
    i = pid * BLOCCO + tl.arange(0, BLOCCO)
    m = i < n
    la = tl.load(lam + i, mask=m, other=0.0)
    corr = tl.load(I + i, mask=m, other=0.0)
    r = tl.load(refr + i, mask=m, other=0.0)
    v = tl.load(V + i, mask=m, other=0.0)
    d = tl.load(dc + i, mask=m, other=0.0)
    c = tl.load(conta + i, mask=m, other=0)
    if ADATTA:
        ad = tl.load(Ad + i, mask=m, other=0.0)
        bd = tl.load(Bd + i, mask=m, other=0.0)
    if PARAMETRI:
        dm_i = tl.load(Dm + i, mask=m, other=1.0)
        vs_i = tl.load(Vs + i, mask=m, other=0.0)
        tr_i = tl.load(Tr + i, mask=m, other=0.0)
        kv_i = tl.load(Kv + i, mask=m, other=0.0)
    else:
        dm_i = dm
        vs_i = v_soglia
        tr_i = t_ref
        kv_i = kv
    e0 = tl.exp(-la)
    q1 = e0 * la / 1
    F1 = e0 + q1
    q2 = q1 * la / 2
    F2 = F1 + q2
    q3 = q2 * la / 3
    F3 = F2 + q3
    zero = i * 0
    for s in range(NPASSI):
        p = passo + s
        cella = (p % D).to(tl.int64) * n + i
        grezzo = tl.load(buf + cella, mask=m, other=0)
        tl.debug_barrier()
        tl.store(buf + cella, grezzo * 0, mask=m)
        if INTERI:
            arrivo = grezzo.to(tl.float32) * (1.0 / SCALA)
        else:
            arrivo = grezzo
        u = tl.rand(seme + p, i)
        k = (tl.where(u > e0, 1.0, 0.0) + tl.where(u > F1, 1.0, 0.0) + tl.where(u > F2, 1.0, 0.0)
             + tl.where(u > F3, 1.0, 0.0))
        pk = q3 * la / 4
        F = F3 + pk
        j = 5
        while (j < 17) & (tl.max(tl.where(u > F, 1, 0), axis=0) > 0):
            oltre = u > F
            k += tl.where(oltre, 1.0, 0.0)
            pk = pk * la / j
            F = F + pk
            j += 1
        corr = corr * ds + arrivo + w_est * k
        attivo = r <= 0.0
        if ADATTA:
            ad = ad * da
        if PARAMETRI:
            if ADATTA:
                v_nuovo = e_l + (v - e_l) * dm_i + corr * kv + (d - ad) * kv_i
            else:
                v_nuovo = e_l + (v - e_l) * dm_i + corr * kv + d * kv_i
        elif ADATTA:
            v_nuovo = e_l + (v - e_l) * dm + (corr + d - ad) * kv
        else:
            v_nuovo = e_l + (v - e_l) * dm + (corr + d) * kv
        v = tl.where(attivo, v_nuovo, v)
        r = tl.where(attivo, r, r - dt)
        spara = attivo & (v >= vs_i) & m
        v = tl.where(spara, v_reset, v)
        r = tl.where(spara, tr_i, r)
        if ADATTA:
            ad = tl.where(spara, ad + bd, ad)
        uno = tl.where(spara, 1, 0)
        tl.store(spara_f + i, uno.to(tl.int8), mask=m)
        c = c + uno
        if tl.max(uno, axis=0) > 0:
            pos = tl.atomic_add(contatore + par + zero, uno, mask=spara)
            tl.store(lista + pos, i, mask=spara)
            tl.store(lista_p + pos, zero + p, mask=spara)
    if ADATTA:
        tl.store(Ad + i, ad, mask=m)
    tl.store(I + i, corr, mask=m)
    tl.store(V + i, v, mask=m)
    tl.store(refr + i, r, mask=m)
    tl.store(conta + i, c, mask=m)


@triton.jit(do_not_specialize=["par"])
def _spedisci_lista(lista, lista_p, contatore, par, indptr, indices, pesi, ritardi, buf, n, D,
                    BS: tl.constexpr, INTERI: tl.constexpr = False, SCALA: tl.constexpr = 4096.0):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    tot = tl.load(contatore + par)
    if pid == 0:
        tl.store(contatore + (1 - par), 0)
    for q in range(pid, tot, G):
        jn = tl.load(lista + q)
        pq = tl.load(lista_p + q).to(tl.int64)
        ini = tl.load(indptr + jn)
        fine = tl.load(indptr + jn + 1)
        for off in range(ini, fine, BS):
            o = off + tl.arange(0, BS)
            mm = o < fine
            post = tl.load(indices + o, mask=mm, other=0)
            w = tl.load(pesi + o, mask=mm, other=0.0).to(tl.float32)
            dd = tl.load(ritardi + o, mask=mm, other=1).to(tl.int64)
            cc = ((pq + dd) % D) * n + post
            if INTERI:
                w = _in_interi(w, SCALA)
            tl.atomic_add(buf + cc, w, mask=mm, sem=_SEM)


class MotoreTriton:
    def __init__(self, col, dispositivo: str = None, pesi_mezzi: bool = False):
        if dispositivo is None:
            dispositivo = "cuda" if torch.cuda.is_available() else "xpu"
        self.dev = torch.device(dispositivo)
        f = dict(device=self.dev)
        self.col = col
        self.n = col.n
        self.npassi = max(1, PASSI_PER_LANCIO)
        self.D = int(col.max_ritardo) + 1 + (self.npassi - 1)

        def sposta(x, tipo):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.dev, dtype=tipo)
            return torch.as_tensor(np.asarray(x), device=self.dev).to(tipo)
        self.indptr = sposta(col.indptr, torch.int64)
        self.indices = sposta(col.indices, torch.int32)
        self.pesi = sposta(col.pesi, torch.float16 if pesi_mezzi else torch.float32)
        self.ritardi = sposta(col.ritardi, torch.uint8)
        if self.npassi > 1:
            self.ritardi.clamp_(min=self.npassi)
        self.interi = os.environ.get("SHIM_INTERI", "") == "1"
        self.buf = torch.zeros(self.D * self.n, dtype=torch.int32 if self.interi else torch.float32, **f)
        self.V = torch.as_tensor(col.V0, dtype=torch.float32, **f).clone()
        self.I = torch.zeros(self.n, dtype=torch.float32, **f)
        self.refr = torch.zeros(self.n, dtype=torch.float32, **f)
        self.lam = torch.as_tensor(col.lam_est, dtype=torch.float32, **f)
        self.dc = torch.as_tensor(np.asarray(col.dc, np.float32), **f)
        self.spara = torch.zeros(self.n, dtype=torch.int8, **f)
        self.conta = torch.zeros(self.n, dtype=torch.int32, **f)
        self.passo = 0

        self.adatta = False
        self.Ad = self.Bd = torch.zeros(1, dtype=torch.float32, **f)
        self.da = 1.0
        self.parametri = False
        self.Dm = self.Vs = self.Tr = self.Kv = torch.zeros(1, dtype=torch.float32, **f)

        self.fuso = self.dev.type == "xpu"
        self.c = dict(n=self.n, D=self.D, w_est=float(col.w_est),
                      dm=math.exp(-DT / TAU_M), ds=math.exp(-DT / TAU_S), kv=DT / C_M, dt=DT,
                      v_soglia=V_SOGLIA, v_reset=V_RESET, e_l=E_L, t_ref=T_REF)
        self.bn = BLOCCO_NEURONI.get(self.dev.type, 64)
        self.bs = BLOCCO_SINAPSI.get(self.dev.type, 128)
        self.griglia_n = (triton.cdiv(self.n, self.bn),)
        self.griglia_p = (triton.cdiv(self.n, GRUPPO),)
        mem = sum(t.numel() * t.element_size() for t in (self.indices, self.pesi, self.ritardi, self.buf))
        print("[triton] %s: %d neuroni, %d sinapsi, %.2f GB sulla GPU" % (
            self.dev, self.n, self.indices.numel(), mem / 1e9), flush=True)

    def passi(self, n: int, seme: int = 7):
        if getattr(self, "lista", LISTA) and getattr(self, "fuso", False):
            if getattr(self, "_lista", None) is None:
                self._lista = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
                self._lista_p = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
                self._contatore = torch.zeros(2, dtype=torch.int32, device=self.dev)
                self._par = 0
                self.griglia_lista = (max(256, self.n // 256),)
                self.griglia_agg = (triton.cdiv(self.n, BLOCCO_LISTA),)
            c = self.c
            if getattr(self, "_lanci", None) is None or self._lanci[0] != (self.npassi, self.bs, self.interi, self.adatta,
                                                                           self.parametri):

                self._lanci = ((self.npassi, self.bs, self.interi, self.adatta, self.parametri),
                               Lancio(_aggiorna_lista, self.griglia_agg, BLOCCO=BLOCCO_LISTA, INTERI=self.interi,
                                      SCALA=SCALA_INTERI, NPASSI=self.npassi, ADATTA=self.adatta,
                                      PARAMETRI=self.parametri),
                               Lancio(_spedisci_lista, self.griglia_lista, BS=self.bs, INTERI=self.interi,
                                      SCALA=SCALA_INTERI))
            agg, sped = self._lanci[1], self._lanci[2]
            for _ in range(n // self.npassi):
                agg(self.V, self.I, self.refr, self.buf, self.lam, self.dc, self.spara, self.conta, self._lista,
                    self._lista_p, self._contatore, self._par, self.passo, seme, c["n"], c["D"], c["w_est"], c["dm"],
                    c["ds"], c["kv"], c["dt"], c["v_soglia"], c["v_reset"], c["e_l"], c["t_ref"],
                    self.Ad, self.Bd, self.da, self.Dm, self.Vs, self.Tr, self.Kv)
                sped(self._lista, self._lista_p, self._contatore, self._par, self.indptr, self.indices, self.pesi,
                     self.ritardi, self.buf, c["n"], c["D"])
                self._par = 1 - self._par
                self.passo += self.npassi
            n = n % self.npassi
        elif getattr(self, "fuso", False) and self.npassi > 1:
            c = self.c
            for _ in range(n // self.npassi):
                _passo[self.griglia_n](self.V, self.I, self.refr, self.buf, self.lam, self.dc, self.spara, self.conta,
                                       self.indptr, self.indices, self.pesi, self.ritardi, self.passo, seme,
                                       c["n"], c["D"], c["w_est"], c["dm"], c["ds"], c["kv"], c["dt"],
                                       c["v_soglia"], c["v_reset"], c["e_l"], c["t_ref"], BLOCCO=self.bn, BS=self.bs,
                                       INTERI=self.interi, SCALA=SCALA_INTERI, NPASSI=self.npassi)
                self.passo += self.npassi
            n = n % self.npassi
        for _ in range(n):
            self.passo_uno(seme)

    def passo_uno(self, seme: int = 7):
        c = self.c
        if getattr(self, "fuso", False):
            _passo[self.griglia_n](self.V, self.I, self.refr, self.buf, self.lam, self.dc, self.spara, self.conta,
                                   self.indptr, self.indices, self.pesi, self.ritardi, self.passo, seme,
                                   c["n"], c["D"], c["w_est"], c["dm"], c["ds"], c["kv"], c["dt"],
                                   c["v_soglia"], c["v_reset"], c["e_l"], c["t_ref"], BLOCCO=self.bn, BS=self.bs,
                                   INTERI=self.interi, SCALA=SCALA_INTERI)
            self.passo += 1
            return
        _aggiorna[self.griglia_n](self.V, self.I, self.refr, self.buf, self.lam, self.dc,
                                  self.spara, self.conta, self.passo, seme,
                                  c["n"], c["D"], c["w_est"], c["dm"], c["ds"], c["kv"], c["dt"],
                                  c["v_soglia"], c["v_reset"], c["e_l"], c["t_ref"], BLOCCO=self.bn,
                                  INTERI=self.interi, SCALA=SCALA_INTERI)
        if getattr(self, "sincro_ogni_kernel", False):
            self.sincro()
        _propaga[self.griglia_p](self.spara, self.indptr, self.indices, self.pesi, self.ritardi,
                                 self.buf, self.passo, c["n"], c["D"], GR=GRUPPO, BLOCCO=self.bs,
                                 INTERI=self.interi, SCALA=SCALA_INTERI)
        if getattr(self, "sincro_ogni_kernel", False):
            self.sincro()
        self.passo += 1

    def sincro(self):
        (torch.cuda.synchronize if self.dev.type == "cuda" else torch.xpu.synchronize)()


class Lancio:

    def __init__(self, fn, griglia, **costanti):
        self.fn, self.costanti = fn, costanti
        g = tuple(griglia) + (1, 1)
        self.g = (int(g[0]), int(g[1]), int(g[2]))
        self.k = None

    def __call__(self, *args):
        if self.k is None:
            self.k = self.fn[self.g](*args, **self.costanti)
            nomi = self.fn.arg_names
            self.coda = tuple(self.costanti[n] for n in nomi[len(args):] if n in self.costanti)
            dev = driver.active.get_current_device()
            self.stream = driver.active.get_current_stream(dev)
            return
        k = self.k
        k.run(self.g[0], self.g[1], self.g[2], self.stream, k.function, k.packed_metadata, None, None, None,
              *args, *self.coda)
