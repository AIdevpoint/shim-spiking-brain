from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from shim.motore import MotoreTriton, _spedisci_lista, SCALA_INTERI

BLOCCO_SHIM = int(os.environ.get("SHIM_BLOCCO", "512"))
RAND4 = os.environ.get("SHIM_RAND4", "0") == "1"
WARP_SHIM = int(os.environ.get("SHIM_WARP", "4"))
STATICO = os.environ.get("SHIM_STATICO", "1") == "1"


@triton.jit
def _u(seme, p, i, n, s: tl.constexpr, passo, r0, r1, r2, r3, r4, r5, r6, r7, RAND4: tl.constexpr):
    if RAND4:
        if s == 0:
            u = r0
        elif s == 1:
            u = r1
        elif s == 2:
            u = r2
        elif s == 3:
            u = r3
        elif s == 4:
            u = r4
        elif s == 5:
            u = r5
        elif s == 6:
            u = r6
        else:
            u = r7
    else:
        u = tl.rand(seme + p, i)
    return u


@triton.jit(do_not_specialize=["par", "passo", "seme"])
def _aggiorna_shim(V, I, refr, buf, lam, dc, spara_f, conta, lista, lista_p, contatore, par, passo, seme,
                  n, D, w_est, dm, ds, kv, dt, v_soglia, v_reset, e_l, t_ref,
                  BLOCCO: tl.constexpr, INTERI: tl.constexpr = False, SCALA: tl.constexpr = 4096.0,
                  NPASSI: tl.constexpr = 1, STATICO: tl.constexpr = True, RAND4: tl.constexpr = False,
                  BANDIERINE: tl.constexpr = True):
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
    if STATICO:
        pk = q3 * la / 4
        F4 = F3 + pk
        pk = pk * la / 5
        F5 = F4 + pk
        pk = pk * la / 6
        F6 = F5 + pk
        pk = pk * la / 7
        F7 = F6 + pk
        pk = pk * la / 8
        F8 = F7 + pk
        pk = pk * la / 9
        F9 = F8 + pk
        pk = pk * la / 10
        F10 = F9 + pk
        pk = pk * la / 11
        F11 = F10 + pk
        pk = pk * la / 12
        F12 = F11 + pk
        pk = pk * la / 13
        F13 = F12 + pk
        pk = pk * la / 14
        F14 = F13 + pk
        pk = pk * la / 15
        F15 = F14 + pk
    if RAND4:
        r0, r1, r2, r3 = tl.rand4x(seme + passo, i)
        r4, r5, r6, r7 = tl.rand4x(seme + passo, i + n)
    else:
        r0 = la
        r1 = la
        r2 = la
        r3 = la
        r4 = la
        r5 = la
        r6 = la
        r7 = la
    zero = i * 0
    for s in tl.static_range(NPASSI):
        p = passo + s
        cella = (p % D).to(tl.int64) * n + i
        grezzo = tl.load(buf + cella, mask=m, other=0)
        tl.debug_barrier()
        tl.store(buf + cella, grezzo * 0, mask=m)
        if INTERI:
            arrivo = grezzo.to(tl.float32) * (1.0 / SCALA)
        else:
            arrivo = grezzo
        u = _u(seme, p, i, n, s, passo, r0, r1, r2, r3, r4, r5, r6, r7, RAND4)
        k = (tl.where(u > e0, 1.0, 0.0) + tl.where(u > F1, 1.0, 0.0) + tl.where(u > F2, 1.0, 0.0)
             + tl.where(u > F3, 1.0, 0.0))
        if STATICO:
            k += (tl.where(u > F4, 1.0, 0.0) + tl.where(u > F5, 1.0, 0.0) + tl.where(u > F6, 1.0, 0.0)
                  + tl.where(u > F7, 1.0, 0.0) + tl.where(u > F8, 1.0, 0.0) + tl.where(u > F9, 1.0, 0.0)
                  + tl.where(u > F10, 1.0, 0.0) + tl.where(u > F11, 1.0, 0.0) + tl.where(u > F12, 1.0, 0.0)
                  + tl.where(u > F13, 1.0, 0.0) + tl.where(u > F14, 1.0, 0.0) + tl.where(u > F15, 1.0, 0.0))
        else:
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
        if BANDIERINE:
            tl.store(spara_f + i, uno.to(tl.int8), mask=m)
        else:
            tl.store(spara_f + i, uno.to(tl.int8), mask=spara)
        c = c + uno
        if tl.max(uno, axis=0) > 0:
            pos = tl.atomic_add(contatore + par + zero, uno, mask=spara)
            tl.store(lista + pos, i, mask=spara)
            tl.store(lista_p + pos, zero + p, mask=spara)
    tl.store(I + i, corr, mask=m)
    tl.store(V + i, v, mask=m)
    tl.store(refr + i, r, mask=m)
    tl.store(conta + i, c, mask=m)


class MotoreSHIM(MotoreTriton):

    def __init__(self, col, statico=None, rand4=None, blocco=None, bandierine=False, warp=None, **kw):
        super().__init__(col, **kw)
        assert not self.adatta and not self.parametri, "SHIM non usa SHIM_TIPI / SHIM_TIPI_NEURONI"
        assert self.npassi <= 8, "RAND4 ha 8 numeri per motore"
        self.statico = STATICO if statico is None else statico
        self.rand4 = RAND4 if rand4 is None else rand4
        self.blocco = BLOCCO_SHIM if blocco is None else blocco
        self.bandierine = bandierine
        self.warp = WARP_SHIM if warp is None else warp

    def passi(self, n: int, seme: int = 7):
        if getattr(self, "_lista", None) is None:
            self._lista = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
            self._lista_p = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
            self._contatore = torch.zeros(2, dtype=torch.int32, device=self.dev)
            self._par = 0
            self.griglia_lista = (max(256, self.n // 256),)
        chiave = (self.npassi, self.bs, self.interi, self.statico, self.rand4, self.blocco, self.bandierine, self.warp)
        if getattr(self, "_lanci_shim", None) is None or self._lanci_shim[0] != chiave:
            from shim.motore import Lancio
            self._lanci_shim = (chiave,
                               Lancio(_aggiorna_shim, (triton.cdiv(self.n, self.blocco),), BLOCCO=self.blocco,
                                      INTERI=self.interi, SCALA=SCALA_INTERI, NPASSI=self.npassi,
                                      STATICO=self.statico, RAND4=self.rand4, BANDIERINE=self.bandierine,
                                      num_warps=self.warp),
                               Lancio(_spedisci_lista, self.griglia_lista, BS=self.bs, INTERI=self.interi,
                                      SCALA=SCALA_INTERI))
            self._lanci = (None, self._lanci_shim[1], self._lanci_shim[2])
        agg, sped = self._lanci_shim[1], self._lanci_shim[2]
        c = self.c
        for _ in range(n // self.npassi):
            agg(self.V, self.I, self.refr, self.buf, self.lam, self.dc, self.spara, self.conta, self._lista,
                self._lista_p, self._contatore, self._par, self.passo, seme, c["n"], c["D"], c["w_est"], c["dm"],
                c["ds"], c["kv"], c["dt"], c["v_soglia"], c["v_reset"], c["e_l"], c["t_ref"])
            sped(self._lista, self._lista_p, self._contatore, self._par, self.indptr, self.indices, self.pesi,
                 self.ritardi, self.buf, c["n"], c["D"])
            self._par = 1 - self._par
            self.passo += self.npassi
        assert n % self.npassi == 0, "SHIM fa passi a gruppi interi"
