from __future__ import annotations

import bz2
import json
import multiprocessing as mp
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

QUI = (Path(__file__).resolve().parents[1] / "stato") / "shim_wiki"
DUMP = QUI / "itwiki.xml.bz2"
FINESTRA = 4

try:
    from .conoscenza_testi import STOP
except Exception:
    STOP = set()


STOP |= set("""dell all nell dall sull quell un' dall' nbsp amp quot lt gt mathbb mathrm mathcal frac text style align class width height
the of and in to for with by from on at is as an or is was are this that which also other not ibid cfr pag isbn doi pdf url http https www
com org html htm jpg png svg thumb miniatura sinistra destra centro left right upright col cite web titolo accesso archiviato wikipedia
nome cognome autore anno data pubblicazione editore lingua""".split())

PAROLA = re.compile(r"[a-zàèéìòóù]+", re.IGNORECASE)
_MODELLO = re.compile(r"\{\{[^{}]*\}\}")
_RIF = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.DOTALL)
_HTML = re.compile(r"<[^>]+>")
_COMM = re.compile(r"<!--.*?-->", re.DOTALL)
_TAB = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_FILE = re.compile(r"\[\[(?:File|Immagine|Categoria|Image|Category|Template|Wikipedia|Aiuto|Portale):[^\]]*(?:\[\[[^\]]*\]\][^\]]*)*\]\]", re.IGNORECASE)
_LINK2 = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_LINK1 = re.compile(r"\[\[([^\]]*)\]\]")
_ESTERNO = re.compile(r"\[https?://[^\s\]]*\s*([^\]]*)\]|https?://\S+")
_TITOLI = re.compile(r"^=+.*?=+\s*$", re.MULTILINE)
_APICI = re.compile(r"'{2,}")


def pulisci(testo: str) -> str:
    t = _COMM.sub(" ", testo)
    t = _RIF.sub(" ", t)
    t = _TAB.sub(" ", t)
    for _ in range(6):
        nuovo = _MODELLO.sub(" ", t)
        if nuovo == t:
            break
        t = nuovo
    t = _FILE.sub(" ", t)
    t = _LINK2.sub(r"\1", t)
    t = _LINK1.sub(r"\1", t)
    t = _ESTERNO.sub(r"\1", t)
    t = _HTML.sub(" ", t)
    t = _TITOLI.sub(" . ", t)
    t = _APICI.sub("", t)
    return t


def frasi(testo: str):
    for fr in re.split(r"[.!?;\n:]+", pulisci(testo)):
        ps = [p.lower() for p in PAROLA.findall(fr)]
        ps = [p for p in ps if len(p) >= 3 and p not in STOP]
        if len(ps) >= 2:
            yield ps


def pagine(percorso=DUMP, max_pagine=None):
    n = 0
    try:
        with bz2.BZ2File(str(percorso)) as f:
            titolo, ns = None, None
            for evento, el in ET.iterparse(f, events=("end",)):
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "title":
                    titolo = el.text
                elif tag == "ns":
                    ns = el.text
                elif tag == "text":
                    testo = el.text or ""
                    if ns == "0" and titolo and not testo.lstrip().upper().startswith(("#RINVIA", "#REDIRECT")) and len(testo) > 800:
                        n += 1
                        yield titolo, testo
                        if max_pagine and n >= max_pagine:
                            return
                elif tag == "page":
                    el.clear()
    except (EOFError, OSError, ET.ParseError):
        return


def _conta(testi):
    c = Counter()
    for t in testi:
        for ps in frasi(t):
            c.update(ps)
    return c


_VOC = None


def _init(voc):
    global _VOC
    _VOC = voc


def _coocc(testi):
    chiavi = Counter()
    V = len(_VOC)
    for t in testi:
        for ps in frasi(t):
            ids = [_VOC[p] for p in ps if p in _VOC]
            for i, a in enumerate(ids):
                for d in range(1, FINESTRA + 1):
                    if i + d < len(ids):
                        b = ids[i + d]
                        if a != b:
                            chiavi[a * V + b] += 1.0 / d
    return list(chiavi.items())


def _gruppi(gen, dim=200):
    buf = []
    for _, t in gen:
        buf.append(t)
        if len(buf) >= dim:
            yield buf
            buf = []
    if buf:
        yield buf


def vocabolario(max_pagine=None, n_parole=20000):
    QUI.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    c = Counter()
    pag = 0
    with mp.Pool(max(2, (os.cpu_count() or 4) - 2)) as pool:
        for k, cc in enumerate(pool.imap_unordered(_conta, _gruppi(pagine(DUMP, max_pagine)), chunksize=1)):
            c.update(cc)
            pag += 200
            if k % 25 == 0:
                print("  %d pagine circa, %d parole diverse, %.0f s" % (pag, len(c), time.time() - t0), flush=True)
    voc = [w for w, _ in c.most_common(n_parole)]
    (QUI / "vocabolario_wiki.json").write_text(json.dumps({"parole": voc, "frequenze": [c[w] for w in voc], "pagine": pag}, ensure_ascii=False), encoding="utf-8")
    print("vocabolario: %d parole su %d diverse (%d pagine circa, %.0f s)" % (len(voc), len(c), pag, time.time() - t0), flush=True)


def associazioni(max_pagine=None):
    import scipy.sparse as sp
    d = json.loads((QUI / "vocabolario_wiki.json").read_text(encoding="utf-8"))
    voc = {w: i for i, w in enumerate(d["parole"])}
    V = len(voc)
    t0 = time.time()
    righe, colonne, valori = [], [], []
    M = sp.csr_matrix((V, V), dtype=np.float32)
    pag = 0
    with mp.Pool(max(2, (os.cpu_count() or 4) - 2), initializer=_init, initargs=(voc,)) as pool:
        for k, parziale in enumerate(pool.imap_unordered(_coocc, _gruppi(pagine(DUMP, max_pagine)), chunksize=1)):
            if parziale:
                kk = np.fromiter((x[0] for x in parziale), dtype=np.int64, count=len(parziale))
                vv = np.fromiter((x[1] for x in parziale), dtype=np.float32, count=len(parziale))
                righe.append(kk // V)
                colonne.append(kk % V)
                valori.append(vv)
            pag += 200
            if len(righe) >= 200:
                M = M + sp.coo_matrix((np.concatenate(valori), (np.concatenate(righe), np.concatenate(colonne))), shape=(V, V), dtype=np.float32).tocsr()
                righe, colonne, valori = [], [], []
            if k % 25 == 0:
                print("  %d pagine circa, %d associazioni, %.0f s" % (pag, M.nnz, time.time() - t0), flush=True)
    if righe:
        M = M + sp.coo_matrix((np.concatenate(valori), (np.concatenate(righe), np.concatenate(colonne))), shape=(V, V), dtype=np.float32).tocsr()
    N = (M + 0.5 * M.T).tocsr()
    sp.save_npz(QUI / "associazioni_wiki.npz", N)
    np.save(QUI / "frequenze_wiki.npy", np.asarray(M.sum(1)).ravel() + np.asarray(M.sum(0)).ravel())
    print("associazioni: %d x %d, %d non nulle, %d pagine circa, %.0f s" % (V, V, N.nnz, pag, time.time() - t0), flush=True)


def unisci():
    import scipy.sparse as sp
    from .associazioni_grandi import AssociazioniGrandi
    from .corpus_shim import FRASI, PAROLE as BASE
    d = json.loads((QUI / "vocabolario_wiki.json").read_text(encoding="utf-8"))
    wiki = d["parole"]
    N = sp.load_npz(QUI / "associazioni_wiki.npz").tocoo()
    freq = np.load(QUI / "frequenze_wiki.npy")
    ins = set(BASE)
    comb = list(BASE) + [w for w in wiki if w not in ins]
    pos = {w: i for i, w in enumerate(comb)}
    mappa = np.array([pos[w] for w in wiki], np.int64)
    V = len(comb)
    Nn = sp.csr_matrix((N.data, (mappa[N.row], mappa[N.col])), shape=(V, V), dtype=np.float32)
    nk = np.zeros(V, np.float32)
    nk[mappa] = freq
    quad = QUI.parent
    (quad / "shim_vocab_wiki.json").write_text(json.dumps(comb, ensure_ascii=False), encoding="utf-8")
    np.savez(quad / "shim_assoc_wiki.npz", data=Nn.data, indices=Nn.indices, indptr=Nn.indptr, shape=np.array(Nn.shape), nk=nk)
    A = AssociazioniGrandi(comb, Nn, nk=nk, file_extra=quad / "shim_fatti_wiki.npz")
    for _ in range(3):
        for f in FRASI:
            A.impara(f)
    A.salva()
    print("unito: %d parole (%d di base + %d dalla Wikipedia), %d associazioni, %d fatti di scuola" % (V, len(BASE), V - len(BASE), Nn.nnz, sum(len(r) for r in A.extra.values())), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    a = sys.argv[1:]
    mx = int(a[a.index("--max-pagine") + 1]) if "--max-pagine" in a else None
    if a and a[0] == "vocabolario":
        vocabolario(mx, int(a[a.index("--parole") + 1]) if "--parole" in a else 20000)
    elif a and a[0] == "associazioni":
        associazioni(mx)
    elif a and a[0] == "unisci":
        unisci()
    else:
        print(__doc__)
