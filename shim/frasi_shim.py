from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

QUI = (Path(__file__).resolve().parents[1] / "stato")
FILE = QUI / ("shim_frasi_2000.json" if os.environ.get("SHIM_CONOSCENZA") == "2000" else "shim_frasi.json")
PICCOLE = set("""il lo la i gli le un uno una l' un' di a da in con su per tra fra e ed o che non è sono ha hanno si del della dello dei delle
degli dell' al alla allo ai alle agli all' dal dalla dallo dai dalle dagli dall' nel nella nello nei nelle negli nell' sul sulla sullo sui
sulle sugli sull' come anche molto""".split())
MAX_IN_MEZZO = 3
ARTICOLI = ("il", "lo", "la", "i", "gli", "le", "l'", "un", "uno", "una", "un'")
MIN_COPPIA, QUOTA_COPPIA = 3, 0.3
ART_DENTRO = {p + a: art for art, suff in (("il", "l"), ("lo", "llo"), ("la", "lla"), ("i", "i"), ("gli", "gli"), ("le", "lle"), ("l'", "ll'"))
              for p, a in (("de", suff), ("a", suff), ("da", suff), ("ne", suff), ("su", suff))}
ART_DENTRO.update({"del": "il", "al": "il", "dal": "il", "nel": "il", "sul": "il", "dei": "i", "ai": "i", "dai": "i", "nei": "i", "sui": "i"})
QUOTA_NOME = 0.35
PAROLA = re.compile(r"[a-zàèéìòóù]+'?", re.IGNORECASE)


def _frasi_testo():
    from .conoscenza_testi import TESTI
    for f in sorted(TESTI.glob("*.txt")):
        txt = re.sub(r"==+[^=]+==+", " . ", f.read_text(encoding="utf-8"))
        for fr in re.split(r"[.!?;:\n()\[\]\"«»]+", txt):
            yield [t.lower() for t in PAROLA.findall(fr)]


def costruisci():
    from .parole_shim import _vocabolario
    piene = set(_vocabolario())
    fra = defaultdict(Counter)
    prima = defaultdict(Counter)
    for toks in _frasi_testo():
        ult, mezzo = None, []
        for t in toks:
            if t in piene:
                prima[t][" ".join(mezzo[-2:])] += 1
                if ult is not None and len(mezzo) <= MAX_IN_MEZZO:
                    fra[ult + " " + t][" ".join(mezzo)] += 1
                ult, mezzo = t, []
            elif t in PICCOLE:
                mezzo.append(t)
            else:
                ult, mezzo = None, []
    tab = {"fra": {}, "articolo": {}}
    for k, c in fra.items():
        tot = sum(c.values())
        w, n = c.most_common(1)[0]
        if n >= MIN_COPPIA and n / tot >= QUOTA_COPPIA:
            tab["fra"][k] = w
    for k, c in prima.items():
        tot = sum(c.values())
        art = Counter()
        for x, n in c.items():
            u = x.split()[-1] if x else ""
            if u in ARTICOLI:
                art[u] += n
            elif u in ART_DENTRO:
                art[ART_DENTRO[u]] += n
        if art and sum(art.values()) / tot >= QUOTA_NOME:
            tab["articolo"][k] = art.most_common(1)[0][0]
    dopo_verbo = defaultdict(Counter)
    for toks in _frasi_testo():
        ult, mezzo = None, []
        for t in toks:
            if t in piene:
                if ult is not None and ult not in tab["articolo"] and t in tab["articolo"] and len(mezzo) <= MAX_IN_MEZZO:
                    dopo_verbo[t][" ".join(mezzo)] += 1
                ult, mezzo = t, []
            elif t in PICCOLE:
                mezzo.append(t)
            else:
                ult, mezzo = None, []
    tab["dopo_verbo"] = {}
    for k, c in dopo_verbo.items():
        w, n = c.most_common(1)[0]
        if n >= MIN_COPPIA and n / sum(c.values()) >= QUOTA_COPPIA and w:
            tab["dopo_verbo"][k] = w
    FILE.write_text(json.dumps(tab, ensure_ascii=False), encoding="utf-8")

    import numpy as np
    from .corpus_shim import FRASI
    from .conoscenza_testi import frasi_dei_testi
    voc = _vocabolario()
    ix = {w: i for i, w in enumerate(voc)}
    F = np.zeros((len(voc), len(voc)), np.float32)
    scuola = FRASI if isinstance(FRASI, list) else [l.split() for l in FRASI.strip().splitlines() if l.strip()]
    for peso, fonte in ((3.0, scuola), (1.0, frasi_dei_testi())):
        for fr in fonte:
            ids = [ix[w] for w in fr if w in ix]
            for a, b in zip(ids, ids[1:]):
                F[a, b] += peso
    if os.environ.get("SHIM_CONOSCENZA") != "2000":
        np.save(QUI / "shim_ordine.npy", F)
    print("coppie: %d, nomi con articolo: %d -> %s" % (len(tab["fra"]), len(tab["articolo"]), FILE), flush=True)
    return tab


SCUOLA_FRA = {"beve acqua": "l'", "beve fiume": "dal", "beve latte": "il", "canta albero": "sull'", "canta sole": "al", "corre casa": "a",
              "corre montagna": "in", "dorme casa": "in", "dorme letto": "nel", "dorme sole": "al", "gioca palla": "con la",
              "mangia carne": "la", "mangia fiore": "il", "mangia formaggio": "il", "mangia mela": "la", "mangia pane": "il",
              "mangia pesce": "il", "mangia topo": "il", "miagola mamma": "alla", "nuota acqua": "nell'", "nuota fiume": "nel",
              "nuota mare": "nel", "vola cielo": "nel", "vola nuvola": "sulla", "abbaia gatto": "al", "brilla cielo": "nel",
              "corre prato": "nel", "corre giardino": "nel", "cade albero": "dall'", "cresce albero": "sull'"}

from .corpus_shim import CATEGORIE as _CAT
VERBI = set(_CAT["azioni"])

_N2000 = None


ARTICOLO_SCUOLA = {"pianeta": "il", "volpe": "la", "radice": "la", "valle": "la", "ape": "l'", "elefante": "l'", "serpente": "il", "pesce": "il",
                   "dente": "il", "squalo": "lo", "scheletro": "lo", "stomaco": "lo", "schermo": "lo", "zaino": "lo", "zoccolo": "lo",
                   "scoiattolo": "lo", "sasso": "il", "satellite": "il", "universo": "l'", "uccello": "l'", "insetto": "l'", "anfibio": "l'",
                   "orso": "l'", "asino": "l'", "ossigeno": "l'", "orecchio": "l'", "algoritmo": "l'", "autobus": "l'", "arcobaleno": "l'",
                   "oceano": "l'", "inverno": "l'", "autunno": "l'", "alunno": "l'", "artiglio": "l'", "animale": "l'", "ala": "l'",
                   "arancia": "l'", "aquila": "l'", "atmosfera": "l'", "erba": "l'", "uva": "l'", "orbita": "l'", "istruzione": "l'",
                   "insalata": "l'", "elettricità": "l'", "magnete": "il", "nervo": "il", "fiore": "il", "mare": "il", "sole": "il",
                   "latte": "il", "pane": "il", "cane": "il", "leone": "il", "corno": "il", "limone": "il", "pomodoro": "il", "vapore": "il",
                   "motore": "il", "confine": "il", "seme": "il", "fulmine": "il", "verme": "il", "nazione": "la", "variabile": "la",
                   "neve": "la", "notte": "la", "carne": "la", "nave": "la", "chiave": "la", "gente": "la", "classe": "la", "lezione": "la",
                   "stagione": "la", "primavera": "la", "gas": "il", "software": "il", "pixel": "il"}


def _NOMI_2000():
    global _N2000
    if _N2000 is None:
        _N2000 = set()
        fc = QUI / "shim_classi_2000.json"
        if os.environ.get("SHIM_CONOSCENZA") == "2000" and fc.exists():
            d = json.loads(fc.read_text(encoding="utf-8"))
            _N2000 = {w for w, c in zip(d["parole"], d["classi"]) if c == "nome"}
            VERBI.update(w for w, c in zip(d["parole"], d["classi"]) if c == "verbo")
    return _N2000


_TAB = None


def tabella():
    global _TAB
    if _TAB is None:
        _TAB = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {"fra": {}, "articolo": {}}
    return _TAB


def componi(parole):
    t = tabella()
    _NOMI_2000()
    out = []
    for i, w in enumerate(parole):
        c = (SCUOLA_FRA.get(parole[i - 1] + " " + w) or t["fra"].get(parole[i - 1] + " " + w)) if i > 0 else None
        if c is None and i > 0 and parole[i - 1] in VERBI and w in VERBI:
            c = "e"
        elif c is None and i > 0 and parole[i - 1] in t["articolo"] and w in t["articolo"]:
            c = "e " + t["articolo"][w]
        if c is not None:
            if c:
                out.append(c)
        elif i > 0 and parole[i - 1] not in t["articolo"] and w in t.get("dopo_verbo", {}):
            out.append(t["dopo_verbo"][w])
        elif w in ARTICOLO_SCUOLA:
            out.append(ARTICOLO_SCUOLA[w])
        elif w in t["articolo"]:
            out.append(t["articolo"][w])
        elif w in _NOMI_2000():
            out.append({"a": "la", "e": "le" if w.endswith("ie") else "il", "i": "i", "o": "il"}.get(w[-1], "il") if w[0] not in "aeiouàèéìòù" else "l'")
        out.append(w)
    s = " ".join(out).replace("' ", "'")
    return s[:1].upper() + s[1:] + "." if s else s


def componi_legate(parole):
    if not parole:
        return ""
    tema = componi(parole[:1]).rstrip(".")
    resto = [componi([w]).rstrip(".") for w in parole[1:]]
    resto = [r[:1].lower() + r[1:] for r in resto]
    if not resto:
        return tema + "."
    return tema + "? " + (" e ".join(resto))[:1].upper() + (" e ".join(resto))[1:] + "."


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "costruisci":
        costruisci()
    else:
        print(componi(sys.argv[1:]))
