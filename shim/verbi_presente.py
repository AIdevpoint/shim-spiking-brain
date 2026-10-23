from __future__ import annotations

IRREGOLARI = {
    "disse": "dice", "fece": "fa", "prese": "prende", "vide": "vede", "rispose": "risponde", "mise": "mette", "venne": "viene", "ebbe": "ha",
    "fu": "è", "volle": "vuole", "seppe": "sa", "stette": "sta", "diede": "dà", "dette": "dà", "corse": "corre", "scrisse": "scrive",
    "lesse": "legge", "chiese": "chiede", "rimase": "rimane", "giunse": "giunge", "cadde": "cade", "nacque": "nasce", "morì": "muore",
    "uscì": "esce", "aprì": "apre", "salì": "sale", "apparve": "appare", "decise": "decide", "pianse": "piange", "rise": "ride",
    "spinse": "spinge", "andò": "va", "bevve": "beve", "tenne": "tiene", "conobbe": "conosce", "ruppe": "rompe", "scese": "scende",
    "accese": "accende", "chiuse": "chiude", "mosse": "muove", "nascose": "nasconde", "parve": "pare", "pose": "pone", "trasse": "trae",
    "vinse": "vince", "visse": "vive", "colse": "coglie", "sciolse": "scioglie", "tolse": "toglie", "volse": "volge", "condusse": "conduce",
    "aggiunse": "aggiunge", "raggiunse": "raggiunge", "riprese": "riprende", "riconobbe": "riconosce", "ricevette": "riceve", "credette": "crede",
    "era": "è", "aveva": "ha", "faceva": "fa", "diceva": "dice", "stava": "sta", "andava": "va", "beveva": "beve", "poteva": "può",
    "voleva": "vuole", "doveva": "deve", "sapeva": "sa", "veniva": "viene", "teneva": "tiene", "pareva": "pare", "vedeva": "vede",
}


def al_presente(w, noti):
    if w in IRREGOLARI:
        p = IRREGOLARI[w]
        return p if p in noti else None
    cand = []
    if w.endswith("ò") and len(w) > 3:
        cand = [w[:-1] + "a"]
    elif w.endswith("è") and len(w) > 3:
        cand = [w[:-1] + "e"]
    elif w.endswith("ì") and len(w) > 3:
        cand = [w[:-1] + "e", w[:-1] + "isce"]
    elif w.endswith("ette") and len(w) > 6:
        cand = [w[:-4] + "e"]
    elif w.endswith("ava") and len(w) > 4:
        cand = [w[:-3] + "a"]
    elif w.endswith("eva") and len(w) > 4:
        cand = [w[:-3] + "e"]
    elif w.endswith("iva") and len(w) > 4:
        cand = [w[:-3] + "e", w[:-3] + "isce"]
    for c in cand:
        if c in noti:
            return c
    return None
