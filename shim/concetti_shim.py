from __future__ import annotations

import json
import re
from pathlib import Path

RELAZIONI = ("è", "ha", "fa", "come", "dove", "mangia")
PASSO = 0.6
ARTICOLI = {"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "un'", "dei", "degli", "delle", "del", "della", "dello"}
PREP_DOVE = {"nell", "sull", "all", "in", "nel", "nella", "nello", "nei", "nelle", "negli", "sul", "sulla", "sui", "sulle", "al", "alla", "ai", "alle", "a", "sotto", "sopra"}
VERBI_DOVE = {"vive", "vivono", "sta", "stanno", "abita", "abitano", "si trova", "si trovano", "cresce", "crescono", "nuota", "nuotano"}
VERBI_MANGIA = {"mangia", "mangiano"}
VERBI_HA = {"ha", "hanno"}
VERBI_E = {"è", "e'", "sono"}


def _parole(testo):
    t = testo.lower().replace("’", "'").replace("e'", "è").replace("cos'è", "cosa è").replace("com'è", "come è").replace("com è", "come è")
    t = re.sub(r"\b(l|un|dell|nell|all|sull|quest|quell)'", r"\1 ", t)
    return re.findall(r"[a-zàèéìòóù]+", t)


def _togli_articoli(ps):
    return [p for p in ps if p not in ARTICOLI]


PLURALI = {"i", "gli", "le", "dei", "degli", "delle", "nei", "negli", "nelle", "ai", "agli", "alle", "sui", "sugli", "sulle"}


PLURALI_IRREGOLARI = {"corna": "corno", "uova": "uovo", "orecchie": "orecchio", "dita": "dito", "ossa": "osso", "braccia": "braccio",
                      "ginocchia": "ginocchio", "labbra": "labbro", "uomini": "uomo", "buoi": "bue", "ali": "ala"}

QUALITA = set("""freddo caldo giallo rosso verde blu bianco nero grigio rosa viola marrone arancione azzurro dolce amaro salato duro morbido
liscio ruvido pesante leggero bagnato asciutto grande piccolo alto basso lungo corto veloce lento forte debole buio chiaro scuro rotondo
pari dispari bello brutto buono cattivo vecchio giovane nuovo""".split())
VERBI_BASE = {"fa", "va", "viene", "sta", "dà", "dice", "vede", "sente", "pensa", "scrive", "legge", "gira", "misura", "spinge", "muove",
              "cresce", "vola", "nuota", "corre", "salta", "canta", "dorme", "respira", "cammina", "striscia", "digerisce", "attira", "beve",
              "mangia", "brilla", "cade", "porta", "serve", "contiene", "produce", "abbaia", "miagola", "insegna", "studia", "scrive",
              "legge", "gioca", "lavora", "guida", "cucina", "caccia", "conta", "cura", "scorre", "parla", "ascolta", "aiuta"}


def singolare(w, noti, art=None):
    if w in PLURALI_IRREGOLARI and PLURALI_IRREGOLARI[w] in noti:
        return PLURALI_IRREGOLARI[w]
    if w.endswith("gli") and w[:-1] + "io" in noti:
        return w[:-1] + "io"
    if w.endswith("i") and len(w) >= 3:
        for c in (w[:-1] + "o", w[:-1] + "e", w[:-1] + "a", w[:-2] + "o" if w.endswith(("chi", "ghi")) else None):
            if c and c in noti:
                return c
    if w.endswith("e") and art in PLURALI and len(w) > 3:
        for c in (w[:-1] + "a", w[:-1] + "ia" if w.endswith(("ce", "ge")) else None):
            if c and c in noti:
                return c
    return w


def _ultimo_nome(ps, noti):
    art = None
    out = None
    for p in ps:
        if p in ARTICOLI or p in PLURALI or p in PREP_DOVE:
            art = p
            continue
        out = singolare(p, noti, art)
        art = None
    return out


def _primo_nome(ps, noti):
    art = None
    for p in ps:
        if p in ARTICOLI or p in PLURALI or p in PREP_DOVE or p == "non":
            art = p
            continue
        return singolare(p, noti, art), art
    return None, None


VERBI_IRREGOLARI = {"fanno": "fa", "vanno": "va", "danno": "dà", "stanno": "sta", "hanno": "ha", "sono": "è", "bevono": "beve", "dicono": "dice",
                    "vengono": "viene", "escono": "esce", "muoiono": "muore", "tengono": "tiene", "possono": "può", "sanno": "sa"}
NUMERI_PAROLA = {"due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "cento", "mille"}
LEGAMI = PREP_DOVE | {"intorno", "dal", "dalla", "dallo", "dall", "dai", "dalle", "da", "di", "del", "della", "dello", "dell", "con", "per"}


def verbo_singolare(w, noti):
    if w in noti:
        return w
    if w in VERBI_IRREGOLARI:
        return VERBI_IRREGOLARI[w]
    for suf, nuovo in (("iscono", "isce"), ("ano", "a"), ("ono", "e")):
        if w.endswith(suf) and w[: -len(suf)] + nuovo in noti:
            return w[: -len(suf)] + nuovo
    return w


def qualita(w, noti):
    if w[-1:] in ("a", "i", "e") and w[:-1] + "o" in noti:
        return w[:-1] + "o"
    if w in noti:
        return w
    for c in (w[:-1] + "e", w[:-1] + "a"):
        if c in noti:
            return c
    return w


def _forma(ps):
    t = " ".join(ps)
    return re.sub(r"\b(l|dell|nell|all|sull|dall) (?=[aeiouàèéìòóù])", r"\1'", t)


def capisci(testo, noti, nomi=None, verbi=None):
    domanda = "?" in testo
    ps = _parole(testo)
    if not ps:
        return None
    verbi = (verbi or set()) | VERBI_BASE
    s = " ".join(ps)
    m = re.match(r"^(?:che )?cosa (?:hanno|c è|ce) (?:in )?comune (?:tra |fra )?(.+?) e (.+)$", s)
    if m and domanda:
        a_, b_ = _ultimo_nome(m.group(1).split(), noti), _ultimo_nome(m.group(2).split(), noti)
        if a_ and b_:
            for w in (a_, b_):
                if w not in noti:
                    return ("sconosciuta", w)
            return ("comune", a_, b_)
    m = re.match(r"^(?:che )?cosa sai (?:di|del|della|dello|dei|degli|delle|dell|sul|sulla|sullo|sui|sugli|sulle|sull)\s+(.+)$", s) \
        or re.match(r"^(?:dimmi|parlami) (?:tutto )?(?:quello che sai )?(?:di|del|della|dello|dei|degli|delle|dell|sul|sulla|sui)\s+(.+)$", s)
    if m:
        sogg = _ultimo_nome(m.group(1).split(), noti)
        if sogg and sogg in noti:
            return ("riassunto", sogg)
    m = re.match(r"^(?:che )?(?:cosa|che) (è|sono|ha|hanno|fa|fanno|mangia|mangiano) (.+)$", s) or re.match(r"^(come) (?:è|sono) (.+)$", s) \
        or re.match(r"^(dove) (?:vive|vivono|sta|stanno|abita|abitano|si trova|si trovano) (.+)$", s)
    if m and domanda:
        rel = {"è": "è", "sono": "è", "ha": "ha", "hanno": "ha", "fa": "fa", "fanno": "fa", "mangia": "mangia", "mangiano": "mangia",
               "come": "come", "dove": "dove"}[m.group(1)]
        sogg = _ultimo_nome(m.group(2).split(), noti)
        if not sogg:
            return None
        return ("domanda", sogg, rel) if sogg in noti else ("sconosciuta", sogg)
    m = re.match(r"^perch[eé] (.+)$", s)
    if m and domanda:
        r = capisci(m.group(1) + "?", noti, nomi, verbi)
        if r and r[0] == "verifica":
            return ("perche",) + tuple(r[1:])
        return r
    m = re.match(r"^quant[ie] (\w+) (?:ha|hanno) (.+)$", s)
    if m and domanda:
        sogg = _ultimo_nome(m.group(2).split(), noti)
        cosa = singolare(m.group(1), noti, "i")
        if sogg and cosa:
            for w in (sogg, cosa):
                if w not in noti:
                    return ("sconosciuta", w)
            return ("quanti", sogg, "ha", cosa)
    m = re.match(r"^(?:chi è |è )?(più|meno) (\w+) (?:è )?(?:tra |fra )?(.+?) o (.+)$", s)
    if m and domanda:
        a_, b_ = _ultimo_nome(m.group(3).split(), noti), _ultimo_nome(m.group(4).split(), noti)
        q_ = qualita(m.group(2), noti)
        if a_ and b_ and a_ in noti and b_ in noti:
            return ("confronto", q_, a_, b_, m.group(1))
        return None
    m = re.match(r"^chi (è|sono|ha|hanno|mangia|mangiano|vive|vivono|sta|stanno) (.+)$", s)
    if m and domanda:
        v = m.group(1)
        rel = {"è": "è", "sono": "è", "ha": "ha", "hanno": "ha", "mangia": "mangia", "mangiano": "mangia"}.get(v, "dove")
        cosa = _ultimo_nome(m.group(2).split(), noti)
        if not cosa:
            return None
        return ("chi", rel, cosa) if cosa in noti else ("sconosciuta", cosa)
    if "non" in ps:
        if domanda:
            return None
        r = capisci(re.sub(r"\bnon\s+", "", testo, flags=re.I), noti, nomi, verbi)
        if r and r[0] == "impara":
            return ("impara_no",) + tuple(r[1:])
        return None
    for i, p in enumerate(ps):
        if i == 0 or len(_togli_articoli(ps[:i])) > 2:
            continue
        vs = verbo_singolare(p, noti)
        if p in VERBI_E:
            rel, j = "è", i + 1
        elif p in VERBI_HA:
            rel, j = "ha", i + 1
        elif p in VERBI_MANGIA:
            rel, j = "mangia", i + 1
        elif p in VERBI_DOVE and ps[i + 1:i + 2] and ps[i + 1] in PREP_DOVE:
            rel, j = "dove", i + 1
        elif vs in verbi and vs in noti:
            rel, j = "fa", i + 1
        else:
            continue
        sogg = _ultimo_nome(ps[:i], noti)
        if not sogg or sogg in LEGAMI:
            return None
        if not any(x in ARTICOLI or x in PLURALI for x in ps[:i]) and not (nomi is not None and sogg in nomi):
            return None
        if rel == "fa":
            if sogg not in noti:
                return ("sconosciuta", sogg)
            return ("verifica" if domanda else "impara", sogg, "fa", vs, (vs + " " + _forma(ps[j:])).strip())
        resto = ps[j:]
        n0 = 0
        while n0 < len(resto) and (resto[n0] in LEGAMI or resto[n0] in ARTICOLI or resto[n0] in PLURALI or resto[n0] in NUMERI_PAROLA):
            n0 += 1
        if n0 >= len(resto):
            return None
        art = next((("i" if x in NUMERI_PAROLA else x) for x in reversed(resto[:n0]) if x in ARTICOLI or x in PLURALI or x in LEGAMI
                    or x in NUMERI_PAROLA), None)
        cosa = singolare(resto[n0], noti, art)
        if rel == "è" and not resto[:n0] and (qualita(resto[n0], noti) in QUALITA or nomi is None or cosa not in nomi):
            rel = "come"
            cosa = qualita(resto[n0], noti)
        forma = _forma(resto[: n0 + 1])
        for w in (sogg, cosa):
            if w not in noti:
                return ("sconosciuta", w)
        return ("verifica" if domanda else "impara", sogg, rel, cosa, forma)
    return None


class Concetti:

    def __init__(self, file=None):
        self.file = Path(file) if file else None
        self.f = {r: {} for r in RELAZIONI}
        self.forme = {}
        self.studiate = []
        self.no = {r: {} for r in RELAZIONI}
        if self.file is not None and self.file.exists():
            d = json.loads(self.file.read_text(encoding="utf-8"))
            for r in RELAZIONI:
                self.f[r] = {a: dict(b) for a, b in d.get(r, {}).items()}
            self.forme = dict(d.get("_forme", {}))
            self.studiate = list(d.get("_studiate", []))
            self.no = {r: {a: dict(b) for a, b in d.get("_no", {}).get(r, {}).items()} for r in RELAZIONI}

    def salva(self):
        if self.file is not None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(dict(self.f, _forme=self.forme, _studiate=self.studiate, _no=self.no), ensure_ascii=False), encoding="utf-8")

    def impara(self, a, rel, b, forma=None, forza=1.0):
        d = self.f[rel].setdefault(a, {})
        d[b] = min(3.0, d.get(b, 0.0) + forza)
        if forma:
            self.forme["%s|%s|%s" % (rel, a, b)] = forma

    def impara_no(self, a, rel, b):
        self.no[rel].setdefault(a, {})[b] = 1.0
        self.f[rel].get(a, {}).pop(b, None)

    def negato(self, a, rel, b):
        if b in self.no[rel].get(a, {}):
            return True
        return any(b in self.no[rel].get(c, {}) for c in self.f["è"].get(a, {}))

    def forma(self, a, rel, b, via=None):
        if rel == "come" and via:
            return None
        return self.forme.get("%s|%s|%s" % (rel, a, b)) or (self.forme.get("%s|%s|%s" % (rel, via, b)) if via else None)

    def n(self):
        return sum(len(b) for r in RELAZIONI for b in self.f[r].values())

    def categorie(self, a, passi=2):
        out, fronte = {}, {a: 1.0}
        for p in range(passi):
            nuovo = {}
            for x, w in fronte.items():
                for c, f in self.f["è"].get(x, {}).items():
                    v = w * (1.0 if p == 0 else PASSO) * min(1.0, f)
                    if c != a and v > out.get(c, 0.0):
                        out[c] = nuovo[c] = v
            fronte = nuovo
        return out

    def punteggi(self, a, rel):
        if rel == "è":
            cat = self.categorie(a, 3)
            return cat, {c: None for c in cat}
        out, via = {}, {}
        for b, f in self.f[rel].get(a, {}).items():
            out[b], via[b] = min(1.0, f), None
        for c, w in self.categorie(a, 2).items():
            for b, f in self.f[rel].get(c, {}).items():
                v = PASSO * w * min(1.0, f)
                if v > out.get(b, 0.0):
                    out[b], via[b] = v, c
        for b in list(out):
            if self.negato(a, rel, b):
                out.pop(b)
                via.pop(b, None)
        return out, via

    def chi(self, rel, b):
        out = {a: min(1.0, d[b]) for a, d in self.f[rel].items() if b in d}
        for a in list(self.f["è"].keys()):
            if a in out:
                continue
            if rel == "è":
                v = self.categorie(a, 3).get(b, 0.0)
            else:
                v = self.punteggi(a, rel)[0].get(b, 0.0)
            if v > 0:
                out[a] = v
        return out

    def contraddice(self, a, rel, b):
        sa = self.punteggi(a, rel)[0]
        return bool(sa) and b not in sa


def _art(w):
    from .frasi_shim import componi
    f = componi([w]).rstrip(".")
    a = f[: len(f) - len(w)].strip().lower()
    a = {"un": "", "uno": "", "una": "", "un'": ""}.get(a, a)
    if not a:
        if w[0] in "aeiouàèéìòù":
            a = "l'"
        elif w[0] == "z" or (w[0] == "s" and len(w) > 1 and w[1] not in "aeiouàèéìòù") or w[:2] in ("gn", "ps"):
            a = "lo" if not w.endswith("a") else "la"
        elif w.endswith(("ione", "tà", "tù", "udine", "ite")) or w in FEMMINILI_E:
            a = "la"
        else:
            a = {"a": "la", "i": "i"}.get(w[-1], "il")
    return a


SENZA_ARTICOLO = {"marte", "giove", "venere", "saturno", "mercurio", "urano", "nettuno", "roma", "milano", "napoli", "torino", "firenze",
                  "venezia", "bologna", "genova", "palermo", "bari", "londra", "parigi", "berlino", "madrid", "internet"}


def con_articolo(w):
    if w in SENZA_ARTICOLO:
        return w.capitalize()
    a = _art(w)
    return (a + w) if a.endswith("'") else (a + " " + w)


FEMMINILI_E = {"arte", "ape", "acqua", "arma", "isola", "erba", "orbita", "estate", "anima", "immagine", "origine", "età", "auto"}


def indeterminativo(w):
    a = _art(w)
    if a == "lo":
        return "uno " + w
    if a == "la":
        return "una " + w
    if a == "l'":
        fem = (w.endswith("a") and w not in ("pianeta", "artista")) or w.endswith(("tà", "tù", "ione", "udine", "ite")) or w in FEMMINILI_E
        return ("un'" if fem else "un ") + w
    if a in ("i", "gli", "le"):
        return a + " " + w
    return "un " + w


def con_preposizione(w, prep="in"):
    a = _art(w)
    return {"il": "nel ", "lo": "nello ", "la": "nella ", "l'": "nell'", "i": "nei ", "gli": "negli ", "le": "nelle "}.get(a, "in ") + w


def frase(sogg, rel, cose, forme=None):
    if not cose:
        return "Non lo so."
    S = con_articolo(sogg)
    S = S[:1].upper() + S[1:]
    forme = forme or {}
    if rel in ("ha", "mangia", "dove", "fa", "come") and any(c in forme for c in cose):
        verbo = {"ha": "ha ", "mangia": "mangia ", "dove": "vive ", "fa": "", "come": "è "}[rel]
        parti = []
        for c in cose:
            f = forme.get(c)
            if f is None:
                f = {"ha": con_articolo(c), "mangia": con_articolo(c), "dove": con_preposizione(c), "fa": c, "come": c}[rel]
            parti.append(f)
        return "%s %s%s." % (S, verbo, " e ".join(parti))
    if rel == "è":
        return "%s è %s." % (S, " e ".join(indeterminativo(c) for c in cose))
    if rel == "come":
        return "%s è %s." % (S, " e ".join(accorda(c, sogg) for c in cose))
    if rel == "dove":
        return "%s vive %s." % (S, " e ".join(con_preposizione(c) for c in cose))
    if rel == "fa":
        return "%s %s." % (S, " e ".join(cose))
    verbo = {"ha": "ha", "mangia": "mangia"}[rel]
    return "%s %s %s." % (S, verbo, " e ".join(con_articolo(c) for c in cose))


def accorda(agg, sogg):
    a = _art(sogg)
    fem = a == "la" or (a == "l'" and ((sogg.endswith("a") and sogg not in ("pianeta",)) or sogg.endswith(("tà", "ione")) or sogg in FEMMINILI_E))
    return agg[:-1] + "a" if fem and agg.endswith("o") else agg


PLURALI_SCRITTI = {v: k for k, v in PLURALI_IRREGOLARI.items()}
PLURALI_SCRITTI.update({"uomo": "uomini", "pianeta": "pianeti"})


def plurale(w):
    if w in PLURALI_SCRITTI:
        return PLURALI_SCRITTI[w]
    if w.endswith(("à", "ù", "è", "ì", "ò")) or len(w) < 3:
        return w
    if w.endswith("io"):
        return w[:-1]
    if w.endswith(("co", "go")):
        return w[:-1] + "hi"
    if w.endswith(("ca", "ga")):
        return w[:-1] + "he"
    if w.endswith("a"):
        return w[:-1] + "e"
    if w.endswith(("o", "e")):
        return w[:-1] + "i"
    return w
