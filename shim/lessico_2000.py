from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

QUI = (Path(__file__).resolve().parents[1] / "stato")
FREQ = Path(r"dati_scaricati\frequenze_it\it_50k.txt")
N_PAROLE = 2000
PAROLACCE = set("""cazzo cazzi cazzate cazzata merda puttana puttane troia culo stronzo stronza stronzi fottuto fottuta fottere fottiti fanculo
vaffanculo coglione coglioni minchia scopare scopato scopata bastardo bastarda bastardi figa porcodio""".split())

ESCLUSE = set("""charlie mike joe george jane jake steve lee kate eddie claire emma you fbi dna college traduzione revisione sottotitoli
stronzate porca frega dannazione""".split())
NUMERI = "uno due tre quattro cinque sei sette otto nove dieci cento mille".split()
CONSONANTE_OK = set("signor buon film bar sport computer internet autobus gas tram camion".split())
ARTICOLI = set("il lo la i gli le l un uno una".split())
PREP_ART = set("""del dello della dei degli delle dell al allo alla ai agli alle all dal dallo dalla dai dagli dalle dall nel nello nella nei negli
nelle nell sul sullo sulla sui sugli sulle sull""".split())
CLITICI = set("non mi ti si ci vi lo la li le gli ne".split())
FINESTRA = 4
PAROLA = re.compile(r"[a-zàèéìòóù]+", re.IGNORECASE)


ACCENTI = {"perche": "perché", "perchè": "perché", "piu": "più", "cosi": "così", "gia": "già", "puo": "può", "citta": "città", "sara": "sarà",
           "pero": "però", "verita": "verità", "liberta": "libertà", "eta": "età", "societa": "società", "papa": "papà", "caffe": "caffè",
           "lunedi": "lunedì", "martedi": "martedì", "mercoledi": "mercoledì", "giovedi": "giovedì", "venerdi": "venerdì", "felicita": "felicità",
           "possibilita": "possibilità", "realta": "realtà", "universita": "università", "qualita": "qualità", "meta": "metà", "tivu": "tivù",
           "potra": "potrà", "dira": "dirà", "fara": "farà", "andra": "andrà", "verra": "verrà", "avra": "avrà", "sapra": "saprà",
           "dara": "darà", "stara": "starà", "vorra": "vorrà", "dovra": "dovrà", "finche": "finché", "affinche": "affinché", "ne'": "né",
           "giu": "giù", "cioe": "cioè", "percio": "perciò", "laggiu": "laggiù", "gesu": "gesù", "saro": "sarò", "diro": "dirò", "andro": "andrò",
           "avro": "avrò", "faro": "farò", "potro": "potrò", "dovro": "dovrò", "vorro": "vorrò", "verro": "verrò"}


def _accenta(w, noti):
    return ACCENTI.get(w, w)


FALSI_VERBI = set("collina singola gene more este marte cellula molecola stabile disciplina popola solare regola secola".split())
N_VERBI = 250


MATEMATICA = """zero undici dodici tredici quattordici quindici sedici diciassette diciotto diciannove venti trenta quaranta cinquanta sessanta
settanta ottanta novanta numero numeri cifra cifre somma sottrazione moltiplicazione divisione addizione risultato uguale pari dispari doppio
triplo metà terzo quarto maggiore minore calcolo contare matematica geometria aritmetica triangolo quadrato cerchio rettangolo lato lati angolo
angoli retta linea punto area perimetro raggio diametro centro forma figura cubo sfera piramide misura metro centimetro chilometro chilo grammo
litro frazione percentuale equazione formula problema operazione operazioni tabellina moltiplicare dividere sommare sottrarre multiplo divisore
quantità infinito""".split()


def terze_persone():
    d = json.loads((QUI / "shim_wiki" / "vocabolario_wiki.json").read_text(encoding="utf-8"))
    wiki, fr = d["parole"], dict(zip(d["parole"], d["frequenze"]))
    ws = set(wiki)
    out = {}
    for w in wiki:
        if len(w) < 6:
            continue
        forme = {"are": [w[:-3] + "a"], "ere": [w[:-3] + "e"], "ire": [w[:-3] + "e", w[:-3] + "isce"]}.get(w[-3:], [])
        for x in forme:
            if x in ws and len(x) >= 4 and x not in FALSI_VERBI:
                out[x] = fr[x]
    return out


NON_NOMI = set("""grande piccolo comune maggior principali propria forte ultimi antica interno propri piccola notevole vero certo proprie certa
unico contrario buon sola ultima strano unica vecchio vivo ottimo giovane vera vecchia brutto incredibile migliori terribile grosso bellissima
brutta scorsa intero ottima intera bellissimo strana enorme grave improvviso grandioso doppio terzo quarto lemma wikizionario signor resti""".split())


SCUOLA = """mammifero insetto rettile anfibio uccello pesce animale pianta maiale elefante orso volpe coniglio serpente rana ape formica farfalla
mosca zanzara ragno balena delfino squalo gallina gallo aquila pinguino pappagallo tartaruga lucertola scimmia giraffa zebra cammello asino
capra pecora mucca cavallo leone tigre lupo topo cane gatto
zampa coda pelo piuma penna ala becco squama pinna artiglio baffi corno zoccolo guscio dente
erba foglia radice tronco ramo seme frutto frutta verdura pera banana arancia limone uva fragola pomodoro patata carota insalata ciliegia
collina bosco foresta deserto oceano vulcano pianura valle roccia pietra sasso sabbia arcobaleno temporale fulmine nuvola
marte giove venere saturno galassia universo cometa satellite orbita atmosfera ossigeno pianeta stella
polmone muscolo orecchio stomaco fegato scheletro nervo gola ginocchio sangue
metallo ferro rame plastica vetro liquido solido gas vapore elettricità magnete
giallo arancione viola marrone grigio rosa azzurro
lento salato amaro morbido duro liscio ruvido pesante leggero bagnato asciutto
cammina respira striscia vola nuota
tastiera schermo algoritmo istruzione variabile software processore pixel
geografia nazione confine
maestra maestro quaderno matita lavagna zaino alunno
motore ruota bicicletta autobus
primavera autunno inverno
proboscide criniera stagno fattoria zucchero organo cuscino rotondo veicolo pagina uovo osso dito digerisce attira""".split()

def vocabolario():
    from .conoscenza_wiki import STOP
    base = json.loads((QUI / "shim_vocab.json").read_text(encoding="utf-8"))
    base = base["parole"] if isinstance(base, dict) else base
    wiki = json.loads((QUI / "shim_wiki" / "vocabolario_wiki.json").read_text(encoding="utf-8"))["parole"]
    noti = set(wiki) | set(base)
    voc = list(base)
    dentro = set(voc)
    for w in NUMERI:
        if w not in dentro:
            voc.append(w)
            dentro.add(w)
    scartate = Counter()
    for riga in FREQ.read_text(encoding="utf-8").splitlines():
        if len(voc) >= N_PAROLE:
            break
        w = riga.split()[0]
        if not re.fullmatch(r"[a-zàèéìòóù]{3,}", w):
            scartate["non parola"] += 1
            continue
        w = _accenta(w, noti)
        if w in STOP or riga.split()[0] in STOP or w in PAROLACCE or w in ESCLUSE or w in dentro:
            scartate["piccola/parolaccia/doppia"] += 1
            continue
        if w[-1] not in "aeiouàèéìòóù" and w not in CONSONANTE_OK:
            scartate["finisce in consonante"] += 1
            continue
        voc.append(w)
        dentro.add(w)
    terze = terze_persone()
    n_v = 0
    for w, _ in sorted(terze.items(), key=lambda kv: -kv[1]):
        if n_v >= N_VERBI:
            break
        if w not in dentro:
            voc.append(w)
            dentro.add(w)
            n_v += 1
    print("verbi in terza persona aggiunti: %d" % n_v, flush=True)
    n_m = 0
    for w in MATEMATICA:
        if w not in dentro:
            voc.append(w)
            dentro.add(w)
            n_m += 1
    print("parole della matematica aggiunte: %d" % n_m, flush=True)
    (QUI / "shim_vocab_2000.json").write_text(json.dumps(voc, ensure_ascii=False), encoding="utf-8")
    nuove = voc[len(base):]
    print("vocabolario: %d parole (%d di prima + %d dal parlato); nuove nella Wikipedia: %d; scartate: %s" % (
        len(voc), len(base), len(nuove), sum(w in set(wiki) for w in nuove), dict(scartate)), flush=True)
    print("nuove (prime 120):", " ".join(nuove[:120]), flush=True)
    return voc


_VOC = None


def _init(voc):
    global _VOC
    _VOC = voc


def _passata(testi):
    from .conoscenza_wiki import pulisci
    co, ordine, prima = Counter(), Counter(), Counter()
    V = len(_VOC)
    for t in testi:
        for fr in re.split(r"[.!?;\n:]+", pulisci(t)):
            toks = [p.lower() for p in PAROLA.findall(fr)]
            ids, prec = [], None
            for tok in toks:
                k = _VOC.get(tok)
                if k is not None:
                    c = 0 if prec is None else (1 if (prec in ARTICOLI or prec in PREP_ART) else (2 if prec in CLITICI else 3))
                    prima[k * 4 + c] += 1
                    ids.append(k)
                prec = tok
            for i, a in enumerate(ids):
                if i + 1 < len(ids) and ids[i + 1] != a:
                    ordine[a * V + ids[i + 1]] += 1
                for d in range(1, FINESTRA + 1):
                    if i + d < len(ids) and ids[i + d] != a:
                        co[a * V + ids[i + d]] += 1.0 / d
    return list(co.items()), list(ordine.items()), list(prima.items())


def conoscenza(max_pagine=None):
    from .conoscenza_wiki import pagine, DUMP, _gruppi
    voc = json.loads((QUI / "shim_vocab_2000.json").read_text(encoding="utf-8"))
    ix = {w: i for i, w in enumerate(voc)}
    V = len(voc)
    N = np.zeros(V * V, np.float64)
    O = np.zeros(V * V, np.float64)
    P = np.zeros(V * 4, np.float64)
    t0, pag = time.time(), 0
    with mp.Pool(max(2, (os.cpu_count() or 4) - 2), initializer=_init, initargs=(ix,)) as pool:
        for k, (co, ordi, pri) in enumerate(pool.imap_unordered(_passata, _gruppi(pagine(DUMP, max_pagine)), chunksize=1)):
            if co:
                kk, vv = zip(*co)
                np.add.at(N, np.asarray(kk, np.int64), np.asarray(vv))
            if ordi:
                kk, vv = zip(*ordi)
                np.add.at(O, np.asarray(kk, np.int64), np.asarray(vv))
            if pri:
                kk, vv = zip(*pri)
                np.add.at(P, np.asarray(kk, np.int64), np.asarray(vv))
            pag += 200
            if k % 50 == 0:
                print("  %d pagine circa, %.0f s" % (pag, time.time() - t0), flush=True)
    N = N.reshape(V, V)
    N = (N + 0.5 * N.T).astype(np.float32)
    nk = N.sum(1) + N.sum(0)
    np.savez(QUI / "shim_assoc_2000.npz", N=N, nk=nk.astype(np.float32))
    np.save(QUI / "shim_ordine_2000.npy", O.reshape(V, V).astype(np.float32))
    P = P.reshape(V, 4)
    tot = P.sum(1) + 1e-9
    classi = []
    for i in range(V):
        if P[i, 1] / tot[i] >= 0.35:
            classi.append("nome")
        elif P[i, 2] / tot[i] >= 0.25:
            classi.append("verbo")
        else:
            classi.append("altro")
    (QUI / "shim_classi_2000.json").write_text(json.dumps({"parole": voc, "classi": classi, "conteggi": P.astype(int).tolist()},
                                                        ensure_ascii=False), encoding="utf-8")
    print("conoscenza: %d parole, %d associazioni non nulle, %d parole mai viste, classi %s, %d pagine circa, %.0f s" % (
        V, int((N > 0).sum()), int((tot < 1).sum()), dict(Counter(classi)), pag, time.time() - t0), flush=True)


STORIE = Path(r"dati_scaricati\storie")


def storie():
    voc = json.loads((QUI / "shim_vocab_2000.json").read_text(encoding="utf-8"))
    ix = {w: i for i, w in enumerate(voc)}
    from .frasi_shim import PICCOLE
    from .verbi_presente import al_presente
    noti = set(voc)
    V = len(voc)
    O = np.zeros((V, V), np.float32)
    n_fr = n_cop = n_conv = 0
    for f in sorted(STORIE.glob("*.txt")):
        t = f.read_text(encoding="utf-8").replace("\u2019", "'")
        for fr in re.split(r"[.!?;:\n\u2014\u00ab\u00bb\"()]+", t):
            prec = None
            for tok in (x.lower() for x in PAROLA.findall(fr)):
                k = ix.get(tok)
                if k is None:
                    p_ = al_presente(tok, noti)
                    if p_ is not None:
                        k = ix[p_]
                        n_conv += 1
                if k is not None:
                    if prec is not None and prec != k:
                        O[prec, k] += 1
                        n_cop += 1
                    prec = k
                elif tok in PICCOLE or tok in ("lo", "gli", "ne", "ci", "si"):
                    continue
                else:
                    prec = None
            n_fr += 1
    np.save(QUI / "shim_ordine_storie.npy", O)
    print("storie: %d frasi, %d coppie vicine, %d coppie diverse, %d verbi ricondotti al presente -> shim_ordine_storie.npy" % (n_fr, n_cop, int((O > 0).sum()), n_conv), flush=True)


DUMP_STORIE = Path(r"dati_scaricati\wikisource\itwikisource.xml.bz2")


def _pagine_wikisource(percorso=DUMP_STORIE):
    import bz2
    import xml.etree.ElementTree as ET
    try:
        with bz2.BZ2File(str(percorso)) as f:
            ns = None
            for evento, el in ET.iterparse(f, events=("end",)):
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "ns":
                    ns = el.text
                elif tag == "text":
                    t = el.text or ""
                    if ns in ("0", "108") and len(t) > 200 and not t.lstrip().upper().startswith("#RINVIA"):
                        yield None, t
                elif tag == "page":
                    el.clear()
    except (EOFError, OSError, ET.ParseError):
        return


def _passata_storie(testi):
    from .conoscenza_wiki import pulisci
    from .verbi_presente import al_presente
    co, ordine = {}, {}
    V = len(_VOC)
    noti = _VOC
    for t in testi:
        t = pulisci(t).replace("\u2019", "'")
        for fr in re.split(r"[.!?;:\n\u2014\u00ab\u00bb\"()]+", t):
            ids, prec = [], None
            for tok in (x.lower() for x in PAROLA.findall(fr)):
                k = _VOC.get(tok)
                if k is None:
                    p_ = al_presente(tok, noti)
                    if p_ is not None:
                        k = _VOC[p_]
                if k is not None:
                    if prec is not None and prec != k:
                        ordine[prec * V + k] = ordine.get(prec * V + k, 0) + 1
                    prec = k
                    ids.append(k)
                elif tok in _PICCOLE:
                    continue
                else:
                    prec = None
            for i, a in enumerate(ids):
                for d in range(1, FINESTRA + 1):
                    if i + d < len(ids) and ids[i + d] != a:
                        co[a * V + ids[i + d]] = co.get(a * V + ids[i + d], 0.0) + 1.0 / d
    return list(co.items()), list(ordine.items())


_PICCOLE = None


def _init_storie(voc):
    global _VOC, _PICCOLE
    _VOC = voc
    from .frasi_shim import PICCOLE
    _PICCOLE = set(PICCOLE) | {"lo", "gli", "ne", "ci", "si"}


def storie_grandi(max_pagine=None):
    from .conoscenza_wiki import _gruppi
    voc = json.loads((QUI / "shim_vocab_2000.json").read_text(encoding="utf-8"))
    ix = {w: i for i, w in enumerate(voc)}
    V = len(voc)
    N = np.zeros(V * V, np.float64)
    O = np.zeros(V * V, np.float64)
    t0, pag = time.time(), 0

    def _lim(gen):
        for k, x in enumerate(gen):
            if max_pagine and k >= max_pagine:
                return
            yield x
    with mp.Pool(max(2, (os.cpu_count() or 4) - 2), initializer=_init_storie, initargs=(ix,)) as pool:
        for k, (co, ordi) in enumerate(pool.imap_unordered(_passata_storie, _gruppi(_lim(_pagine_wikisource())), chunksize=1)):
            if co:
                kk, vv = zip(*co)
                np.add.at(N, np.asarray(kk, np.int64), np.asarray(vv))
            if ordi:
                kk, vv = zip(*ordi)
                np.add.at(O, np.asarray(kk, np.int64), np.asarray(vv))
            pag += 200
            if k % 100 == 0:
                print("  %d pagine circa, %.0f s" % (pag, time.time() - t0), flush=True)
    N = N.reshape(V, V)
    N = (N + 0.5 * N.T).astype(np.float32)
    np.savez(QUI / "shim_assoc_storie.npz", N=N, nk=(N.sum(1) + N.sum(0)).astype(np.float32))
    np.save(QUI / "shim_ordine_storie.npy", O.reshape(V, V).astype(np.float32))
    print("storie grandi: %d pagine circa, %d coppie vicine (%d diverse), %d associazioni, %.0f s" % (
        pag, int(O.sum()), int((O > 0).sum()), int((N > 0).sum()), time.time() - t0), flush=True)


POSSESSIVI = set("mio mia miei mie tuo tua tuoi tue suo sua suoi sue nostro nostra nostri nostre vostro vostra vostri vostre loro".split())


def riclassifica():
    f = QUI / "shim_classi_2000.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    P = np.asarray(d["conteggi"], np.float64)
    tot = P.sum(1) + 1e-9
    terze = terze_persone()
    nuove = []
    for w, c, p, t in zip(d["parole"], d["classi"], P, tot):
        if w in POSSESSIVI:
            c = "altro"
        elif w.endswith(("are", "ere", "ire", "arsi", "ersi", "irsi")):
            c = "infinito"
        elif c == "verbo" and w[-1] in "oi":
            c = "verbo12"
        elif c == "altro" and w[-1] in "ae" and p[2] / t >= 0.15:
            c = "verbo"
        elif w in terze and p[1] / t < 0.4:
            c = "verbo"
        nuove.append(c)
    d["classi_conteggi"] = d["classi"]
    d["classi"] = nuove
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print("riclassificate:", dict(Counter(nuove)), flush=True)
    print("verbi:", [w for w, c in zip(d["parole"], nuove) if c == "verbo"][:60], flush=True)


def aggiungi_scuola():
    f = QUI / "shim_vocab_2000.json"
    voc = json.loads(f.read_text(encoding="utf-8"))
    dentro = set(voc)
    nuove = [w for w in dict.fromkeys(SCUOLA) if w not in dentro]
    f.write_text(json.dumps(voc + nuove, ensure_ascii=False), encoding="utf-8")
    print("parole della scuola aggiunte: %d (vocabolario %d -> %d): %s" % (len(nuove), len(voc), len(voc) + len(nuove), " ".join(nuove)), flush=True)
    return nuove


if __name__ == "__main__":
    mp.freeze_support()
    a = sys.argv[1:]
    if a and a[0] == "vocabolario":
        vocabolario()
    elif a and a[0] == "aggiungi_scuola":
        aggiungi_scuola()
    elif a and a[0] == "storie":
        storie()
    elif a and a[0] == "storie_grandi":
        storie_grandi(int(a[a.index("--max-pagine") + 1]) if "--max-pagine" in a else None)
    elif a and a[0] == "riclassifica":
        riclassifica()
    elif a and a[0] == "conoscenza":
        conoscenza(int(a[a.index("--max-pagine") + 1]) if "--max-pagine" in a else None)
    else:
        print(__doc__)
