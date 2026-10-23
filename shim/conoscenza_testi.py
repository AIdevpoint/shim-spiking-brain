from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

QUI = (Path(__file__).resolve().parents[1] / "stato")
TESTI = QUI / "shim_testi"
N_PAROLE = 700
MIN_FREQ = 15
FINESTRA = 4

TITOLI = """Gatto_domestico Cane Cavallo Bovini Pecora Maiale Gallina Uccello Pesce Lupo Orso Leone Elefante Topo Coniglio Ape Farfalla Serpente
Balena Delfino Aquila Rana Formica Ragno Tigre Scimmia Giraffa Pinguino Pane Latte Formaggio Pasta_(alimento) Pizza Mela Arancia
Uovo_(alimento) Carne Riso Zucchero Miele Vino Caffè Cioccolato Pomodoro Patata Frutta Verdura Sole Luna Mare Fiume Montagna Albero
Fiore Pioggia Neve Vento Terra Cielo Nuvola Lago Foresta Deserto Isola Vulcano Terremoto Stagione Estate Inverno Primavera Autunno Cuore
Mano Occhio Cervello_umano Sangue Dente Pelle Orecchio Sonno Muscolo Osso Neurone Casa Porta Finestra Letto Tavolo Sedia Orologio
Telefono Automobile Bicicletta Treno Aeroplano Nave Libro Scuola Città Strada Ponte Madre Padre Nonno Bambino Amicizia Famiglia Lavoro
Gioco Musica Sport Calcio Cinema Teatro Pittura Danza Italia Roma Abruzzo Pescara Europa Africa Asia America Oceano Francia Germania
Spagna Napoli Milano Venezia Firenze Acqua Fuoco Aria Energia Elettricità Luce Colore Tempo_(fisica) Numero Matematica Fisica Chimica
Biologia Computer Internet Python Programmazione Intelligenza_artificiale Robot Atomo Cellula Pianeta Stella Sistema_solare Gravità
Dinosauro Evoluzione_biologica Storia Dante_Alighieri Leonardo_da_Vinci Antica_Roma Medioevo Rinascimento Guerra Pace Lingua_italiana
Alfabeto Parola Emozione Paura Gioia Memoria_(neuroscienze) Linguaggio Medico Ospedale Medicina Farmaco Gelato Biscotto Torta Zuppa
Insalata Pesca Agricoltura Fattoria Mercato Denaro Banca Negozio Ristorante Albergo Museo Biblioteca Chiesa Castello Piazza""".split()

STOP = set("""il lo la i gli le un uno una di a da in con su per tra fra e ed o ma che non è sono era erano essere ha hanno aveva avere come
anche più molto molti molte molta poco poche quello quella quelli quelle questo questa questi queste suo sua suoi sue loro dei delle degli
del della dello dai dalle dagli dal dalla dallo nel nella nello nei nelle negli sul sulla sullo sui sulle sugli al alla allo ai alle agli
col si se ci ne mi ti vi lui lei noi voi essi esse quale quali quando dove cui chi cosa tutto tutti tutte tutta altro altra altri altre
ogni stesso stessa stessi stesse fu furono sarà saranno sia siano può possono può essere stato stata stati state viene vengono detto
dopo prima durante senza sotto sopra verso presso oltre tuttavia però quindi poiché perché mentre ancora già solo soltanto così dunque
isbn treccani wikipedia archiviato url accesso sito pag ibid cfr http https www doi pdf sul web
wikimedia commons file consultato immagini immagine originale gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre
novembre dicembre xix xviii xvii xvi xv xx xxi viii vii iii note voci bibliografia collegamenti esterni altri progetti voce pagina
anno anni secolo detta detti dette ad ed od l' un' dell' all' nell' sull' dall' c' v' m' t' s' n' d' e' quest' quell' po' fa fanno
fare fatto fatta fatti fatte oggi circa tipo parte parti modo altre nome nomi chiamato chiamata chiamati chiamate diverse diversi diversa
diverso grande grandi nuovo nuova nuovi nuove primo prima primi prime secondo seconda secondi seconde terzo terza due tre quattro cinque
sei sette otto nove dieci cento mille milioni uno una può possibile varie vari vario varia ogni inoltre infatti ad""".split())

PAROLA = re.compile(r"[a-zàèéìòóù]+(?:'[a-zàèéìòóù]+)?", re.IGNORECASE)


def scarica():
    TESTI.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in TITOLI:
        f = TESTI / (re.sub(r"[^\w()-]", "_", t) + ".txt")
        if f.exists() and f.stat().st_size > 500:
            ok += 1
            continue
        url = ("https://it.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles="
               + urllib.parse.quote(t.replace("_", " ")))
        txt = ""
        for tentativo in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SHIM-ricerca/1.0 (esperimento di neuroscienze simulate)"})
                d = json.loads(urllib.request.urlopen(req, timeout=40).read())
                txt = list(d["query"]["pages"].values())[0].get("extract", "")
                break
            except Exception as e:
                if "429" in str(e) and tentativo < 3:
                    time.sleep(45 * (tentativo + 1))
                    continue
                print("  non riuscito %s: %s" % (t, e), flush=True)
                break
        if len(txt) > 500:
            f.write_text(txt, encoding="utf-8")
            ok += 1
        else:
            print("  vuoto %s" % t, flush=True)
        time.sleep(1.6)
    print("voci scaricate: %d su %d" % (ok, len(TITOLI)), flush=True)


def voci(parte=None):
    fs = sorted(TESTI.glob("*.txt"))
    if parte == "prova":
        return [f for i, f in enumerate(fs) if i % 5 == 4]
    if parte == "impara":
        return [f for i, f in enumerate(fs) if i % 5 != 4]
    return fs


def frasi_dei_testi(parte=None):
    for f in voci(parte):
        txt = f.read_text(encoding="utf-8")
        txt = re.sub(r"==+[^=]+==+", " . ", txt)
        for fr in re.split(r"[.!?;\n]+", txt):
            ps = [p.lower().strip("'") for p in PAROLA.findall(fr)]
            ps = [p for p in ps if len(p) >= 3 and p not in STOP and "'" not in p]
            if len(ps) >= 2:
                yield ps


def costruisci(parte=None):
    from .associazioni_shim import Associazioni
    from .corpus_shim import FRASI, PAROLE as BASE
    conta = Counter()
    n_fr = 0
    for ps in frasi_dei_testi(parte):
        conta.update(ps)
        n_fr += 1
    extra = [w for w, c in conta.most_common(N_PAROLE * 2) if c >= MIN_FREQ and w not in BASE][:N_PAROLE]
    parole = list(BASE) + extra
    (QUI).mkdir(parents=True, exist_ok=True)
    (QUI / "shim_vocab.json").write_text(json.dumps(parole, ensure_ascii=False), encoding="utf-8")
    A = Associazioni(parole)
    for ps in frasi_dei_testi(parte):
        A.impara(ps)
    for _ in range(3):
        for f in FRASI:
            A.impara(f)
    A.file = QUI / "shim_assoc.npz"
    A.salva()
    print("frasi dei testi: %d | parole: %d (le 62 di base + %d dai testi)" % (n_fr, len(parole), len(extra)), flush=True)
    return A, parole


def esempi(A, parole, sonde=("gatto", "sole", "acqua", "mamma", "casa", "pane", "cervello", "computer", "roma", "mare")):
    for w in sonde:
        if w in A.idx:
            k = A.idx[w]
            print("  %-10s | associate: %s | simili: %s" % (w, " ".join(parole[j] for j in A.prossime(k, 6)), " ".join(parole[j] for j in A.simili(k, 4))), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scarica":
        scarica()
    else:

        A, parole = costruisci("impara" if (len(sys.argv) > 2 and sys.argv[2] == "impara") else None)
        esempi(A, parole)
