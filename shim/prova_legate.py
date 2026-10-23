import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T
from shim import prova_conversazione as C
from shim.frasi_shim import componi

T.URL = C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_legate.json"
TEMI = ("famiglia soldi viaggio amici scuola lavoro festa casa cane mare pane notte libro musica guerra amore città strada medico treno "
        "bambino donna cuore tempo gioco uomo sole acqua fuoco cibo").split()


def prime5():
    from shim.associazioni_shim import Associazioni
    from shim.corpus_shim import CATEGORIE
    from shim.lessico_2000 import NON_NOMI
    B = (Path(__file__).resolve().parents[1] / "stato")
    voc = json.loads((B / "shim_vocab_2000.json").read_text(encoding="utf-8"))
    dc = json.loads((B / "shim_classi_2000.json").read_text(encoding="utf-8"))
    nomi = {w for c in ("animali", "cibo", "natura", "casa", "persone") for w in CATEGORIE[c]} - {"ciao"}
    cl = dict(zip(dc["parole"], dc["classi"]))
    P = np.asarray(dc["conteggi"], float)
    art = P[:, 1] / (P.sum(1) + 1e-9)
    ixc = {w: i for i, w in enumerate(dc["parole"])}
    forte = np.array([((w in nomi) or (cl.get(w) == "nome" and art[ixc[w]] >= 0.5)) and w not in NON_NOMI for w in voc])
    W, S = Associazioni(voc, B / "shim_assoc_2000.npz"), Associazioni(voc, B / "shim_assoc_storie.npz")
    ix = {w: i for i, w in enumerate(voc)}
    out = {}
    for w in TEMI:
        k = ix[w]
        rw = W.ppmi()[k] / (W.ppmi()[k].max() + 1e-9) + W.forza(k)
        rs = S.ppmi()[k] / (S.ppmi()[k].max() + 1e-9) + S.forza(k)
        r = np.sqrt(rw * rs) * forte
        r[k] = 0
        out[w] = [voc[i] for i in np.argsort(-r)[:5]]
    return out


def main():
    from shim.lessico_2000 import NON_NOMI
    attese = prime5()
    ris = {"t": time.strftime("%d/%m %H:%M"), "x_reale_inizio": T.get("/stato")["x_reale"], "prime5": attese}
    l1 = l2 = l3 = 0
    det = []
    for w in TEMI:
        r = C.turno(componi([w]).rstrip(".") + "?")
        parole = [x for d in r["detti"] for x in d["parole"]]
        legate = [x for x in parole if x != w]
        l1 += w in parole
        l2 += any(x in attese[w] for x in legate)
        l3 += any(x in NON_NOMI for x in legate)
        det.append({"tema": w, "detti": [d["testo"] for d in r["detti"]], "parole": parole, "primo_s": r["primo_s"]})
        print("%-9s %5s s | %s | attese %s" % (w, r["primo_s"], " / ".join(d["testo"] or "" for d in r["detti"]), attese[w]), flush=True)
        time.sleep(1.5)
    ris.update(L1_quota=l1, L1=l1 >= 27, L2_quota=l2, L2=l2 >= 20, L3_aggettivi=l3, L3=l3 == 0, dettaglio=det,
               x_reale_fine=T.get("/stato")["x_reale"], L4="da giudicare a mano (dichiarato)")
    print("L1 %d/30 %s | L2 %d/30 %s | L3 aggettivi %d %s" % (l1, "SI'" if ris["L1"] else "NO", l2, "SI'" if ris["L2"] else "NO",
                                                         l3, "SI'" if ris["L3"] else "NO"), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
