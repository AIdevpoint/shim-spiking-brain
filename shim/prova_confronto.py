import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import prova_conversazione as C
from shim.numeri_shim import PAROLE_NUMERO

C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_confronto.json"
LONTANI = [(2, 9), (8, 1), (3, 10), (6, 1), (1, 5), (9, 4), (2, 7), (10, 5), (4, 8), (7, 2)]
VICINI = [(6, 7), (4, 3), (8, 9), (2, 1), (5, 6), (10, 9), (3, 4), (7, 6), (1, 2), (9, 8)]


def domanda(a, b, grande):
    return "È più %s il %s o il %s?" % ("grande" if grande else "piccolo", PAROLE_NUMERO[a - 1], PAROLE_NUMERO[b - 1])


def main():
    ris = {"t": time.strftime("%d/%m %H:%M"), "x_reale_inizio": C.T.get("/stato")["x_reale"], "turni": []}
    quote = {}
    for nome, coppie in (("lontani", LONTANI), ("vicini", VICINI)):
        g = 0
        for i, (a, b) in enumerate(coppie):
            grande = i % 2 == 0
            giusta = PAROLE_NUMERO[(max(a, b) if grande else min(a, b)) - 1]
            r = C.turno(domanda(a, b, grande))
            detto = " ".join(d["testo"] or "" for d in r["detti"]).lower()
            ok = ("il %s." % giusta) in detto
            g += ok
            r.update(coppia=[a, b], grande=grande, giusta=giusta, ok=ok)
            ris["turni"].append(r)
            print("%-8s %-36s | %s %s | letti %s" % (nome, domanda(a, b, grande), detto, "OK" if ok else "NO",
                                                     (r.get("info") or {}).get("risultato")), flush=True)
            time.sleep(0.5)
        quote[nome] = g / len(coppie)
    ris.update(Q1_quota=quote["lontani"], Q2_quota=quote["vicini"], Q1=quote["lontani"] >= 0.9, Q2=quote["vicini"] >= 0.6,
               Q3=quote["lontani"] - quote["vicini"] >= 0.10, x_reale_fine=C.T.get("/stato")["x_reale"])
    print("Q1 lontani %.0f%% %s | Q2 vicini %.0f%% %s | Q3 effetto distanza %+.0f punti %s" % (
        100 * quote["lontani"], "SI'" if ris["Q1"] else "NO", 100 * quote["vicini"], "SI'" if ris["Q2"] else "NO",
        100 * (quote["lontani"] - quote["vicini"]), "SI'" if ris["Q3"] else "NO"), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
