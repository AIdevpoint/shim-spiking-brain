import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T
from shim import prova_conversazione as C

T.URL = C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_perche.json"
P1 = [("Perché il cavallo ha il pelo?", "mammifero"), ("Perché l'aquila ha le piume?", "uccello"), ("Perché lo squalo ha le pinne?", "pesce"),
      ("Perché il serpente ha le squame?", "rettile"), ("Perché la balena ha i polmoni?", "mammifero"), ("Perché la gallina ha le ali?", "uccello")]
P3 = [("Quante zampe ha l'ape?", "sei"), ("Quante zampe ha la formica?", "sei"), ("Quanti lati ha il triangolo?", "tre"),
      ("Quanti angoli ha il quadrato?", "quattro")]
P4 = ["Perché il gatto è un pesce?", "Perché la terra è una stella?"]


def chiedi(t):
    r = C.turno(t)
    a = " ".join(d["testo"] or "" for d in r["detti"])
    print("  %-34s -> %s   (%s s)" % (t, a, r["primo_s"]), flush=True)
    time.sleep(1.0)
    return a.lower()


def main():
    ris = {"t": time.strftime("%d/%m %H:%M"), "x_reale_inizio": T.get("/stato")["x_reale"], "dettaglio": []}
    p1 = sum(att in chiedi(d) for d, att in P1)
    chiedi("La tigre ha le strisce.")
    p2 = int("insegnato" in chiedi("Perché la tigre ha le strisce?"))
    p3 = sum(chiedi(d).startswith(att) for d, att in P3)
    p4 = sum(("non è vero" in a or "non lo so" in a) for a in (chiedi(d) for d in P4))
    ris.update(P1=p1, P2=p2, P3=p3, P4=p4, verdetti={"P1": p1 >= 5, "P2": p2 >= 1, "P3": p3 >= 3, "P4": p4 == 2},
               x_reale_fine=T.get("/stato")["x_reale"])
    print("P1 perche' %d/6 | P2 insegnato %d/1 | P3 quanti %d/4 | P4 non inventa %d/2 | %s" % (p1, p2, p3, p4, ris["verdetti"]), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
