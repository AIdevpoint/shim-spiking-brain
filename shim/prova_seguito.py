import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T
from shim import prova_conversazione as C

T.URL = C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_seguito.json"
S1 = [("Cos'è il gatto?", "E il cane?", "mammifero"), ("Cos'è l'aquila?", "E il pinguino?", "uccello"), ("Dove vive la balena?", "E il cammello?", "deserto"),
      ("Cosa mangia il gatto?", "E il coniglio?", "carota"), ("Il cavallo ha il pelo?", "E il leone?", "sì")]
S2 = [("Cos'è il coniglio?", "Cosa mangia?", "carota"), ("Cos'è la balena?", "Dove vive?", "mare"), ("Cos'è il triangolo?", "Cosa ha?", "lati"),
      ("Cos'è la neve?", "Com'è?", "fredda")]
S3 = [("L'aquila ha le piume?", "E perché?", "uccello"), ("La balena ha i polmoni?", "E perché?", "mammifero")]


def chiedi(t):
    r = C.turno(t)
    a = " ".join(d["testo"] or "" for d in r["detti"])
    print("  %-26s -> %s   (%s s)" % (t, a, r["primo_s"]), flush=True)
    time.sleep(1.0)
    return a.lower()


def giro(coppie):
    g = 0
    for prima, dopo, attesa in coppie:
        chiedi(prima)
        g += attesa in chiedi(dopo)
    return g


def main():
    ris = {"t": time.strftime("%d/%m %H:%M")}
    s1, s2, s3 = giro(S1), giro(S2), giro(S3)
    ris.update(S1=s1, S2=s2, S3=s3, verdetti={"S1": s1 >= 4, "S2": s2 >= 3, "S3": s3 >= 1})
    print("S1 %d/5 | S2 %d/4 | S3 %d/2 | %s" % (s1, s2, s3, ris["verdetti"]), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
