import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T
from shim import prova_conversazione as C

T.URL = C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_eccezioni.json"
E1 = [("Il pinguino vola?", "no"), ("La gallina vola?", "no"), ("L'aquila vola?", "sì"), ("Il pappagallo vola?", "sì")]
E3 = [("È più grande l'elefante o il topo?", "elefante"), ("Chi è più veloce, il cavallo o la tartaruga?", "cavallo"),
      ("È più piccola la formica o la balena?", "formica"), ("È più alta la montagna o la collina?", "montagna"),
      ("È più pesante l'elefante o la piuma?", "elefante")]


def chiedi(t):
    r = C.turno(t)
    a = " ".join(d["testo"] or "" for d in r["detti"])
    print("  %-46s -> %s   (%s s)" % (t, a, r["primo_s"]), flush=True)
    time.sleep(1.0)
    return a.lower()


def main():
    ris = {"t": time.strftime("%d/%m %H:%M")}
    e1 = sum(chiedi(d).startswith(att) for d, att in E1)
    chiedi("Il cane non nuota.")
    e2 = int(chiedi("Il cane nuota?").startswith("no"))
    e3 = sum(att in chiedi(d).split(".")[0] for d, att in E3)
    e4 = int("non lo so" in chiedi("È più grande la musica o il governo?"))
    ris.update(E1=e1, E2=e2, E3=e3, E4=e4, verdetti={"E1": e1 >= 3, "E2": e2 >= 1, "E3": e3 >= 4, "E4": e4 >= 1})
    print("E1 eccezioni %d/4 | E2 insegnata %d/1 | E3 confronti %d/5 | E4 non so %d/1 | %s" % (e1, e2, e3, e4, ris["verdetti"]), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
