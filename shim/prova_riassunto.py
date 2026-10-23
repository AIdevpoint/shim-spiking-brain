import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T
from shim import prova_conversazione as C

T.URL = C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_riassunto.json"
R1 = [("gatto", "mammifero"), ("aquila", "uccello"), ("squalo", "pesce"), ("ape", "insetto"), ("rana", "anfibio")]
R2 = [("il gatto", "il cane", "mammiferi"), ("l'aquila", "il pinguino", "uccelli"), ("lo squalo", "la balena", "animali"),
      ("la mela", "la pera", "frutti"), ("l'ape", "la formica", "insetti")]


def chiedi(t):
    r = C.turno(t)
    a = " ".join(d["testo"] or "" for d in r["detti"])
    print("  %-50s -> %s   (%s s)" % (t, a, r["primo_s"]), flush=True)
    time.sleep(1.0)
    return a


def main():
    ris = {"t": time.strftime("%d/%m %H:%M")}
    r1 = 0
    for w, cat in R1:
        a = chiedi("Cosa sai del %s?" % w if w[0] not in "aeiou" else "Cosa sai dell'%s?" % w)
        r1 += a.count(".") >= 2 and cat in a.lower()
    r2 = sum(att in chiedi("Cosa hanno in comune %s e %s?" % (x, y)).lower() for x, y, att in R2)
    a3 = chiedi("Parlami del lavoro").lower()
    r3 = int("lavoro" in a3 and "non lo so" not in a3)
    ris.update(R1=r1, R2=r2, R3=r3, verdetti={"R1": r1 >= 4, "R2": r2 >= 4, "R3": r3 == 1})
    print("R1 cosa sai %d/5 | R2 in comune %d/5 | R3 cose legate %d/1 | %s" % (r1, r2, r3, ris["verdetti"]), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
