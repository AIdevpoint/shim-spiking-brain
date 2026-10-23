import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import prova_conversazione as C

C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_matematica.json"
SOMME = [("Quanto fa tre più due?", "cinque"), ("Uno più uno?", "due"), ("Quattro più tre?", "sette"), ("2 + 6", "otto"), ("Cinque più cinque?", "dieci"),
         ("Sei più uno?", "sette"), ("Quanto fa due più due?", "quattro"), ("Sette più due?", "nove"), ("1 + 4", "cinque"), ("Otto più due?", "dieci")]
SOTTRAZIONI = [("Quanto fa cinque meno due?", "tre"), ("Dieci meno tre?", "sette"), ("Sette meno uno?", "sei"), ("4 - 2", "due"), ("Nove meno quattro?", "cinque"),
               ("Sei meno tre?", "tre")]
PAROLE = [("Il triangolo?", "triangolo"), ("Il cerchio?", "cerchio"), ("La somma?", "somma"), ("La geometria?", "geometria"), ("Il quadrato?", "quadrato"),
          ("La frazione?", "frazione")]


def main():
    ris = {"t": time.strftime("%d/%m %H:%M"), "x_reale_inizio": C.T.get("/stato")["x_reale"], "turni": []}
    tot = {}
    for nome, lista in (("M1", SOMME), ("M2", SOTTRAZIONI)):
        g = 0
        for testo, giusta in lista:
            r = C.turno(testo)
            detto = " ".join(d["testo"] or "" for d in r["detti"]).lower()
            ok = ("fa %s" % giusta) in detto
            g += ok
            r.update(giusta=giusta, ok=ok)
            ris["turni"].append(r)
            print("%s %-28s %5s s | %s %s" % (nome, testo, r["primo_s"], detto, "OK" if ok else "NO"), flush=True)
            time.sleep(1.0)
        tot[nome] = g
    g = 0
    for testo, arg in PAROLE:
        r = C.turno(testo)
        parole = [w for d in r["detti"] for w in d["parole"]]
        ok = arg in parole
        g += ok
        r.update(argomento=arg, ok=ok)
        ris["turni"].append(r)
        print("M3 %-20s %5s s | %s %s" % (testo, r["primo_s"], " / ".join(d["testo"] or "" for d in r["detti"]), "OK" if ok else "NO"), flush=True)
        time.sleep(1.0)
    tot["M3"] = g
    ris.update(M1_giuste=tot["M1"], M2_giuste=tot["M2"], M3_giuste=tot["M3"], M1=tot["M1"] >= 7, M2=tot["M2"] >= 4, M3=tot["M3"] >= 5,
               x_reale_fine=C.T.get("/stato")["x_reale"])
    print("M1 %d/10 %s | M2 %d/6 %s | M3 %d/6 %s" % (tot["M1"], "SI'" if ris["M1"] else "NO", tot["M2"], "SI'" if ris["M2"] else "NO",
                                                   tot["M3"], "SI'" if ris["M3"] else "NO"), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
