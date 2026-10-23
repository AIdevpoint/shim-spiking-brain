import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T

T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_conversazione.json"
FORMULE = [("Ciao!", "ciao"), ("Ciao, come stai?", "sto|sono|mi annoio|paura"), ("Come ti chiami?", "shim"), ("Grazie!", "prego"),
           ("Cosa sai fare?", "so dire|conosco"), ("Bravo!", "grazie"), ("Come va?", "sto|sono|mi annoio|paura"), ("Arrivederci", "a presto")]
ARGOMENTI = [("Il cane?", "cane"), ("Cosa fa l'uccello?", "uccello"), ("Parlami del pesce", "pesce"), ("E il gatto?", "gatto"),
             ("La mucca?", "mucca"), ("Il bambino", "bambino"), ("Dimmi del cavallo", "cavallo"), ("E la nonna?", "nonna")]
STRANE = ["Xyzw qrtp", "Blorf snak?", "Kappa zeta omega"]


def turno(testo):
    t0 = time.time()
    r = T.post("/dialogo", {"testo": testo})
    primo, detti, da = None, [], 0
    while time.time() - t0 < 60:
        q = T.get("/pezzi?id=%d&da=%d" % (r["id"], da))
        da += len(q["pezzi"])
        for p in q["pezzi"]:
            primo = primo if primo is not None else time.time() - t0
            detti.append({"testo": p.get("frase") or p.get("testo"), "parole": p.get("parole", [])})
        if q["fine"] and not q["pezzi"]:
            break
        time.sleep(0.1)
    return {"tu": testo, "primo_s": None if primo is None else round(primo, 2), "detti": detti, "info": q.get("info")}


def main():
    ris = {"t": time.strftime("%d/%m %H:%M"), "x_reale_inizio": T.get("/stato")["x_reale"], "turni": []}
    c1g = c1t = 0
    for testo, attesa in FORMULE:
        r = turno(testo)
        tutto = " ".join(d["testo"] or "" for d in r["detti"]).lower()
        ok = any(a in tutto for a in attesa.split("|"))
        c1g += ok
        c1t += r["primo_s"] is not None and r["primo_s"] <= 4.0
        r["giusta"] = ok
        ris["turni"].append(r)
        print("C1 %-18s %5s s | %s %s" % (testo, r["primo_s"], tutto, "OK" if ok else "NO"), flush=True)
        time.sleep(1.5)
    c2g = c2t = 0
    for testo, arg in ARGOMENTI:
        r = turno(testo)
        parole = [w for d in r["detti"] for w in d["parole"]] + " ".join(d["testo"] or "" for d in r["detti"]).lower().replace(".", " ").split()
        ok = arg in parole
        c2g += ok
        c2t += r["primo_s"] is not None and r["primo_s"] <= 10.0
        r["giusta"] = ok
        ris["turni"].append(r)
        print("C2 %-18s %5s s | %s | in testa %s %s" % (testo, r["primo_s"], " / ".join(d["testo"] or "" for d in r["detti"]),
                                                        (r["info"] or {}).get("in_testa"), "OK" if ok else "NO"), flush=True)
        time.sleep(1.5)
    c3 = 0
    for testo in STRANE:
        r = turno(testo)
        tutto = " ".join(d["testo"] or "" for d in r["detti"]).lower()
        c3 += "non ho capito" in tutto
        ris["turni"].append(r)
        print("C3 %-18s %5s s | %s" % (testo, r["primo_s"], tutto), flush=True)
        time.sleep(1.5)
    c4 = sum(1 for r in ris["turni"] if r["detti"])
    ris.update(C1_giuste=c1g, C1_veloci=c1t, C2_giuste=c2g, C2_veloci=c2t, C3=c3, C4=c4, x_reale_fine=T.get("/stato")["x_reale"])
    ris["C1"] = bool(c1g >= 7 and c1t >= 7)
    ris["C2"] = bool(c2g >= 6 and c2t >= 6)
    ris["C3_ok"] = bool(c3 == 3)
    ris["C4_ok"] = bool(c4 == len(ris["turni"]))
    print("C1 %d/8 giuste, %d/8 entro 4 s %s | C2 %d/8 sull'argomento, %d/8 entro 10 s %s | C3 %d/3 %s | C4 %d/%d %s" % (
        c1g, c1t, "SI'" if ris["C1"] else "NO", c2g, c2t, "SI'" if ris["C2"] else "NO", c3, "SI'" if ris["C3_ok"] else "NO",
        c4, len(ris["turni"]), "SI'" if ris["C4_ok"] else "NO"), flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
