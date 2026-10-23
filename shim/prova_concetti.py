import json
import sys
import time
from pathlib import Path

PORTA = sys.argv[1] if len(sys.argv) > 1 else "8800"
sys.argv = sys.argv[:2]
from shim import tara_memoria as T
from shim import prova_conversazione as C

T.URL = C.T.URL = "http://127.0.0.1:%s" % PORTA
OUT = Path(__file__).resolve().parents[1] / "results" / "prova_concetti.json"

NUOVE = [("La volpe mangia la gallina.", "Cosa mangia la volpe?", "gallina"), ("Il delfino mangia il pesce.", "Cosa mangia il delfino?", "pesce"),
         ("La zebra vive nella pianura.", "Dove vive la zebra?", "pianura"), ("Il pappagallo è verde.", "Com'è il pappagallo?", "verde"),
         ("La tigre ha le strisce.", "Cosa ha la tigre?", "strisce"), ("Il ragno è un animale.", "Cos'è il ragno?", "animale"),
         ("L'orso mangia il miele.", "Cosa mangia l'orso?", "miele"), ("Il vulcano è caldo.", "Com'è il vulcano?", "caldo"),
         ("La giraffa ha il collo.", "Cosa ha la giraffa?", "collo"), ("Il rame è rosso.", "Com'è il rame?", "rosso")]
RAGIONA = ["Il cavallo ha il pelo?", "L'aquila ha le piume?", "Lo squalo ha le pinne?", "L'ape ha le zampe?", "Il serpente ha le squame?",
           "Il leone ha il cuore?", "Il pinguino ha il becco?", "La balena ha i polmoni?", "Il delfino ha lo scheletro?", "La gallina ha le ali?"]
CONTRO = ["Il gatto è un pesce?", "L'aquila è un mammifero?", "Lo squalo è un uccello?", "L'ape è un rettile?", "La rana è un insetto?",
          "La terra è una stella?"]
COSA_E = [("Cos'è il gatto?", "mammifero"), ("Cos'è l'aquila?", "uccello"), ("Cos'è la terra?", "pianeta"), ("Cos'è il sole?", "stella"),
          ("Cos'è la mela?", "frutto"), ("Cos'è il ferro?", "metallo"), ("Cos'è il triangolo?", "figura"), ("Cos'è la formica?", "insetto"),
          ("Cos'è la rana?", "anfibio"), ("Cos'è il computer?", "macchina")]
NON_SA = ["Cos'è il governo?", "Cosa ha la speranza?", "Dove vive il pensiero?", "Cosa mangia la musica?", "Cos'è la questione?"]


def chiedi(t):
    r = C.turno(t)
    testo = " ".join(d["testo"] or "" for d in r["detti"])
    print("  %-30s -> %s   (%s s, %s)" % (t, testo, r["primo_s"], (r.get("info") or {}).get("modo")), flush=True)
    time.sleep(1.0)
    return testo, r


def main():
    ris = {"t": time.strftime("%d/%m %H:%M"), "x_reale_inizio": T.get("/stato")["x_reale"], "dettaglio": {}}
    det = ris["dettaglio"]
    k1 = 0
    det["K1"] = []
    for frase_, domanda, attesa in NUOVE:
        a, _ = chiedi(frase_)
        b, _ = chiedi(domanda)
        ok = attesa[:-1] in b.lower()
        k1 += ok
        det["K1"].append([frase_, a, domanda, b, ok])
    k2 = 0
    det["K2"] = []
    for d in RAGIONA:
        a, _ = chiedi(d)
        ok = a.lower().startswith("sì")
        k2 += ok
        det["K2"].append([d, a, ok])
    k3 = 0
    det["K3"] = []
    for d in CONTRO:
        a, _ = chiedi(d)
        ok = a.lower().startswith("no")
        k3 += ok
        det["K3"].append([d, a, ok])
    k4 = 0
    det["K4"] = []
    for d, attesa in COSA_E:
        a, _ = chiedi(d)
        ok = attesa in a.lower()
        k4 += ok
        det["K4"].append([d, a, ok])
    k5 = 0
    det["K5"] = []
    for d in NON_SA:
        a, _ = chiedi(d)
        ok = "non lo so" in a.lower() or "non conosco" in a.lower()
        k5 += ok
        det["K5"].append([d, a, ok])
    k6 = 0
    det["K6"] = []
    for testo, attesa in C.FORMULE:
        a, _ = chiedi(testo)
        ok = any(x in a.lower() for x in attesa.split("|"))
        k6 += ok
        det["K6"].append([testo, a, ok])
    ris.update(K1=k1, K2=k2, K3=k3, K4=k4, K5=k5, K6=k6,
               verdetti={"K1": k1 >= 8, "K2": k2 >= 8, "K3": k3 >= 5, "K4": k4 >= 8, "K5": k5 >= 4, "K6": k6 == len(C.FORMULE)},
               x_reale_fine=T.get("/stato")["x_reale"])
    print("K1 imparare %d/10 | K2 ragionare %d/10 | K3 contraddire %d/6 | K4 cos'e' %d/10 | K5 non so %d/5 | K6 frasi fatte %d/%d" % (
        k1, k2, k3, k4, k5, k6, len(C.FORMULE)), flush=True)
    print("verdetti:", ris["verdetti"], flush=True)
    OUT.write_text(json.dumps(ris, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
