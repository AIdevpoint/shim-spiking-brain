import json
import sys
import time
import urllib.request

A = [x for x in sys.argv[1:] if x]
PORTA = A[0] if len(A) > 0 else "8800"
GS = [float(x) for x in (A[1] if len(A) > 1 else "0,0.8,1.2,1.6,2.0").split(",")]
MODO = A[2] if len(A) > 2 else "spingi"
CAMPO = A[3] if len(A) > 3 else "g_rec"
FISSI = json.loads(A[4]) if len(A) > 4 else {}
URL = "http://127.0.0.1:%s" % PORTA


def post(p, d):
    r = urllib.request.Request(URL + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())


def get(p):
    return json.loads(urllib.request.urlopen(URL + p, timeout=60).read())


def aspetta(s):
    t0 = get("/stato")["t"]
    while get("/stato")["t"] - t0 < s:
        time.sleep(0.2)


def main():
    ris = []
    for i, g in enumerate(GS):
        parola = ["gatto", "sole", "pane", "casa", "luna", "acqua"][i % 6]
        post("/memoria", dict(FISSI, **{CAMPO: g, "svuota": True, "finestre": 5}))
        aspetta(3.0)
        prima = post("/memoria", {})
        if MODO == "senti":
            post("/memoria", {"senti": parola, "finestre": 6})
        else:
            post("/memoria", {"spingi": parola, "pa": 300.0, "s": 0.3})
        righe = {CAMPO: g, "parola": parola, "prima": prima["in_testa"][:3], "prima_zmax": prima["z_max"]}
        t_prec = 0.0
        for t in (0.5, 2.0, 5.0, 10.0, 20.0):
            aspetta(t - t_prec)
            t_prec = t
            m = post("/memoria", {})
            tenute = dict((w, z) for w, z in m["in_testa"])
            righe["%gs" % t] = {"z_parola": tenute.get(parola), "altre": len([w for w in tenute if w != parola]), "prima": m["in_testa"][:1],
                                "s_max": m.get("nmda", {}).get("s_max"), "s_accese": m.get("nmda", {}).get("s_accese")}
        ris.append(righe)
        print(json.dumps(righe, ensure_ascii=False), flush=True)
    post("/memoria", {"svuota": True})
    print("x_reale", get("/stato")["x_reale"])


if __name__ == "__main__":
    main()
