"""SHIM: python -m shim [--porta 8800]"""
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
PESI = RADICE / "pesi"
STATO = RADICE / "stato"

CONFIG = {
    "PYTHONIOENCODING": "utf-8",
    "SHIM_PASSI": "5",
    "SHIM_INTERI": "1",
    "SHIM_BLOCCO": "256",
    "SHIM_WARP": "8",
    "SHIM_SOLO_CERVELLO": "1",
    "SHIM_FRASI_SVO": "1",
    "SHIM_MEMORIA": "1",
    "SHIM_PENSA_IN_SILENZIO": "1",
    "SHIM_CONOSCENZA": "2000",
    "SHIM_M_CODICE": "20",
    "SHIM_CODICI_SEPARATI": "1000",
    "SHIM_MEM_XREL": "1",
    "SHIM_MEM_KFF": "20",
    "SHIM_MEM_LEGGI_S": "1",
    "SHIM_MEM_N": "1000",
    "SHIM_NUMERI": "1",
    "SHIM_NUM_TIENI": "0",
    "SHIM_LEGGI_FORZA": "2",
    "SHIM_CONCETTI": "1",
    "SHIM_PAGINA": "shim",
}
MEMORIA = {"g_rec": 0.8, "g_ff": 2, "g_nmda": 300, "inib": 6, "x_nmda": 9, "alfa": 0.5}


def prepara_stato():
    STATO.mkdir(exist_ok=True)
    for f in PESI.iterdir():
        if f.is_file():
            shutil.copy2(f, STATO / f.name)
    (STATO / "tarature").mkdir(exist_ok=True)
    for f in (PESI / "tarature").glob("*.npy"):
        if not (STATO / "tarature" / f.name).exists():
            shutil.copy2(f, STATO / "tarature" / f.name)


def memoria_dopo_il_risveglio(porta):
    url = "http://127.0.0.1:%d" % porta
    while True:
        try:
            urllib.request.urlopen(url + "/stato", timeout=3).read()
            break
        except Exception:
            time.sleep(2)
    for corpo in (dict(MEMORIA, svuota=True, finestre=10), MEMORIA):
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/memoria", data=json.dumps(corpo).encode(), method="POST"), timeout=30).read()
        except Exception as e:
            print("[SHIM] memoria di lavoro non impostata: %s" % e, flush=True)
    print("[SHIM] pronto: apri %s/" % url, flush=True)


def main():
    porta = int(sys.argv[sys.argv.index("--porta") + 1]) if "--porta" in sys.argv else 8800
    prepara_stato()
    for k, v in CONFIG.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("SHIM_LESSICO", str(STATO / "shim_lessico_2000.npz"))
    sys.argv = [sys.argv[0], "--porta", str(porta)]
    threading.Thread(target=memoria_dopo_il_risveglio, args=(porta,), daemon=True).start()
    from shim import server
    server.main()


if __name__ == "__main__":
    main()
