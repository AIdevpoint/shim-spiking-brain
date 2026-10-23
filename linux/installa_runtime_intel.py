"""Estrae il runtime Intel ufficiale nel repo, senza installare pacchetti di sistema."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".local" / "intel-runtime"
CACHE = ROOT / ".cache" / "intel-runtime"
VERSION = "26.35.39758.10"
# Release ufficiali e SHA256 pubblicati dai rispettivi progetti GitHub.
PACKAGES = [
    ("intel/compute-runtime", VERSION, "libze-intel-gpu1_26.35.39758.10-0_amd64.deb",
     "c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470"),
    ("intel/compute-runtime", VERSION, "libigdgmm12_22.10.0_amd64.deb",
     "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54"),
    ("intel/compute-runtime", VERSION, "intel-ocloc_26.35.39758.10-0_amd64.deb",
     "c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7"),
    ("intel/intel-graphics-compiler", "v2.41.5", "intel-igc-core-2_2.41.5+22716_amd64.deb",
     "0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628"),
    ("intel/intel-graphics-compiler", "v2.41.5", "intel-igc-opencl-2_2.41.5+22716_amd64.deb",
     "779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058"),
    ("oneapi-src/level-zero", "v1.34.0", "libze1_1.34.0+u24.04_amd64.deb",
     "45210e4549cd965ad7b9f160eefe53cbcdba01af341bea7ac7ba0aedeeb613ba"),
    ("oneapi-src/level-zero", "v1.34.0", "libze-dev_1.34.0+u24.04_amd64.deb",
     "2813bb3b06d0d8c0f3c771051defabdd74da2e454661cd654ebabcfb5ab21c01"),
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Questo runtime richiede Linux x86_64 (Ubuntu 24.04 o compatibile).")
    if not shutil.which("dpkg-deb"):
        raise SystemExit("dpkg-deb non trovato; nessun pacchetto di sistema viene installato automaticamente.")
    if DEST.exists():
        manifest = DEST / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["version"] == VERSION:
            print(f"Runtime Intel locale gia presente: {DEST}")
            return
        raise SystemExit(f"La destinazione {DEST} esiste gia con contenuto diverso; non la sovrascrivo.")
    CACHE.mkdir(parents=True, exist_ok=True)
    (ROOT / ".tmp").mkdir(exist_ok=True)
    DEST.parent.mkdir(exist_ok=True)
    manifest_packages = []
    with tempfile.TemporaryDirectory(prefix="intel-runtime-", dir=ROOT / ".tmp") as stage_name:
        stage = Path(stage_name) / "runtime"
        stage.mkdir()
        for repo, tag, name, digest in PACKAGES:
            path = CACHE / name
            url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
            if not path.exists() or sha256(path) != digest:
                print(f"Scarico {name}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "SHIM-local-runtime"})
                partial = path.with_suffix(".download")
                try:
                    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    if sha256(partial) != digest:
                        raise RuntimeError(f"SHA256 non valido: {name}")
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
            print(f"Verificato SHA256; estraggo {name}", flush=True)
            # Solo estrazione dell'archivio dati: niente dpkg -i o maintainer script.
            subprocess.run(["dpkg-deb", "--extract", str(path), str(stage)], check=True)
            manifest_packages.append({"file": name, "url": url, "sha256": digest})
        (stage / "manifest.json").write_text(json.dumps({
            "version": VERSION, "packages": manifest_packages,
        }, indent=2) + "\n")
        stage.rename(DEST)
    print(f"Runtime Intel estratto in {DEST}. Nessuna installazione di sistema.")


if __name__ == "__main__":
    main()
