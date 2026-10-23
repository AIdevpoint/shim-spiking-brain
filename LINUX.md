# Installazione Linux e WSL2 isolata

Eseguire gli script dalla cartella del repository, anche se il percorso contiene spazi.
Richiedono Bash e `uv` gia disponibile: non installano o aggiornano uv.

```bash
./installa.sh --solo-base
```

Scarica Python 3.11 in `.python/`, crea `.venv/` con quel Python e installa
NumPy e SciPy. Il Python gestito da pyenv o dal sistema non viene usato come
base dell'ambiente. Non vengono eseguiti sudo, apt, attivazioni del venv o
modifiche a PATH, profili della shell, configurazioni pyenv e Python globali.
Il comando puo essere ripetuto senza ricreare un ambiente gia esistente.
Un ambiente che usa un interprete esterno viene rifiutato.

I percorsi sono configurati solo nel processo degli script e dei suoi figli:

| Cartella | Contenuto |
| --- | --- |
| `.python/` | Interprete Python e libreria standard |
| `.venv/` | Librerie e programmi Python del progetto |
| `.cache/` | Cache uv, Python, PyTorch, Triton e download dei modelli opzionali |
| `.tmp/` | File temporanei |
| `.local/` | Dati/configurazioni locali e elenco `pacchetti-installati.txt` |
| `.local/intel-runtime/` | Runtime Intel Level Zero e compilatore estratti localmente |
| `stato/`, `results/` | Stato e risultati dell'applicazione |

Le variabili Python che potrebbero introdurre librerie esterne vengono escluse;
il caricamento dei pacchetti utente e disabilitato. Gli script controllano che
interprete, libreria standard e percorsi Python siano dentro il repository.
Usare gli script anche per l'avvio, affinche cache e temporanei restino locali.
Le cartelle dell'ambiente e delle cache sono escluse da Git.

## GPU Intel: PyTorch XPU

La base da sola non avvia la simulazione: il motore Triton richiede una GPU,
senza backend CPU. Per Intel, dopo aver verificato i prerequisiti:

```bash
./installa.sh --xpu
./avvia.sh
```

Il comando installa le wheel PyTorch XPU dall'indice ufficiale; includono il
Triton compatibile per Intel. Non installare `triton-windows` su Linux e non
sovrascrivere Triton XPU con il pacchetto generico per CUDA.

In WSL2 il driver Windows espone la GPU tramite `/dev/dxg`; il guest Linux
richiede anche il runtime Intel Level Zero, che non e incluso nelle wheel
PyTorch. Per aggiungerlo mantenendo l'installazione dentro il repository:

```bash
./installa.sh --runtime-intel
./avvia.sh
```

Questa opzione richiede la `.venv` gia creata e PyTorch XPU installato.
Scarica pacchetti ufficiali Intel/oneAPI con versioni e SHA256 fissati nel file
`linux/installa_runtime_intel.py`, verifica i checksum e usa soltanto
`dpkg-deb --extract` per estrarli in `.local/intel-runtime/`. Non esegue
installazione dei pacchetti, script di sistema, sudo, apt o ldconfig.
La versione corrente richiede Linux x86_64 con glibc >= 2.38 e libstdc++ >= 13,
come Ubuntu 24.04; i download occupano circa 111 MiB e il runtime circa 424 MiB.
Le licenze dei componenti estratti sono conservate nelle directory `usr/share/doc`.

Gli script caricano il runtime con `LD_LIBRARY_PATH`, `LIBRARY_PATH` e
`LEVEL_ZERO_V1_SDK_PATH` soltanto nei propri processi; PATH e la shell chiamante
rimangono invariati. La distinta dei pacchetti e delle fonti si trova in
`.local/intel-runtime/manifest.json`.

Il driver Windows rimane un prerequisito esterno: gli script non lo aggiornano.
Se la verifica GPU fallisce, stampa il problema di caricamento Level Zero o il
risultato di `zeInit`. Una sessione isolata che nasconde `/dev/dxg` non puo
accedere alla GPU anche se il runtime locale e completo.

Riferimenti ufficiali:

- [PyTorch su GPU Intel](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Prerequisiti Intel e istruzioni WSL2](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html)
- [Triton per Intel incluso nelle wheel PyTorch XPU](https://github.com/intel/intel-xpu-backend-for-triton)
- [Percorsi locali per gli interpreti e le cache uv](https://docs.astral.sh/uv/reference/storage/)

## GPU NVIDIA

Con driver NVIDIA e GPU visibili in Linux/WSL:

```bash
./installa.sh --cuda
./avvia.sh
```

Questo usa l'indice CUDA 12.4 previsto dall'installer originale, con Triton
Linux fornito dalle dipendenze di PyTorch. Nessun toolkit CUDA o driver globale
viene installato. Senza opzioni, l'installer seleziona CUDA solo se
`nvidia-smi` funziona; altrimenti installa soltanto la base.

## Avvio e arresto

Il server resta in primo piano, stampa i log nel terminale e ascolta solo su
`127.0.0.1`. Aprire `http://127.0.0.1:8800/` nel browser; arrestare con Ctrl+C.
Per una porta diversa usare `./avvia.sh --porta 8802`.
Non vengono creati servizi o processi di avvio automatico.

Su Linux, quando manca un backend audio, le risposte proseguono in testo e
non viene invocato il fallback PowerShell di Windows. Per ascoltare la voce
occorre un backend compatibile (per esempio Piper e un modello locale).
Questa installazione non aggiunge pacchetti vocali o servizi esterni.

L'isolamento riguarda installazione Python, dipendenze e percorsi configurati
dagli script; non e un container o una sandbox. Driver GPU e librerie native
di base rimangono prerequisiti del sistema. Aprire il browser e un'azione
manuale e il browser mantiene normalmente i propri dati fuori dal repository.

Per rimuovere l'ambiente si possono eliminare `.venv/`, `.python/`, `.cache/`,
`.tmp/` e `.local/` nel repo. Nessuna disinstallazione di sistema e necessaria.
