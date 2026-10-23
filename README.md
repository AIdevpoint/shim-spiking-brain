---
license: other
license_name: mit-and-cc-by-sa-4.0
license_link: LICENSE
language:
- it
tags:
- spiking-neural-network
- neuromorphic
- computational-neuroscience
- triton
---

# SHIM — un cervello simulato a impulsi, con cui si parla

**SHIM** è un piccolo cervello fatto di **561 251 neuroni a impulsi** e **76,7 milioni di sinapsi**, diviso in **9 aree**, che gira in
tempo quasi reale sulla scheda grafica di un portatile. Si parla con lui in una chat. **Non contiene un modello di linguaggio**: la parola
di ogni risposta esce dai suoi neuroni (vince il gruppo di neuroni che spara di più), e quello che sa sta in tabelle di conoscenza che i
neuroni leggono.

È una versione **statica**: i pesi sono congelati. Durante una conversazione può imparare quello che gli insegni, ma a ogni accensione
riparte da com'era.

> *English summary.* SHIM is a frozen, 9-area spiking neural network (561k leaky integrate-and-fire neurons, 76.7M synapses) running
> on a laptop GPU via Triton, with an Italian chat interface. No language model is involved: answer words are read out from competing
> neuron groups. It learns school-style facts during a chat, reasons in two steps over taught categories, says "I don't know" instead
> of inventing, and counts to ten. Italian only; child-level sentences.

---

## Cosa sa fare (misurato)

Ogni prova ha la previsione scritta **prima**; i verdetti, anche i NO, sono in [`PROVE_SHIM.md`](PROVE_SHIM.md). Numeri del 23/10/2026.

| Capacità | Esempio | Risultato |
|---|---|---|
| Conversazione di base | «Ciao, come stai?» → «Ciao! Sto bene, sono tranquillo.» (lo stato viene dai neuroni) | 8/8, ~1 s |
| Imparare concetti detti come a scuola | «La volpe mangia la gallina.» … «Cosa mangia la volpe?» → «La volpe mangia la gallina.» | 10/10 |
| Ragionare in due passi | «La balena ha i polmoni?» → «Sì. La balena ha i polmoni, perché è un mammifero.» (mai insegnato) | 10/10 |
| Dire no, non inventare | «Il gatto è un pesce?» → «No.»; «Cos'è il governo?» → «Non lo so.» | 6/6 e 4/5 |
| Cos'è una cosa | «Cos'è il gatto?» → «Il gatto è un mammifero.» | 10/10 |
| Spiegare il perché | «Perché il cavallo ha il pelo?» → «Perché è un mammifero: il mammifero ha il pelo.» | 6/6 |
| Eccezioni e confronti | «Il pinguino vola?» → «No. Il pinguino non vola.»; «È più grande l'elefante o il topo?» | 4/4 e 5/5 |
| Raccontare quello che sa | «Cosa sai del gatto?»; «Cosa hanno in comune l'aquila e il pinguino?» | 5/5 e 5/5 |
| Domande di seguito | «Cos'è il coniglio?» … «Cosa mangia?» | 11/11 |
| Parlare di una cosa | «Il cerchio?» → «Il cerchio? Il triangolo e il centro.» | 28/30 sull'argomento |
| Contare | «Quanto fa tre più due?» → «Tre... quattro, cinque. Fa cinque.» | somme 10/10, sottrazioni 6/6 |
| Confrontare numeri | «È più grande il sei o il nove?» → «Il nove.» | 100% lontani, 90% vicini |

**Velocità:** 0,5× il tempo reale (Intel Arc 140V integrata, portatile alla corrente). Le frasi fatte rispondono in ~1 s, le domande su
cose in 4–5 s.

## A cosa serve

- **Capire come può funzionare un cervello che parla**, guardandolo da dentro: ogni risposta si può far risalire ai neuroni che hanno
  vinto e al concetto che li ha spinti (pulsante «Neuroni» nella chat: cosa ha letto, cosa tiene in testa).
- **Insegnare come a un bambino**: gli dici «Il ragno ha otto zampe.» e poi glielo chiedi; ragiona sulle categorie che gli hai dato.
- **Non inventa**: se non sa, dice «Non lo so». Nessuna risposta "plausibile ma falsa" come quelle di un modello di linguaggio.
- **Gira in locale**, senza internet e senza servizi esterni, su un portatile.
- Una base per la ricerca: neuroni a impulsi, memoria di lavoro con una parte lenta (NMDA), numeri su una linea mentale, tutto misurabile.

## Cosa NON è

- **Non è un assistente come ChatGPT.** Conosce circa **2 500 parole** italiane, parla con frasi brevi da bambino, costruite su schemi.
- Il ragionamento vale solo su quello che gli è stato insegnato nella forma «X è / ha / fa Y».
- La conoscenza sta in **tabelle lette dai neuroni**, non nelle sinapsi come in un cervello vero. È il limite più grande.
- **Regole scritte a mano (dichiarate):** lo schema che capisce le frasi di scuola, le frasi fatte (saluti, grazie, il tuo nome), le parole
  piccole delle frasi (articoli, preposizioni), la filastrocca dei numeri, e il filtro della memoria di lavoro (vedi `PROVE_SHIM.md`).
- Solo italiano. Non legge testi lunghi e non scrive testi.

---

## Com'è fatto

### Le 9 aree

| Area | Neuroni | Quota | Gruppi con un compito |
|---|---:|---:|---|
| Vista | 142 001 | 25,3% | `v1_sx`, `v1_dx` (l'occhio: retina → corteccia visiva primaria) |
| Prefrontale | 114 502 | 20,4% | `pf_sx`, `pf_dx` (memoria di lavoro: 1 000 assemblee da 40 neuroni) |
| Linguaggio | 77 958 | 13,9% | `parole` (2 486 parole, 20 neuroni ciascuna), `bocca` (le sillabe) |
| Corteccia (motoria, tatto, resto) | 70 862 | 12,6% | — |
| Memoria | 47 283 | 8,4% | — |
| Parietale | 39 620 | 7,1% | `ips` (i numeri da 1 a 10: 10 gruppi da 60 neuroni) |
| Temporale | 31 289 | 5,6% | — |
| Udito | 29 720 | 5,3% | `udito_1` … `udito_10` (l'orecchio: 24 bande di frequenza) |
| Profondo (sotto la corteccia) | 8 016 | 1,4% | `amigdala`, `accumbens`, `talamo` (le emozioni lette dai neuroni) |
| **Totale** | **561 251** | | 27 gruppi |

La struttura è tutta in [`shim_mappa.json`](shim_mappa.json): le aree, i gruppi e una tabella 9 × 9 con la probabilità che una
sinapsi lunga parta da un'area e arrivi in un'altra (ricavata dalla trattografia dello Human Connectome Project e **sommata per area**).

### I neuroni e le sinapsi

- **Integra-e-spara** (costante di tempo 10 ms, capacità 250 pF), passo di 0,1 ms; 80% eccitatori, 20% inibitori.
- **~134 sinapsi per neurone**: metà dentro il proprio gruppo, metà lunghe (solo dagli eccitatori) verso le altre aree secondo la tabella;
  ritardi 0,75–1,5 ms dentro il gruppo, ~5 ms fra le aree. L'inibizione totale è 1,25 volte l'eccitazione (rete asincrona).
- Sinapsi aggiunte: memoria di lavoro 1,70 milioni (assemblee nella prefrontale, con una parte lenta tipo NMDA), numeri 16 200.
- **Taratura**: all'accensione la corrente di ogni popolazione si regola finché gli eccitatori sparano ~3 Hz e gli inibitori ~6 Hz.
- Il motore è scritto in **Triton** e gira sulla scheda grafica.

### Le conoscenze (in `pesi/`)

- **Lessico**: 2 486 parole (le più usate dell'italiano parlato, la scuola, la matematica), con i suoni di 300 parole imparati.
- **Associazioni** fra le parole lette nella Wikipedia italiana e nelle storie di Wikisource.
- **Concetti**: un libro di scuola di 362 frasi, 53 definizioni dalla Wikipedia, 147 fatti da Wikidata («è un», «ha»).

---

## Come si usa

**Linux / WSL2:** gli script `installa.sh` e `avvia.sh` usano un Python 3.11
locale al repository e un ambiente `.venv` isolato. Vedi [LINUX.md](LINUX.md)
per l'installazione e i prerequisiti GPU (Intel XPU o NVIDIA CUDA).

**Serve:** Windows, Python 3.11, una scheda grafica **Intel Arc** (provato: Arc 140V integrata, 32 GB di RAM) o **NVIDIA** (non ancora
provato), qualche GB libero su disco (PyTorch è grande; SHIM pesa ~95 MB).

1. `INSTALLA_SHIM.bat` (una volta): crea l'ambiente e installa PyTorch, Triton e il resto.
2. `AVVIA_SHIM.bat`: accende il cervello e apre la chat su `http://127.0.0.1:8800/`. La prima accensione dura 1–3 minuti.
3. Scrivi. Prova a insegnargli qualcosa e poi chiederglielo.

Su Intel, Triton compila col compilatore di **Visual Studio 2022 Build Tools** e con l'SDK **level-zero** (indicato con la variabile
`SHIM_LEVEL_ZERO`). Tieni il computer **alla corrente**: a batteria la scheda va molto più lenta. Le prove si rifanno a SHIM acceso con
`python -m shim.prova_concetti 8800` (e le altre in `PROVE_SHIM.md`).

La voce è facoltativa (pulsante «Voce» nella chat): usa Piper se installato (`pip install piper-tts` e una voce italiana in `shim/voci/`),
altrimenti la voce di Windows.

## File

| | |
|---|---|
| `shim/` | il codice (Python + Triton) e la chat; `python -m shim` lo avvia |
| `shim_mappa.json` | la struttura del cervello (9 aree, 27 gruppi) |
| `pesi/` | lessico, associazioni, concetti, taratura (congelati) |
| `results/` | i risultati delle prove |
| `PROVE_SHIM.md` | previsioni e verdetti |
| `ATTRIBUZIONI.md` | da dove vengono i dati e con che licenza |
| `LICENSE` | MIT per il codice, CC BY-SA 4.0 per i pesi |

## Licenza

- **Codice** (`shim/`, i file `.bat`, `shim_mappa.json`): **MIT**.
- **Pesi e conoscenze** (`pesi/`): **CC BY-SA 4.0**, perché derivano in parte da Wikipedia e Wikisource; le fonti sono in
  [`ATTRIBUZIONI.md`](ATTRIBUZIONI.md).

Testo completo in [`LICENSE`](LICENSE).
