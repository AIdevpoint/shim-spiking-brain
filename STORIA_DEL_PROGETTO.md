# CLM — la storia del progetto, Pepito e HIM

*Scritto il 23 ottobre 2026, sera. Ogni numero viene da una prova fatta nel progetto: i file sono indicati accanto. Le prove hanno sempre la
previsione scritta PRIMA di lanciarle; quando una prova è andata male è scritto "NO".*

---

## In una frase

CLM costruisce **cervelli simulati fatti di neuroni a impulsi** (ogni neurone ha un potenziale elettrico che sale, spara, e manda l'impulso ad
altri neuroni con un ritardo), **cablati da dati veri** del cervello di un animale o dell'uomo, li mette **in un corpo e in un mondo**, e li fa
**vivere e imparare in tempo reale** su un portatile. Non è un modello di linguaggio come ChatGPT e non è una rete neurale "classica".

**Il record che cerchiamo** non è il cervello più grande (quelli da miliardi di neuroni sono pezzi di corteccia copiati e incollati), ma il
**cervello fatto giusto**: cablato da misure vere, in un corpo, che impara mentre vive, in tempo reale — come la mosca di FlyWire, ma più avanti.

---

## 1. La storia

### Agosto – settembre 2026: il verme e il moscerino
- **C. elegans** (302 neuroni, il connettoma vero): versioni da 0.01 a 0.10. Simulatore, prima validazione biologica (il tocco accende gli
  interneuroni giusti, 6/6, come Chalfie 1985), muscoli (552 giunzioni, 95 muscoli), abituazione, andatura, un corpo in un mondo.
  Documenti: `CLM_C_elegans_v0.07.md` … `v0.10.md`, `CLM_REVISIONE.md` (23 agosto: "siamo alla 0.08, non alla 0.09").
- **Drosophila** (moscerino) e poi **vespa, formica e colonia** (versioni 0.1–0.3, `CLM_Drosophila_v0.*.md`): il motore diventa 3,2 volte più
  veloce e dimostrato esatto; si scopre perché un cervello generato da mezzo milione di neuroni non resta organizzato.
- La regola del progetto nasce qui: **non si passa al livello dopo finché il precedente non è validato**, e **ogni prova ha la previsione scritta
  prima**.

### 4 ottobre 2026: nasce Pepito
Pepito è il gatto vero di Alessandro. Il progetto diventa: **dare a un gatto virtuale il cervello di un gatto vero**.
- 4–7 ottobre: il cervello del gatto (aree del gatto di Scannell 1999, colonne di corteccia di Potjans-Diesmann), il motore sulla scheda grafica
  (Triton, Intel Arc), la stanza 3D, i sensi.
- 10–11 ottobre: Pepito guida la **macchinina vera** (Arduino, sensore a ultrasuoni, telefono come occhi), la **casa finta** per le prove.
  Tempo reale: 104 784 neuroni a 1,03× il tempo reale imparando.
- 12–13 ottobre: tipi di neuroni, **sonno**, stanchezza, legame col padrone, memoria dei posti, **cervelletto**, **chimica** (nuclei veri del
  sonno e dello stress).
- 16–17 ottobre: neofobia, gioco col filo, pupille e coda, **sonno REM e sogni**, morso da troppe carezze, sete, verifica della costruzione.
- 18–22 ottobre: le slide per il talk di **Python Pescara** (22/10) — il talk poi **non c'è stato** (proposta non accettata in tempo). Forse il
  26 novembre o a dicembre.

### 18 ottobre 2026: nasce HIM
HIM è un **cervello umano** simulato: le 374 zone della mappa dello Human Connectome Project. L'obiettivo di Alessandro: "un cervello umano
che sa un sacco di roba, ma è nel cervello, e parla lui su quello che sa" — **senza** un modello di linguaggio dentro.
- 18–19 ottobre: costruzione, taglie da 100 000 a 1 000 000 di neuroni, sente (coclea), parla (voce letta dalle zone della parola), avatar.
- 20–21 ottobre: impara parole dal suono, legge la Wikipedia italiana intera (20 007 parole associate), pensa catene di parole, dice "non so".
- 22 ottobre: memoria di lavoro, frasi, relazioni ("capitale di…"), primo dialogo; 1 milione di neuroni provato (0,33× il tempo reale).
- 23 ottobre: 2 486 parole, matematica di base, "quanti sono?", **concetti insegnati come a scuola e ragionamento**, conoscenza da Wikidata,
  **orecchio più grande**, l'aula in realtà virtuale. (Dettagli nella sezione 3.)

---

## 2. Pepito, il gatto

### Il cervello
- **114 100 neuroni e 116 milioni di sinapsi** (configurazione del visore, con cervelletto), in **tempo reale** sul portatile (1,0× alla
  corrente). Verificato il 17/10 (`clm/gatto/verifica_costruzione.py`): legge di Dale, strati come Potjans-Diesmann (0,02% di scarto), vie fra
  le aree identiche ai dati di Scannell, ~1000 sinapsi per neurone dentro le aree.
- Parti: 53 aree di corteccia e 42 nuclei del talamo del gatto, gangli della base, amigdala, cervelletto (~9 000 neuroni, impara a chiudere
  l'occhio prima del soffio), chimica (VLPO, locus coeruleus, rafe, orexina, ossitocina, CRF), sonno con REM e sogni (onde PGO che rigiocano
  ricordi visivi).
- File: `clm/gatto/cervello_gatto.py`, `cervello.py` (la Mente), `cervelletto.py`, `chimica.py`, `rem.py`, `kernel_triton.py` (il motore).

### Cosa fa (tutte cose provate, ognuna con la sua prova in `results/`)
- Vede, sente, annusa, sente le carezze sui baffi e sul corpo; impara il suo nome **da qualunque voce** e viene quando lo chiami (6–8 su 7–8).
- Gioca (con la mano, col filo e la piuma: agguato, balzo, lotta), caccia un topo lanciato (8/9), mangia, beve, dorme dove vuole (sul divano),
  si stanca, si stira, si lava.
- Si spaventa per un suono nuovo (coda gonfia, pupille dilatate) e si abitua; dopo troppe carezze morde; dopo uno spavento va su un mobile alto.
- Si ricorda dove ha mangiato e dormito e ci ritorna girando intorno ai mobili; la sua memoria lunga (sinapsi imparate) si salva ogni 2 minuti.
- Vive nel **visore Meta Quest 3S** in realtà mista, nella casa vera, e può guidare la **macchinina** vera.

### Come si avvia
`AVVIA_PEPITO_LIBERO.bat grande` (sul PC), `QUEST_PEPITO.bat` (nel visore), `PRESENTAZIONE_*.bat` (la demo del talk). Porta 8791.
**Pepito e HIM non possono girare insieme** sulla stessa scheda.

### Cosa manca
Il gatto non ha un connettoma a livello di singoli neuroni (esiste solo quello fra aree): il cablaggio dentro le aree è statistico. Alcune scelte
sono regole dichiarate. Le prove sulla macchinina e nel visore vero sono state poche.

---

## 3. HIM, il cervello umano

### Il cervello
- **561 251 neuroni, 76,7 milioni di sinapsi**, 374 zone (360 di corteccia di Glasser + 14 sotto la corteccia) collegate come nella mappa
  dello Human Connectome Project (`data/umano/`). Gira a **~0,5–0,6× il tempo reale** sul portatile.
- Zone ingrandite dove serve (dichiarato in `AVVIA_HIM_2000.bat`): vista ×4,5, linguaggio ×6–12, prefrontale ×2, memoria ×3, parietale e
  temporale ×1,5, **udito ×4** (dal 23/10 sera).
- Sensi: una **coclea** simulata che manda il suono alla corteccia uditiva; un **occhio** (retina + V1) che riceve immagini o la webcam.
- Voce: le parole che escono dai neuroni delle zone del linguaggio vengono pronunciate da una voce sintetica (Piper).
- File: `clm/umano/cervello_umano.py`, `him.py` (il server), `parole_him.py` (il lessico), `kernel_him.py` (il motore).

### Cosa sa fare (misurato il 23/10, col cervello acceso)
| Capacità | Esempio | Risultato |
|---|---|---|
| Conversazione di base | "Ciao, come stai?" → "Ciao! Sto bene, sono tranquillo." (lo stato viene dai neuroni) | 8/8, risposta subito |
| Parlare di cose | "Il medico?" → "Il paziente e l'ospedale." | 11/12 sull'argomento; ~21/30 sensate |
| Imparare concetti | "La volpe mangia la gallina." → "Cosa mangia la volpe?" → "La volpe mangia la gallina." | 10/10 |
| Ragionare in due passi | "La balena ha i polmoni?" → "Sì. … perché è un mammifero." (mai insegnato) | 10/10 |
| Spiegare il perché | "Perché il cavallo ha il pelo?" → "Perché è un mammifero: il mammifero ha il pelo." | 6/6 |
| Contare dalle conoscenze | "Quante zampe ha la formica?" → "Sei." | 4/4 |
| Dire no, non inventare | "Il gatto è un pesce?" → "No. Il gatto è un mammifero."; "Cos'è il governo?" → "Non lo so." | 13/13 |
| Domande di seguito | "Cos'è il coniglio?" … "Cosa mangia?" → "Il coniglio mangia la carota." | 11/11 |
| Matematica | "Quanto fa tre più due?" → "Tre... quattro, cinque. Fa cinque." | somme 10/10, sottrazioni 6/6 |
| Confrontare numeri | "È più grande il sei o il nove?" → "Il nove." (dalla grandezza letta nel parietale) | 10/10 lontani, 9/10 vicini |
| Quanti sono (puntini) | gli mostri 4 puntini → "Sono quattro." | 75% esatti, 97% entro uno |
| Capire parole dal suono | 60 parole, varianti mai sentite | 66% → **83% con l'udito ×4** |
| … con 300 parole imparate | le stesse domande di prima | 44% → **57%** (12 parole vecchie: 25% → 58%) |
| Eccezioni | "Il pinguino vola?" → "No. Il pinguino non vola." (gli uccelli volano, ma il pinguino no) | 5/5 |
| Confronti | "È più grande l'elefante o il topo?" → "L'elefante. L'elefante è grande e il topo è piccolo." | 5/5 |
| Immagini insegnate a voce | "Questo è un triangolo." … "Cos'è questo?" → "È un triangolo." | 9/10 (le 6 di prima 5/6) |
| Raccontare quello che sa | "Cosa sai del gatto?" → "Il gatto è un mammifero. Ha la coda e i baffi. Mangia il topo. Miagola." | 5/5 |
| Cosa hanno in comune | "Cosa hanno in comune l'aquila e il pinguino?" → "Sono tutti e due uccelli. Hanno le ali e le piume." | 5/5 |
| Ricordarsi di te | "Mi chiamo Alessandro" → "Piacere, Alessandro!"; "Come mi chiamo?" | sì |

File delle prove: `clm/umano/prova_concetti.py`, `prova_perche.py`, `prova_seguito.py`, `prova_legate.py`, `prova_lessico_2000.py`,
`prova_conversazione.py`, `prova_matematica.py`, `prova_confronto.py`, `prova_quanti.py`, `prova_orecchio.py`; risultati in `results/`.

**Velocità delle risposte:** circa 2 secondi (media 1,8 s), dal 23/10 sera: prima il cervello, che va a metà del tempo reale, finiva di
riascoltare la propria voce della risposta precedente prima di rispondere (fino a 12 s); ora chi gli parla lo interrompe.

### Cosa sa (la conoscenza)
- **2 486 parole**: le 762 della scuola e delle prime voci della Wikipedia, le più usate dell'italiano parlato (sottotitoli OpenSubtitles),
  verbi, matematica, e 169 parole della scuola (animali, piante, corpo, cielo, materiali, informatica).
- **Associazioni** fra le parole lette nella **Wikipedia italiana intera** e nelle **storie di Wikisource** (Pinocchio, Cuore, fiabe…).
- **Concetti** (`concetti_him.py`): un **libro di scuola** di 362 frasi scritte da Claude (dichiarato) in tutti i campi; **53 definizioni** dalla
  prima frase delle voci della Wikipedia; **107 "è un" e 40 "ha"** da **Wikidata** (filtrati). E tutto quello che gli insegni tu parlando.

### Come funziona il "pensare" (onestamente)
- Le **parole escono dai neuroni**: ogni parola ha un gruppo di neuroni nelle zone del linguaggio; le conoscenze danno una "spinta" ai gruppi
  giusti e vince il gruppo che spara di più. Per questo può sbagliare, e il "sì/no" dipende da quale parola esce davvero.
- Il **ragionamento a due passi** è la diffusione su una rete di concetti (Collins e Quillian, 1969): quello che vale per la categoria vale per
  chi ci sta dentro.
- **Regole scritte a mano (dichiarate)**: lo schema che capisce le frasi della scuola, le frasi fatte (saluti, grazie, il tuo nome, "sono
  stanco"), le parole piccole delle frasi (articoli e preposizioni), la filastrocca dei numeri.
- **Dal 24/10 la conoscenza sta nelle SINAPSI**: ogni legame imparato ("il gatto è un mammifero") diventa 400 sinapsi fra i 20 neuroni
  di una parola e i 20 dell'altra (478 legami, ~190 000 sinapsi); la domanda apre il "cancello" del tipo di legame (dichiarato) e la
  risposta corre negli impulsi: gatto → mammifero → pelo. Con le sinapsi tutte le prove dei concetti, delle eccezioni e delle domande di
  seguito sono SÌ (4 tentativi, dichiarati). "Perché" e "in comune" usano ancora la spinta calcolata dalla tabella (con le sinapsi
  andavano peggio: le parole collegate a tante altre, come "mammifero", si indeboliscono troppo).

### Come si usa
- `AVVIA_HIM_2000.bat`: accende HIM e apre **l'aula**, il mondo in realtà virtuale (lavagna, HIM in piedi, un banco con oggetti). Gli parli
  col microfono o scrivi; gli mostri gli oggetti sul banco; gli insegni parole con la tua voce e concetti a voce.
- `QUEST_HIM.bat`: la stessa aula nel visore Meta Quest (grilletto destro = parla, tasto laterale = prendi gli oggetti, menu sulla mano sinistra).
  **Non ancora provata col visore vero.**
- `http://127.0.0.1:8800/chat`: la chat leggera (senza messaggi preimpostati: i pulsanti mostrano solo l'immagine, la domanda la fai tu).
- Cose da insegnargli come a scuola: "Il gatto è un mammifero.", "I mammiferi hanno il pelo.", "Il ragno ha otto zampe.", poi chiedi
  "Cos'è…?", "Cosa ha…?", "Perché…?", "Quante…?". Con un'immagine: "Questo è un gatto." e poi "Cos'è questo?".

### Cosa NON sa fare
- **Non parla e non ragiona come noi.** Le frasi sono da bambino piccolo, costruite su schemi; il ragionamento vale solo su quello che gli è stato
  insegnato in forma "X è / ha / fa Y".
- Capire la voce di una persona mai sentita dal **suono** con centinaia di parole: migliorato con l'orecchio più grande, ma non ancora affidabile
  (nella chat e nell'aula la frase la scrive un riconoscitore — Edge o Whisper — e HIM la legge; dichiarato).
- Memoria di lavoro: tiene in testa le parole lette solo per le prime 1 000 parole (con quelle nuove 0 su 20); il "ripasso" l'ha peggiorata
  (provato e scartato).
- Le frasi coi verbi da sole ("chi fa cosa") dai testi: al massimo ~4 su 10 buone.

---

## 4. Velocità e grandezza (misurato)

| Cervello | Neuroni | Velocità sul portatile |
|---|---|---|
| Pepito (visore) | 114 100 | 1,0× (tempo reale) |
| HIM di oggi | 561 251 | 0,5–0,6× |
| HIM da 1 milione (22/10) | 996 778 | 0,33× |
| Gatto da 1 milione su scheda A40 a noleggio | 1 004 154 | 0,30× |

Portatile: Intel Core Ultra 7 258V, scheda Intel Arc 140V, 32 GB. **Le misure valgono solo alla corrente** (a batteria la scheda va ~2,5 volte
più lenta). Dettagli: `results/scala_him_2026-10-21.txt`, `UN_MILIONE.md`.

**La lezione più importante del 23/10:** più neuroni messi a caso non rendono il cervello più capace; **messi nel posto giusto sì** — l'udito ×4
(+22 000 neuroni) ha portato il riconoscimento delle parole dal 66% all'83% alla stessa velocità.

---

## 5. Il metodo (perché ci si può fidare dei numeri)
- **Previsione scritta prima** di ogni prova, nel file della prova; poi il verdetto (SÌ / NO) coi numeri, anche quando è NO.
- Ogni scelta non misurata è scritta **DICHIARATO** nel codice.
- Le prove si rifanno: ogni numero ha il suo script.
- Errori trovati e dichiarati lungo la strada (alcuni): un lessico imparato su un cervello diverso non vale per un altro; le costanti di tempo
  vanno nel tempo del cervello, non dell'orologio; controllare sempre la taratura prima di fidarsi di una prova; mai due cervelli insieme sulla scheda.

---

## 6. Prossimi passi
1. **Ingrandire per scopo, misurando**: memoria di lavoro (prefrontale) perché tenga anche le parole nuove; vista con più risoluzione.
2. **Velocizzare il motore** (le zone a riposo aggiornate meno spesso, il "rumore di fondo" più leggero), per pagare meno ogni neurone.
3. **Il milione a velocità reale**: solo con una scheda più potente (a noleggio ~1–2 $ l'ora, o comprata). Decide Alessandro.
4. **Il salto vero**, cominciato il 24/10: la conoscenza nelle sinapsi (fatto per i concetti). Prossimo: che le sinapsi si formino da sole
   con l'apprendimento (plasticità), non costruite dal legame insegnato; e reggano i nodi collegati a tanti.
5. Provare l'aula e Pepito nel visore vero; rifare le slide per il talk di novembre/dicembre.

---

## 7. Dove sono le cose
- Diari dettagliati: `PENSIERO_HIM.md` (HIM, tutte le prove), `RIPRENDERE.md` e `COME_FUNZIONA_CLM.md` (Pepito), `ROADMAP.md` (le fasi),
  `CLM.md` (istruzioni per chi lavora al progetto), `UN_MILIONE.md`, `SHIM.md`.
- Codice: `clm/gatto/` (Pepito), `clm/umano/` (HIM), `clm/ai/`, `clm/bio/`, `clm/app/` (verme, mosca, vespa, formiche).
- Dati imparati (fuori dal progetto): `%LOCALAPPDATA%\clm_gpu\` (lessico, associazioni, concetti, tarature).
- Risultati delle prove: `results/`.
