# Le prove di SHIM

Ogni prova ha la **previsione scritta prima** di accendere il cervello; il verdetto (SÌ / NO) si aggiunge dopo, coi numeri, anche quando è NO.
Le prove passano dalla conversazione vera (`/dialogo`): la frase si capisce con uno schema (dichiarato), la parola della risposta esce dai
neuroni (vince il gruppo di neuroni che spara di più). Ogni prova si rifà con un comando, a SHIM acceso:

    python -m shim.prova_concetti 8800      (e così prova_perche, prova_seguito, prova_eccezioni, prova_riassunto,
                                                  prova_conversazione, prova_matematica, prova_confronto)

I risultati finiscono in `results/`.

## Previsioni (scritte il 23/10/2026, prima della prima accensione di SHIM)

Il riferimento è la versione di partenza di SHIM, non pubblicata (stesse conoscenze, stesso numero di neuroni, più aree), misurato il 23/10.

| | Previsione | Riferimento |
|---|---|---|
| A1 | La taratura arriva (eccitatori fra 2 e 4,5 Hz in media) | — |
| A2 | Velocità fra 0,45× e 0,65× il tempo reale | 0,5–0,6× |
| A3 | `prova_concetti`: tutte le soglie della prova (K1 ≥ 8/10, K2 ≥ 8/10, K3 ≥ 5/6, K4 ≥ 8/10, K5 ≥ 4/5, K6 8/8) | 10, 10, 6, 10, 4, 8 |
| A4 | `prova_perche`, `prova_seguito`, `prova_eccezioni`, `prova_riassunto`: le soglie di ogni prova | tutte SÌ |
| A5 | `prova_conversazione`: C1–C4 SÌ | tutte SÌ |
| A6 | `prova_matematica`: M1–M3 SÌ; `prova_confronto`: Q1 e Q2 SÌ | M SÌ; Q1 Q2 SÌ, Q3 NO |
| A7 | Risposta in conversazione: in media sotto i 3 secondi | ~2 s |

Perché ci si aspetta di passare: le conoscenze (concetti, associazioni) sono le stesse, e i gruppi che contano (parole, memoria, numeri)
hanno gli stessi neuroni. Il rischio: dentro un'area grande i collegamenti sono più "mescolati" che nel cervello completo, e la memoria di lavoro
o la competizione fra le parole potrebbero comportarsi diversamente. Se una prova va peggio, si scrive NO.

## Verdetti (23/10/2026 sera, portatile alla corrente, Intel Arc 140V, nessun altro programma sulla scheda)

**Primo giro (21:09), il codice così com'era:**

| | Esito | Numeri |
|---|---|---|
| A1 | **SÌ** | eccitatori 3,28 Hz in media al giro 24 |
| A2 | **SÌ** | 0,49–0,54× il tempo reale |
| A3 | **SÌ** | K1 10/10, K2 10/10, K3 6/6, K4 10/10, K5 4/5, K6 8/8 |
| A4 | **SÌ** | perché 6/6, insegnato 1/1, quanti 4/4, non inventa 2/2; seguito 5/5, 4/4, 2/2; eccezioni 4/4, confronti 5/5; cosa sai 5/5, in comune 5/5 |
| A5 | **SÌ** | C1 8/8, C2 8/8, C3 3/3, C4 19/19 |
| A6 | **NO** | M1 10/10, M2 6/6, **M3 0/6** (parlare di "il cerchio?"); confronto Q1 100%, Q2 100% (Q3 NO, come nel riferimento) |
| — | **NO** | `prova_legate` (non prevista sopra, soglie della prova): L1 21/30, L2 19/30 |

**Causa (trovata dopo):** a ogni turno la memoria di lavoro si svuota e poi legge le parole della frase; in SHIM alcune assemblee della
prefrontale si accendono da sole (a riposo 8 su 1 000) e la risposta partiva da quelle ("Il cerchio?" → "La repubblica?"). Cambiare la
soglia della parte lenta (9, 12, 15 Hz) o l'inibizione (6–20) non basta: o restano gli intrusi o non tiene più la parola letta.
**Correzione (regola scritta a mano, DICHIARATA):** valgono solo le parole tenute in testa che sono state lette in quel turno; le altre
sono rumore e si scrivono in `info.intrusi`.

**Secondo giro (21:37), con la correzione, tutte le prove rifatte:**

| | Esito | Numeri |
|---|---|---|
| A3 | **SÌ** | K1 10/10, K2 10/10, K3 6/6, K4 10/10, K5 4/5, K6 8/8 |
| A4 | **SÌ** | come al primo giro |
| A5 | **SÌ** | C1 8/8 (tutte entro 4 s), C2 8/8, C3 3/3, C4 19/19 |
| A6 | **SÌ** | M1 10/10, M2 6/6, M3 5/6; confronto Q1 100%, Q2 90% (Q3 effetto distanza NO, come nel riferimento) |
| — | **SÌ** | `prova_legate`: L1 28/30, L2 26/30 (riferimento 30 e 22) |
| A7 | **NO** | frasi fatte ~1 s (mediana 1,0 s), domande su cose 4–5 s in media (riferimento ~2 s, misurato su altre domande) |

**Terzo giro (22:52–23:08), il codice nella forma pubblicata (pacchetto `shim/`), tutte le prove rifatte:**

| | Esito | Numeri |
|---|---|---|
| A1 | **SÌ** | eccitatori 3,28 Hz in media (identica al primo giro: stesso cervello) |
| A3 | **SÌ** | K1 10/10, K2 10/10, K3 6/6, K4 10/10, K5 4/5, K6 8/8 |
| A4 | **SÌ** | perché 6/6, 1/1, 4/4, 2/2; seguito 5/5, 4/4, 2/2; eccezioni 4/4, 1/1, confronti 5/5, 1/1; cosa sai 5/5, in comune 5/5, 1/1 |
| A5 | **SÌ** | C1 8/8, C2 8/8, C3 3/3, C4 19/19 |
| A6 | **SÌ** | M1 10/10, M2 6/6, M3 6/6; confronto Q1 100%, Q2 100% (Q3 effetto distanza NO) |
| — | **SÌ** | `prova_legate`: L1 28/30, L2 26/30 |

Al primo passaggio di questo giro K6 era 7/8: la risposta «Mi chiamo SHIM.» era giusta, ma nella prova la parola attesa era stata
cambiata per errore durante la rinomina dei file. Corretta la prova, `prova_concetti` rifatta: K6 8/8.

I file: `results/prova_*.json` (terzo giro). Il primo e il secondo giro sono descritti qui sopra coi loro numeri.
