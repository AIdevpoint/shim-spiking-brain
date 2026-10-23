# Dati usati da SHIM e loro licenze

| Cosa | Da dove | Licenza | Dove sta in SHIM |
|---|---|---|---|
| Collegamenti fra le aree | Human Connectome Project, matrice strutturale distribuita dal progetto ENIGMA Toolbox, **aggregata in 9 aree** | BSD-3 (ENIGMA Toolbox); dati HCP: Open Access Data Use Terms | `shim_mappa.json` (solo la tabella 9 × 9) |
| Associazioni fra le parole, ordine delle parole, definizioni | Wikipedia in italiano | CC BY-SA 4.0 | `pesi/shim_assoc_2000.npz`, `shim_ordine_2000.npy`, `shim_classi_2000.json`, `shim_frasi_2000.json`, `shim_definizioni_2000.json` |
| "È un" e "ha" | Wikidata | CC0 | `pesi/shim_wikidata_2000.json` |
| Associazioni delle storie | Wikisource in italiano (Pinocchio, Cuore, fiabe; testi di pubblico dominio, trascrizioni CC BY-SA) | CC BY-SA 4.0 | `pesi/shim_assoc_storie.npz` |
| Parole più usate dell'italiano parlato | elenco di frequenze dai sottotitoli OpenSubtitles (hermitdave/FrequencyWords) | MIT (elenco); CC BY-SA 4.0 (dati) | `pesi/shim_vocab_2000.json` |
| Libro di scuola (362 frasi) | scritto per il progetto | come il codice | dentro `pesi/shim_concetti_2000.json` |

I file derivati da Wikipedia, Wikidata e Wikisource vanno ridistribuiti **con attribuzione e con la stessa licenza (CC BY-SA 4.0)**.
Da verificare prima di pubblicare: i termini HCP per un derivato aggregato a 9 aree; la licenza dell'elenco di frequenze.
