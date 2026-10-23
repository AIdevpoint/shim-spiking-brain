from __future__ import annotations

PRIME_12 = ["ciao", "mamma", "papà", "acqua", "palla", "gatto", "casa", "pane", "latte", "nonna", "sole", "luna"]

CATEGORIE = {
    "animali": ["gatto", "cane", "pesce", "uccello", "mucca", "cavallo", "topo", "pecora"],
    "cibo": ["latte", "pane", "mela", "formaggio", "carne", "pasta", "uovo", "torta"],
    "natura": ["sole", "luna", "mare", "albero", "fiore", "cielo", "nuvola", "pioggia", "fiume", "montagna", "acqua", "giorno", "notte"],
    "casa": ["casa", "porta", "letto", "tavolo", "finestra", "sedia", "palla"],
    "persone": ["mamma", "papà", "nonna", "bambino", "amico", "ciao"],
    "azioni": ["beve", "mangia", "dorme", "corre", "vola", "nuota", "cresce", "brilla", "cade", "canta", "abbaia", "miagola", "gioca"],
    "qualità": ["grande", "piccolo", "caldo", "freddo", "dolce", "bianco", "azzurro"],
}


PAROLE = list(PRIME_12) + [w for c in CATEGORIE.values() for w in c if w not in PRIME_12]
CATEGORIA_DI = {w: c for c, ws in CATEGORIE.items() for w in ws}

FRASI = """
gatto miagola
gatto beve latte
gatto dorme letto
gatto mangia pesce
gatto mangia topo
gatto corre casa
gatto piccolo
gatto bianco
gatto gioca palla
gatto dorme sole
gatto beve acqua
gatto miagola mamma
cane abbaia
cane corre
cane mangia carne
cane beve acqua
cane gioca palla
cane grande
cane dorme casa
cane mangia pane
cane abbaia gatto
cane corre montagna
cane amico bambino
pesce nuota
pesce nuota mare
pesce nuota fiume
pesce piccolo
pesce nuota acqua
pesce mangia
pesce freddo acqua
uccello vola
uccello vola cielo
uccello canta
uccello piccolo
uccello canta albero
uccello vola nuvola
uccello mangia mela
uccello canta sole
mucca mangia
mucca beve acqua
mucca grande
mucca latte
mucca mangia fiore
mucca dorme
cavallo corre
cavallo grande
cavallo mangia mela
cavallo bianco
cavallo corre montagna
cavallo beve fiume
topo piccolo
topo corre
topo mangia formaggio
topo mangia pane
topo dorme casa
topo corre gatto
pecora mangia
pecora piccolo montagna
pecora bianco
pecora latte
latte bianco
latte dolce
latte mamma
latte freddo
latte caldo
pane caldo
pane mangia nonna
pane bianco
pane dolce
mela dolce
mela albero
mela cade
mela mangia bambino
formaggio mangia topo
formaggio latte
carne mangia cane
pasta caldo
pasta mamma
pasta mangia papà
uovo bianco
uovo mangia
torta dolce
torta nonna
torta mangia bambino
torta caldo
sole caldo
sole brilla
sole cielo
sole azzurro
sole grande
sole fiore cresce
sole mare
sole giorno
luna brilla
luna cielo
luna bianco
luna notte
luna dorme
luna freddo
mare grande
mare azzurro
mare pesce
mare acqua
mare sole
mare freddo
albero grande
albero cresce
albero mela
albero uccello
albero fiore
albero cielo
fiore cresce
fiore dolce
fiore sole
fiore bianco
fiore mamma
cielo azzurro
cielo grande
cielo nuvola
cielo sole
cielo luna
nuvola bianco
nuvola cielo
nuvola pioggia
nuvola vola
pioggia acqua
pioggia cade
pioggia freddo
pioggia nuvola
pioggia mare
fiume acqua
fiume pesce
fiume corre
fiume montagna
fiume mare
montagna grande
montagna freddo
montagna bianco
montagna cielo
montagna fiume
acqua freddo
acqua beve
acqua pioggia
acqua mare
acqua fiume
acqua pesce nuota
casa grande
casa porta
casa finestra
casa mamma
casa letto
casa tavolo
casa caldo
casa nonna
porta casa
porta grande
finestra sole
finestra cielo
finestra casa
letto dorme
letto bambino
letto caldo
tavolo pane
tavolo mangia
tavolo casa
sedia tavolo
sedia papà
palla gioca
palla bambino
palla cane
palla gatto
palla corre
palla grande
palla bianco
mamma ciao
mamma bambino
mamma latte
mamma casa
mamma canta
mamma dolce
papà ciao
papà casa
papà corre
papà bambino
papà mangia
papà grande
nonna dolce
nonna torta
nonna casa
nonna pane
nonna dorme
nonna canta
bambino gioca
bambino corre
bambino dorme
bambino piccolo
bambino mangia
bambino ciao
bambino amico
bambino mamma
amico ciao
amico gioca
amico bambino
amico corre
ciao amico
ciao mamma
ciao papà
ciao nonna
ciao bambino
""".strip().splitlines()

FRASI = [f.split() for f in FRASI if f.strip()]
