# Stage 03 — Måleapparat

## Formål

Kunne sætte et tal på en transskription. Tegnfejl på den fladede tekst er
tallet, vi træffer beslutninger på; linjeopdelingen bevares og bruges til en
særskilt kontrol for, om modellen har digtet hen over et linjeskift.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Facit | `../02_facit/output/facit.jsonl` |
| Forarbejde | `../../references/stadscer.md` |

## Process

Beslutningerne herunder er truffet af lead 2026-08-20/21 og er låst. Numrene
henviser til beslutningstabellerne i rod-`CONTEXT.md`.

1. Overtag målekoden fra `J-Hoffi/StadsCER` frem for at skrive ny:
   `normalize`, `strip_diacritics`, `levenshtein`, `align` og de fem
   varianter `raa`, `uden_versaler`, `uden_diakritika`, `uden_tegnsaetning`,
   `lempeligst`. Alle fem rapporteres side om side; ingen af dem må vælges
   efter, hvilken der klæder resultatet bedst.
   **Arbejdstallet er `uden_versaler + uden_tegnsaetning`** (beslutning 26);
   det rå tal er det, der står i leverancen. Vi lover ikke mere, end det rå
   tal kan bære.
2. **Mål på `alt_*`-udgaven af facit, ikke `rettet_*`** (beslutning 24).
   Modellen bliver bedt om at læse hele siden og bliver udtrykkeligt ikke
   bedt om at afgøre, hvad der er streget ud — måler vi mod den rettede
   læsning, straffes den 33 steder for at gøre det, vi bad om.
3. **Mål både på den fladede tekst og pr. linje** (lead 2026-08-20).
   Linjemålingen skal parre linjerne, før den sammenligner, så den ikke
   skrider efter det første afvigende linjebrud.
   **UBEVIST antagelse, må ikke bygges på** (beslutning 35): vi ved ikke, om
   modellen laver sine egne linjeskift eller følger sidens. Der er ikke kørt
   et eneste modelkald endnu, og kollegaens app har allerede et
   linje-for-linje-skema, hvilket peger den anden vej. Måleapparatet skal
   virke uanset svaret. Stage 05's første kørsel svarer på det gratis.
4. Byg det, StadsCER mangler: samling af orddeling hen over linjeskift, som
   dagbogen dér udpeger som det dominerende fejlmønster. Dette er projektets
   bidrag tilbage. **Bemærk**: bindestreg sidst på en linje samles kun, når
   næste linje fortsætter med lille bogstav — materialet bruger også
   bindestreg som punktum (beslutning 21). Facit gør det allerede sådan;
   måleapparatet skal gøre det samme ved modellens tekst.
5. Byg hallucinationskontrollen: find steder hvor modellen har fuldført et ord
   hen over et linjeskift, som facit deler. Kontrollen må ikke kræve, at model
   og facit har samme linjeopdeling — i StadsCER er linjerne givet på
   forhånd, hos os er de det muligvis ikke (se punkt 3).
6. **Behandling af ulæselige steder er fastlagt** (beslutning 23 og 38):
   ulæselige steder må hverken tælle for eller imod, og **hele linjen skæres
   fra målingen**, ikke bare selve stedet. Tegn-for-tegn-opstilling af hele
   teksten er ikke til at stole på dér, hvor teksterne afviger — og det gør
   de netop omkring de ulæselige steder.
   Målt pris: 422 af 3.680 linjer (11,5 %), 10.898 af 89.770 tegn (12,1 %).
   (Linjetallet var 3.526 ved opmålingen 21. august, som blev lavet før
   facit blev bygget igen samme dag. De 422 svære linjer holder.)
   Ingen side mister over halvdelen; 33 sider mister intet.
7. **Byg ÉN funktion, ikke fire features** (CONTEXT.md 2026-08-21):

       forankr(facit_linje, modeltekst) -> (fundne_stumper, gab, ikke_fundne)

   Den finder facits kendte stumper i modellens tekst ved almindelig
   tekstsøgning — ikke ved opstilling af hele siden. Ud af den ene funktion
   falder tre ting: de fundne stumper måles (genvinder op mod 9.404 tegn =
   86 % af de svære linjer); gabet mellem to fundne stumper er
   hallucinations-signalet; samme gab er modellens bud på det ulæselige sted.
   Regler der holder den lille: stumper under 5 tegn bruges ikke (122 af
   647); et gab tælles kun med stumper fundet på begge sider; en uforankret
   linje falder tilbage til punkt 6. **Punkt 6 forbliver grundreglen** —
   forankringen er en forbedring ovenpå med en defineret vej tilbage.
8. **Ordfejl rapporteres ved siden af tegnfejl** (beslutning 28). Ingen
   målrettet opmærkning af datoer, temperaturer eller medicinnavne — det
   kræver et større apparat, og det er ikke dét, teksten skal bruges til.
   leads egne ord: det er den fortællende optagelsestekst om patientens
   forhistorie og levevilkår ("har haft mæslinger", "underernæret"), der er
   målet; temperaturer og medicin er mindre vigtige.
9. **Dækningen SKAL stå ved siden af hvert eneste tal.** Det, vi skærer fra,
   er ikke tilfældigt — det er de sværeste linjer på siden, og det er derfor,
   transskribenten ikke kunne læse dem. Tallet bliver systematisk for pænt.
   Formuleringen skal fremgå af rapporten, fx "målt på 88 % af linjerne; de
   udeladte 12 % er de sværeste". Uden den er tallet misvisende, uanset hvor
   korrekt det er udregnet.
   *(Forankringen, punkt 7 og 13, henter det meste af det tilbage — målt til
   en dækning omkring 97 % på øvemængden. Kravet står uændret: procenten
   skal skrives ud, hvad den end lyder på.)*
10. **Ingen kvalitetsgrænse fastsættes på forhånd** (beslutning 27). Lead vil
    se det første rigtige tal, før der sættes grænser, og grænserne bliver
    derefter forskellige alt efter brugen ("kan søges i" mod "kan citeres").
11. **Facit rummer fejl, og det accepteres** (beslutning 37). Én er bekræftet:
    `37554_001491` skriver "for 2 Dage siden", på siden står "for 3 Dage
    siden". Anslået størrelsesorden hvis raten holdt: ~60 fejl = 0,06 % af
    tegnene, altså et forsvindende gulv under hovedtallet. Men det betyder,
    at en ENKELT uenighed mellem model og facit ikke automatisk er modellens
    fejl. Skriv det i rapportformatet.
12. Byg rapportformatet: fem varianter i tabel, hallucinationstal opgjort for
    sig, dækningen anført, og de værste enkeltopslag udpeget, så de kan ses
    efter med øjnene.

13. **Forankringen søger med tolerance, ikke efter ordret træf** (beslutning
    39, truffet under bygningen 2026-08-22). En stump regnes for fundet, hvis
    det nærmeste stykke modeltekst afviger med højst 40 % af stumpens længde.
    Grunden er, at stumperne skal MÅLES: krævede forankringen ordret træf,
    ville hver forankret stump per definition have nul fejl, og apparatet
    ville bekræfte sig selv. Grænsen er en knap, der kan bruges til at pynte —
    strammes den, falder dækningen, og de linjer der bliver tilbage, er de
    letteste. Derfor står dens virkning målt op i `output/selvtest.md`.
14. **Orddelingen afgøres på FACITS linjer og bruges på begge sider**
    (beslutning 42). Ellers straffes en model, der har læst rigtigt: facit
    deler `Infektions-` / `sygdomme.`, og en model der skriver ordet samlet,
    ville få en fejl for det. Linjedelingen er sidens artefakt, ikke tekstens
    indhold.
15. **Fuldside-kontrollen er fast del af rapporten** (beslutning 43). På de
    sider, der slet ikke har `[?]`, sammenlignes hele siden direkte uden
    forankring. Det er rapportens eneste tal, der ikke kan pynte på noget:
    forankringen ser hverken tekst, modellen har fundet på, eller tekst, den
    har sprunget over — kun det, der kunne parres.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/rapportformat.md` | Eksempel på en færdig måling, så formatet er aftalt før første kørsel |
| `output/selvtest.md` | Måleapparatet kørt mod facit selv og mod bevidst forvanskede udgaver |
| `output/gab_eksempel.csv` | Alle gab fra samme prøvemåling — hvad "modellen" skrev, hvor facit siger `[?]`. Formatet på den fil, en rigtig kørsel skal aflevere. Ikke facit, og må aldrig skrives ind i det |

Koden ligger i `src/andenside/`: `cer.py` (overtaget fra StadsCER), `maal.py`
(forankringen og målingen) og `rapport.py` (rapportformatet). Selvtesten
køres med `scripts/selvtest_maaleapparat.py`.

## Test Contract

Måleapparatet skal give nul fejl, når facit sammenlignes med sig selv, og et
kendt, forud udregnet tal på konstruerede forvanskninger. Der skal være en
test for hver af de fem varianter, en test for orddelingssamlingen, og en test
for at to kørsler på samme data giver nøjagtig samme rapport — rækkefølge fra
mængder og ordbøger har tidligere givet ikke-reproducerbare resultater i andre
projekter.

## Handoff

Næste stage er `05_foerste_transskription`. `04_billedforberedelse` blev
godkendt og låst 2026-08-18, altså før denne stage blev bygget — rækkefølgen i
mappenavnene er ikke den rækkefølge, arbejdet faktisk faldt i.

Reviewed betyder, at du har set `output/rapportformat.md` og er enig i, at det
er DE FELTER, du vil træffe valg ud fra. Tallene i filen er konstruerede og
betyder ingenting — det er formatet, der skal godkendes, ikke resultatet.

Stage 05 er det første, der kræver modelkald. Ingen kørsel uden dit go.
