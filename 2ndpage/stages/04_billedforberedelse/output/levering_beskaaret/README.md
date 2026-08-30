# Begge snit kørt på hele leveringen

Kørt 2026-08-30 med `scripts/beskaer_levering.py`. **307 sider**, begge snit:
først falsen (`skraa.beskaer_langs_fals`), derefter yderkanten
(`yderkant.beskaer_ydre`). Ingen modelkald — ren lokal billedbehandling.

Det er **første gang yderkant-snittet faktisk skrives** til billeder; indtil
nu havde det kun været målt.

| Gruppe | Sider | Fjernet i alt (median) | Usikre |
|---|---:|---|---:|
| `oeve/` | 173 | 28,0 % (15,2-42,9 %) | 3 |
| `ekstra_uden_facit/` | 50 | 31,8 % (9,0-45,3 %) | 0 |
| `proeve_LAAST/` | 84 | 30,4 % (5,2-42,9 %) | 1 |

"Fjernet i alt" dækker **begge** snit — falsstrimlen plus yderkanten — så
tallet er naturligt større end de ~11 %, yderkanten alene fjerner.

## Falssnittet er rettet (2026-08-30, efter leads gennemgang)

Lead fandt fire sider, hvor **falssnittet** var gået helt galt, og fire mere
hvor det var lidt galt. Diagnosen var entydig: båndene fandt hver især
*noget*, men lå ikke på samme linje, og `fals_graense` interpolerede frit
igennem dem. Ét udskridende bånd kunne trække snittet 400 px ind over siden.
`sikker` sagde ja til dem alle, fordi kolonnen kun talte, om båndene fandt
noget.

**Målt på leveringens 307 sider afviger falsen 5 px fra en ret linje i
median og 11 px ved 90-percentilen.** De sider, lead kaldte gået galt,
afveg 245-412 px — ti gange for meget til at være en krumning. Instrumentet
skiller hans tre grupper rent:

| leads dom | max afvigelse fra ret linje |
|---|---|
| gået galt | 245-412 px |
| lidt galt | 11-72 px |
| god | 3-7 px |

`skraa.fjern_udskridende` kaster nu de bånd, der ikke ligger på flertallets
linje — den værste ad gangen, med ny linje hver gang, indtil resten ligger
inden for 25 px. Efter rettelsen har **alle** de gale sider afvigelse 0, og
fjernet andel er gået fra 0-2 % til 8,5-33 %. `sikker` kræver nu også, at
båndene er enige.

**21 af 307 sider fik kastet bånd.** De er værd at se efter, selv om de nu
ser rigtige ud — listen står i `../fals_kvalitet.csv`, sorteret. De otte
værste: `273102_001070`, `273108_001604`, `273069_000072`, `273086_000042`,
`273067_000906`, `273085_001208`, `273109_000083`, `273108_001539`.
Bemærk at fire af dem er i **øvemængden** — de har været skåret forkert hele
tiden, uden at nogen måling sagde fra.

## Kontaktarkene viser nu det bortskårne

Lead påpegede, at man ikke kan bedømme et snit på resultatet alene: er der
skåret for meget, ses det ikke, for det manglende er jo netop ikke i
billedet længere. Arkene viser nu hele siden med **falsen tonet rød** og
**yderkanten tonet blå**, så begge snit kan ses og skelnes.

## De fire usikre — start her

| Side | Hvad der er galt |
|---|---|
| `273111_001380` | falssnittet fandt ikke nok bånd |
| `273111_001381` | samme |
| `37554_001496` | yderkanten hviler på kun 7 af 24 bånd (kendt marginal side) |
| `273107_001884` | yderkanten hviler på 10 af 24 bånd (prøvemængden) |

## Derefter: de tyndest funderede før de værste

Lektien fra stage 03 gælder her: en side, hvor detektionen næsten intet
fandt, får et pænt tal, netop fordi den knap blev rørt. Se derfor disse
efter, før du ser på dem med de største tal:

- **Mindst fjernet** — kan betyde, at snittet næsten ikke rykkede sig:
  `273102_001070` (15 %), `273086_000042` (9 %), `273101_001105` (5 %)
- **Færrest bånd bag yderkanten**: `37554_001496` (7/24),
  `273069_000072` (12/24), `273108_001555` (12/24)
- **Mest fjernet** — kan være rigtigt på bind med dyb fals, men bør ses:
  `273037_000562` (45 %), `273024_001127` (44 %), `273102_001065` (43 %)

## Om prøvemængden

`proeve_LAAST/` er beskåret efter leads udtrykkelige ønske. Værnet
(`vaern.sikr_oevemaengde`) handler om at **se facit** for de sider, ikke om
at behandle billedpunkter — en beskæring rører ikke facit. Resultatet ligger
i sin egen mappe med sin egen målefil, så ingen senere kørsel kan blande de
to mængder sammen. **De sider må stadig ikke måles på** før den endelige
bedømmelse.

## Bemærk om de 50 uden facit

De er andensider fra 1889-1897, altså uden for de 15 måneder, vi har facit
for. **Ingen af dem blev mærket usikker**, hvilket er det første tegn på, at
billedforberedelsen også holder uden for facit-perioden. Det er dog kun et
fravær af alarm, ikke en måling — de kan ikke måles mod facit, så kun
øjnene kan afgøre, om snittene er rigtige.
