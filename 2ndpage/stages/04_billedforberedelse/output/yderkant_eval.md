# Yderkanten — evaluering af forsøg A og B

Skrevet 2026-08-28. **Ingen af tallene her er godkendt af lead.** Facit er
min egen visuelle gennemgang; den skal efterses, før tallene bruges til
noget. Ingen beskårne billeder er skrevet — det kræver go.

## Hvad problemet var

Falsbeskæringen (`skraa.py`) renser den ene kant. På den modsatte —
sidens yderkant — ligger enten **bogsnittet** (bogblokkens sammenpressede
sidekanter set fra siden, brunt, uden læsbar tekst) eller et **blad
længere inde i bindet**, som er faldet fladt ud og blevet fotograferet
med. I sidste tilfælde står der fremmed håndskrift langs kanten, som
falsbeskæringen aldrig kan nå.

## Facit (`yderkant_facit.csv`) — min gennemgang, ikke leads

Alle 118 yderkantsstrimler gennemset i `yderkant_ark/` (17 ark à 7
strimler):

| Klasse | Antal |
|---|---|
| `ren` — intet fremmed blæk uden for vores kant | 110 |
| `fremmed_tekst` — læsbar fremmed skrift uden for kanten | **7** |
| `usikker` — kan ikke afgøres | 1 |

De syv: `273099_001445`, `273103_001437`, `273107_001864`,
`273107_001866`, `37554_001492`, `37554_001494`, `37554_001496`.
I bind 37554 er det det **samme** blad, der rager ud på alle tre sider.

Et synligt udragende blad (med eller uden tekst) findes på 19 sider,
usikkert på 8. Den kolonne er min svageste vurdering.

**Fem sider blev efterprøvet i fuld opløsning, fordi kontaktarket førte
mig på afveje.** To af dem havde jeg klassificeret forkert:
`273104_001645` er vores EGEN tekst, spejlvendt i arket, og
`273105_001708` er sidens egne punktummer — ikke fremmed blæk.

## Hvad detektionen gør

Papirets grundlyshed pr. kolonne (85-percentil, ikke gennemsnit — blækket
drukner gennemsnittet), målt bånd for bånd som i `skraa.py`, fordi
bladene ligger skævt. Hvert bånd melder **alle** betydelige fald udad;
derefter vælges den **inderste rette linje**, som mindst 6 bånd kan enes
om. Rækkefølgen er hele pointen: først inderste kant (et blad ligger
altid uden for vores side), derefter bedst støttede linje langs den kant.

## Snittets retning — rettet efter leads indsigelse

Lead så på den første gennemgang, at snittet stedvis tog "tre-fire
bogstaver" af ordenderne, og bad om at prioritere kantens **ydre** side.
To ting blev ændret:

1. Kanten meldes nu i faldets **bund** i stedet for ved dets begyndelse.
   Begyndelsen ligger op til 12 kolonner inde på vores egen side.
2. Bufferen er hævet fra 0,5 % til **1,2 %** af bredden (~16 px).

Målt over alle 118: snittet flyttede **22 px udad i median** (20–58 px).
Fjernet andel faldt fra 12,5 % til **10,9 %** i median. Alle syv sider med
fremmed tekst holder stadig strimlen uden for snittet — bæltet uden for
kanten måles nu til 78–130 px mod før 90–142 px.

Afvejningen er bevidst usymmetrisk: et tabt bogstav er en fejl i
transskriptionen, mens en tilbageblevet flig af naboen kun er støj, som
prompten kan bede modellen se bort fra.

## Kontaktarket bedrog — stregen er nu erstattet af en tonet flade

Den første udgave tegnede snittet som en **rød streg oven på billedet**.
Stregen dækker de bogstaver, der står tættest på snittet, og efter
nedskaleringen til arket ser de ud til at være klippet af. Både lead og
jeg blev ført bag lyset af det: `273101_001164` blev af mig noteret som
"skærer gennem teksten", men er ved fuld opløsning helt korrekt.

Arket toner nu i stedet det bortskårne rødt. Intet males over: det, der
beholdes, står urørt, og det, der ryger, kan stadig læses igennem tonen.

## Forsøg A — skær ved sidens egen kant, alle sider

Alle 118 snit set efter med øjnene i `yderkant_snit_ark/`:

| Resultat | Antal |
|---|---|
| Snittet sidder på sidens kant | **116** |
| Snittet skærer gennem vores egen skrift | **1** (`273108_001555`) |
| Omtvistet | 1 (`273103_001463`) |
| Sidens egen kant fjernet af bredden | median 10,9 % (4,5–21,2 %) |

- `273108_001555`: snittet ligger ~40 px inde på siden og halverer
  bogstaverne. **Mærket usikker af koden** (9 af 24 bånd) — værnet virker.
- `273103_001463`: snittet ligger ~60 px inde, og der står blæk uden for
  det. Om det blæk er vores eget eller et udragende blads, kan jeg ikke
  afgøre — det er netop den side, facit også kalder usikker. **Ikke
  mærket af koden.** Kræver leads øje.
- Alle **7** sider med fremmed tekst skæres korrekt: den fremmede strimmel
  havner uden for snittet på alle syv.

Nedskalerede ark kan udpege mistanker; de kan ikke afgøre dem. Hver eneste
mistanke i denne evaluering er efterprøvet i fuld opløsning, og tre af dem
holdt ikke.

## Forsøg B — find kun de sider med et fremmed blad

Afgøres på **bredden** af det lyse bælte uden for kanten. Ikke på "kommer
papiret igen": også et rent bogsnit giver ~20 px lyst papir lige uden for
kanten, så det spørgsmål siger ja til alt.

| Facit siger blad | Sider | B siger ja |
|---|---|---|
| ja | 19 | 16 |
| nej | 91 | 26 |
| usikker | 8 | 3 |

På de syv sider med fremmed tekst siger B ja på **6 af 7**. Den, der
tabes, er `37554_001496`, hvor bladets eget bælte kun måles til 16 px.

B rammer altså ved siden af på 26 af 91 rene sider. Det var det tilsigtede
bytte (recall før precision), men prisen er høj, og B fanger færre af
problemsiderne end A skærer rigtigt.

## To mål, der ikke virkede — noteret, ikke skjult

Begge forsøg på at måle snittets rigtighed automatisk blev opgivet.

Det første: "står der blæk på begge sider af linjen?". **Det bestod ikke
sin egen prøve.**
`273108_001555`, hvor snittet beviseligt går gennem skriften, scorede
LAVERE end sider, hvor snittet sidder rigtigt. Bogsnittets mørke striber
tæller som blæk uanset tærskel.

Det andet: "hvor mange rækker har blæk lige inden for snittet?". Det
udpegede `273104_001645` og `273110_001526` som de værste — begge er ved
fuld opløsning helt rene. Målet talte papirkantens egen skygge mod den
sorte baggrund.

Begge er fjernet igen frem for trimmet, til de så rigtige ud. Snittene må
ses efter med øjnene.

Til gengæld siger `sikker`-kolonnen nu faktisk nej: 2 af 118 sider mærkes
(`273108_001555`, `37554_001496`), mod falsbeskæringens 0 af 118. Den ene
af de to reelle fejl fanges dermed af værnet.

## Hvad tallene IKKE viser

- Intet af dette er målt på prøvemængdens 50 sider. De er urørte med vilje.
- Der er stadig ikke kørt ét modelkald. Om den fremmede strimmel
  overhovedet ender i en transskription, er **uafprøvet** — vi ved kun, at
  den er der.
- Facit hviler på mit øje, ikke leads. Tre af mine egne aflæsninger viste
  sig forkerte undervejs, alle rettet ved at gå til fuld opløsning.
