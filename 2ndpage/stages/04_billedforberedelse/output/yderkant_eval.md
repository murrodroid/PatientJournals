# Yderkanten — evaluering

Skrevet 2026-08-28, omskrevet 2026-08-29 efter leads gennemsyn.
**Ingen beskårne billeder er skrevet — det kræver go.**

## Hvad problemet var

Falsbeskæringen (`skraa.py`) renser den ene kant. På den modsatte —
sidens yderkant — ligger enten **bogsnittet** (bogblokkens sammenpressede
sidekanter set fra siden, brunt, uden læsbar tekst) eller et **blad
længere inde i bindet**, som er faldet fladt ud og blevet fotograferet
med. I sidste tilfælde står der fremmed håndskrift langs kanten.

Facit (`yderkant_facit.csv`, min gennemgang rettet efter lead):
**8 af 118 øvesider har fremmed tekst** langs yderkanten, 110 er rene.

Lead valgte **forsøg A**: skær alle 118 ved sidens egen kant.

## Måleapparatet — bygget FØRST, og efterprøvet

Alle tidligere runder led af det samme: jeg bedømte snittene med øjnene på
nedskalerede kontaktark og tog fejl. Derfor blev et måleapparat bygget og
holdt op mod leads egne domme, før det blev brugt til noget.

Det måler **sømdybde**: hvor mørk en fordybning snittet ligger i, målt mod
papirets eget niveau lige omkring, ned gennem hele siden. Sidens kant er en
fysisk ting — papiret slipper, og kanten kaster en smal skygge. En linje
trukket hen over åbent papir gør ikke.

Holdt op mod leads seks domme, på koden som den så ud da han dømte:

| side | leads dom | sømdybde |
|---|---|---|
| `273105_001569` | gået galt | **3,0** |
| `273103_001437` | lidt galt | **5,0** |
| `273108_001555` | god | 15,0 |
| `273111_001376` | god | 18,0 |
| `37554_001492` | god | 25,0 |
| `37554_001494` | god | 12,0 |

Dommene skiller sig rent. **Tre tidligere måleforsøg blev kasseret**, fordi
de ikke bestod denne prøve:

1. *"Står der blæk på begge sider af linjen?"* — `273108_001555`, hvor
   snittet beviseligt gik galt, scorede LAVERE end gode sider. Bogsnittets
   mørke striber tæller som blæk uanset tærskel.
2. *"Hvor mange rækker har blæk lige inden for snittet?"* — udpegede to
   sider som de værste; begge var ved fuld opløsning helt rene. Målet
   talte papirkantens egen skygge mod den sorte baggrund.
3. *Klippet blæk* (den nuværende støttemåling) har stadig kendte falske
   udslag på op til 145 rækker på sider, der er i orden. Den bruges som
   støtte, aldrig som dommer.

## Hvad der faktisk var galt — og hvad der ikke var

Undervejs blev der bygget en **svag-bekræftelse**: linjer måtte foreslås af
de bånd, der så kanten tydeligt, og bekræftes af bånd, der kun anede den.
Den blev tilføjet for at redde `273108_001555`.

**Den var årsagen til begge leads fejl.** Fjernes den, ligger snittet
rigtigt på `273105_001569` (sømdybde 3,0 → 51,0) og `273103_001437`
(5,0 → 54,0), og `273108_001555` bliver god alligevel. Den er derfor
fjernet igen, sammen med de to konstanter, den trak med sig.

Det er værd at skrive ned som en fejltype: en mekanisme tilføjet for at
redde ét tilfælde, som ødelagde flere andre — og hvis skade først blev
synlig, da der fandtes et måleapparat.

## Sømkravet — beholdt som værn, ikke som løsning

Kandidatkanter skal nu bevise deres søm, før de kommer i betragtning.
**Målt ændrer det ingenting på øvemængden**: med og uden kravet er
resultatet identisk på alle 118 sider, efter at svag-bekræftelsen er væk.

Det bliver stående som værn for materiale, vi ikke har set — prøvemængdens
50 sider og de ~71.000 rigtige sider — og som det, der ville fange netop
den fejltype igen. Er der ingen kant med en rigtig søm, skæres siden ikke.

Gulvet er 6. De forkerte linjer målte 3,0 og 5,0; de rigtige måler 10-163,
**på nær `273107_001866`, hvis rigtige kant kun når 5-7**. Der er altså
overlap ved 5, og gulvet ligger lige over det højeste målte falske.
**Marginen er tynd, og det er løsningens svageste led.**

## Resultat

- Alle 118 sider skæres; ingen afstår.
- **Nul sider** har et snit uden rigtig søm (mod fire, da lead dømte).
- Leads to forkerte sider er rettet; hans fire gode er uændret gode.
- Fjernet andel af bredden: median 11,5 % (5,1-21,8 %).
- Alle 8 sider med fremmed tekst får strimlen uden for snittet.
- `sikker`-kolonnen mærker 2 sider (`273108_001555`, `37554_001496`) —
  begge er ved eftersyn i orden, så mærket er indtil videre kun set give
  falsk alarm. Det er stadig bedre end falsbeskæringens kolonne, som gav
  10/10 på alle 118 og dermed ikke skilte noget fra.

## Forsøg B — find kun de sider med et fremmed blad

Afgøres på bredden af det lyse bælte uden for kanten. Ikke på "kommer
papiret igen": også et rent bogsnit giver ~20 px lyst papir lige uden for
kanten, så det spørgsmål siger ja til alt.

| Facit siger blad | Sider | B siger ja |
|---|---|---|
| ja | 20 | 17 |
| nej | 90 | 25 |
| usikker | 8 | 3 |

B er ikke valgt. Den fanger færre af problemsiderne end A skærer rigtigt,
og rammer alligevel forbi på 25 af 90 rene sider.

## Hvad tallene IKKE viser

- Intet er målt på prøvemængdens 50 sider. De er urørte med vilje.
- Der er stadig ikke kørt ét modelkald. Om den fremmede strimmel
  overhovedet ender i en transskription, er **uafprøvet**.
- Kontaktarkene i `yderkant_snit_ark/` er ikke gennemset af lead under den
  endelige kode. Mine egne øjne på nedskalerede ark har taget fejl fire
  gange i dette arbejde og skal ikke regnes for en godkendelse.
