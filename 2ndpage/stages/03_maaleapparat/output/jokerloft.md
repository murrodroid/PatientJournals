# Hvor meget flytter jokerloftet den strenge måling?

Målt 2026-08-31 på alle 16 gemte kørsler. Ingen modelkald — svarene lå på disken.

## Hvorfor spørgsmålet opstod

Den strenge måling (beslutning 44) udelader hele linjer med et `[?]`. Under
forankringen kunne den udelade modellens modstykke til sådan en linje, fordi
forankringen udpegede det. Uden forankring findes der intet modstykke at
udelade, og den eneste vej er at erstatte hele linjen med ét jokermærke.

Så skal det mærke have et **loft**: hvor meget må modellen skrive dér, uden at
det koster? Det er en knap, og dette projekt har erfaring med knapper, der
pynter — `MAKS_AFVIGELSE` kunne sænke tegnfejlen fra 7,50 % til 7,13 % og se
pænere ud, mens færre og lettere linjer blev målt.

Derfor er knappen målt, ikke valgt.

## Målingen

Tegnfejl (`arbejdstal`), gennemsnit over de 16 kørsler:

| Loft for en udeladt linje | Strengt tal | Forhold til hovedtallet |
|---|---:|---|
| Linjens egen længde **(valgt)** | 11,5 % | ca. 1 procentpoint HØJERE |
| Linjens længde + 15 | 9,7 % | lavere |
| Slet intet loft | 9,7 % | lavere |
| Fast 15 (som et `[?]` inde i en linje) | 19,6 % | voldsomt højere |

Hovedtallet ligger til sammenligning på 10,3 % i samme opgørelse.

## Hvad målingen viser

**"Linjens længde + 15" er ikke et mildt slæk — det er slet intet loft.** De to
kolonner giver nøjagtig samme tal i 15 af de 16 kørsler. Loftet blev med andre
ord aldrig bindende, og varianten var i praksis en fribillet uden grænse. Det
var ikke tydeligt på forhånd og er grunden til, at de 15 tegn blev fjernet igen.

**Et fast loft på 15 kan ikke bruges.** De 15 tegn er udledt af, hvor langt ét
ord er i materialet, og er det rigtige mål for et `[?]` inde i en linje. En hel
linje er typisk 25-45 tegn, så modellens tekst for den udeladte linje løber langt
over, og overskuddet tælles som fejl. Tallet fordobles og måler noget andet, end
det giver sig ud for.

**Linjens egen længde er den eneste af de fire, der ikke er systematisk mildere
end hovedtallet.** Det er det afgørende: en strengere måling, der er mildere end
den almindelige, kan ikke bruges som kontrol af den.

## Følge, som skal afgøres

Med det valgte loft ligger den strenge måling **konsekvent ca. 1 procentpoint
over hovedtallet** — i alle 16 kørsler, uden undtagelse.

Beslutning 44 siger: *"Er den strenge højere end hovedtallet, har redningen
pyntet, og så er det den strenge, der gælder."* Den regel blev skrevet, da den
strenge måling var et værn mod forankringens glidende rabat. **Den rabat findes
ikke længere**, og reglen ville nu udløses hver eneste gang.

Forskellen har en anden og helt uskyldig forklaring: hovedtallet har en
fribillet ved hvert `[?]`, som den strenge måling ikke har, fordi den slet ikke
ser de linjer. At hovedtallet ligger lavere er derfor forventeligt og ikke
et tegn på, at noget er blevet pyntet.

Reglens formulering bør revideres. Det er ikke en kodeændring, og det er ikke
agentens kald — det står som et åbent punkt til lead.
