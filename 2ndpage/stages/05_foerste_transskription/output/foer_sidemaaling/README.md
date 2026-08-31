# Arkiv: målinger fra FØR sidemålingen

Disse rapporter og gab-filer er regnet med det **gamle måleapparat**, som
forankrede: hver facit-linje blev søgt frem i modellens tekst, og en linje, der
ikke kunne findes, faldt helt ud af målingen — i både tæller og nævner.

**Tallene her gælder ikke længere.** De står her alligevel, og det er med vilje.

## Hvorfor de bevares

Den vigtigste læring fra 30. august er, at det gamle apparat **kunne vende en
rangorden**. På `273107_001864` står vendingen "ingen Snue" to gange i facit
selv. Søgningen efter linje 1 fandt det ordrette match nede i linje 26, flyttede
søgepunktet dertil, og de 24 mellemliggende linjer var derefter uden for
rækkevidde — 26 af 29 linjer tabt på én side. Det gjorde forsøgets bedste
variant til dens dårligste.

Den læring kan kun efterprøves, hvis de gamle tal stadig findes. Slettes de,
står dagbogen med tal, ingen kan kontrollere.

## Hvad "dækning" betød her, og hvorfor det var farligt

Hver rapport i denne mappe opgør en **dækning**: hvor stor en del af facits tegn
målingen faktisk nåede at måle på. Den lå typisk på 88-94 %.

Dækningen var **glidende og variantafhængig**. Jo mere en variant fik modellen
til at afvige fra siden, jo flere linjer kunne søgningen ikke finde, jo mere
tekst faldt ud af målingen — og jo bedre så varianten ud. Det er en rabat, der
belønner netop den variant, der læser dårligst.

Begrebet findes ikke i de nye rapporter. Hele siden er altid målt.

Den nye rapport har stadig ét tal, der ligner: den strenge målings faste
udeladelse af linjer med et `[?]`. Det er **ikke** det samme. Den afhænger
udelukkende af facit, er den samme for alle varianter og alle modeller, og kan
derfor ikke give nogen rabat.

## Hvordan filerne er fremkommet

Fire kørsler (`161626`, `161900`, `162704`, `164036`) er de originale filer fra
30. august, urørte.

De øvrige tolv blev overskrevet ved en fejl den 31. august, før arkiveringen var
gjort. De er **genskabt** ved at køre den arkiverede kode fra commit `3391110`
på de samme, urørte modelsvar. Det gamle apparat er deterministisk — det er
efterprøvet i fuld skala tidligere — så de genskabte filer er tegn for tegn, hvad
der stod der før. Modelsvarene (`svar/`, `raa_skemasvar.json`) er aldrig blevet
rørt.

## Facit-udgaven

Alle tal — gamle som nye — er regnet på `alt_*`-udgaven af facit
(`stages/02_facit/output/facit.jsonl`), sådan som den stod 31. august 2026.
Rettes facit senere, er hverken disse eller de nye tal sammenlignelige med tal
regnet bagefter.
