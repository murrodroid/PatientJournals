# Falssnittets buffer — samme side ved tre værdier

Lavet 2026-08-30, efter lead skrev at falssnittet "sidder stadig lidt langt
inde på den anden side på nogle, men ingen er gået ægte galt".

Hver fil viser **samme side tre gange**: buffer 2,0 % (som i dag), 1,0 % og
0,5 %. Det røde er det, snittet fjerner. Kun den halvdel, falsen ligger i,
er vist.

## Hvad knappen gør

`skraa.BUFFER_ANDEL` flytter snittet **væk fra vores egen tekst** og ind mod
falsen. Stor buffer = mere af naboens strimmel bliver stående, men vores
ordender er sikre. Lille buffer = renere snit mod naboen, men risiko for at
barbere skrift, hvor siden krummer ind i folden.

På en side på ~1700 px er 2,0 % ≈ 34 px, 1,0 % ≈ 17 px, 0,5 % ≈ 8 px.

## Hvorfor den står på 2 % i dag

Den blev sat **op** fra 1 % den 2026-08-27, fordi 1 % ikke reddede de
nederste linjer på `273104_001639` — dér krummer siden ind mod falsen, og
skriveren skrev helt ud. Den side er derfor med i sammenligningen; den er
prøvestenen for, hvor lavt bufferen kan gå.

## Sider i mappen

| Side | Hvorfor med |
|---|---|
| `273104_001639`, `_001640` | grunden til at bufferen blev sat op til 2 % |
| `273098_001496`, `273100_001258` | almindelige sider fra øvemængden |
| `273103_001437`, `273105_001569`, `37554_001492`, `273111_001376` | sider lead selv har dømt tidligere |
| `273069_000072` | en af dem, hvor falssnittet var gået galt før rettelsen |

## Om målingen, der ikke blev til noget

Der blev bygget et forsøg på at måle luften mellem snittet og vores egen
skrift, så valget kunne træffes på et tal. **Det bestod ikke sin egen
prøve**: falsen er selv mørk og tæller som blæk, så margenen kom ud negativ
på alle 307 sider — også dem, lead netop havde godkendt. Det er tredje gang
et "klipper vi blæk"-mål strander på præcis det. Målet er kasseret, ikke
trimmet, og valget lægges frem for øjnene i stedet.
