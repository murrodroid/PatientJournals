"""Omrokering: hvor mange facit-linjer staar i forkert indbyrdes raekkefoelge.

Baggrund (lead-beslutning 2026-08-30): maalingen sammenligner nu hele siden
i raekkefoelge, i stedet for at soege hver linje frem i modelteksten uden
hensyn til hvor de andre linjer landede. Lead oenskede foerst, at maalingen
skulle TILGIVE ombyttede linjer -- afvist, fordi raekkefoelgen er data: en
patientjournal laeses kronologisk, og resultatet afleveres som `PageLine`-
poster til en anden app, hvor raekkefoelgen betyder noget. En ombyttet linje
er derfor en fejl, man vil SE, ikke en fejl der skal glattes ud.

Beslutningen blev: maal strengt i raekkefoelge (det ligger i `maal.py`), men
opgoer omrokering som sit eget tal ved siden af. Det er dette modul.

## Fremgangsmaade

1. Par hver facit-linje med den modellinje, den ligner mest (Levenshtein paa
   normaliseret tekst). En linje uden et rimeligt modstykke er UPARRET og
   taeller ikke som omrokeret -- den kan jo ikke vaere "paa forkert plads",
   naar den slet ikke er fundet.
2. Laes af, hvilken position hver parret facit-linje fik i modelteksten. Det
   giver en talraekke, én per parret facit-linje, i facits egen raekkefoelge.
3. Det MINDSTE antal linjer, der skal flyttes for at faa raekken paa plads,
   er antallet af linjer minus laengden af den laengste voksende delfoelge
   (LIS). Det er IKKE det samme som antallet af inversioner: to ombyttede
   nabolinjer giver ét fejlplaceret par, men taeller som ÉN inversion pr.
   fejlplaceret par ganget op mod alt de krydser -- for to naboer bliver det
   ganske vist ogsaa 1, men for en linje flyttet fra top til bund giver
   inversion n-1, mens LIS-maalet korrekt siger 1 flytning. Inversionstal
   overdriver derfor kraftigt, jo laengere en fejlplaceret linje "rejser".

## Kendt svaghed ved parringen -- undersoegt, ikke fjernet

Grov "naermeste match uden hensyn til andre linjer" ville lade to
facit-linjer matche SAMME modellinje, hvis linjerne minder om hinanden --
og det goer de tit i journalmateriale (gentagne vitalvaerdier: "Puls 80.",
"Puls 80.", "Temperatur 37."). Faar to facit-linjer samme model-position,
bliver "hvilken raekkefoelge stod de i" meningsloest for dem.

Derfor parres her ét-til-ét: hver modellinje bruges hoejst én gang. Facit-
linjerne behandles i deres egen raekkefoelge, og hver faar den bedste
LEDIGE modellinje. Det fjerner kollisionen, men loeser ikke det dybere
problem: et venstre-mod-hoejre grådigt valg kan stadig laase en tidlig
facit-linje fast paa en model-linje, der reelt hoerte til en senere
facit-linje, naar flere linjer er naesten ens. Det er IKKE rettet her --
en fuld loesning kraever global optimal tildeling (fx Ungarsk algoritme
over alle facit x model-linjer), hvilket er en stoerre aendring end denne
opgave daekker. Se rapportens vurdering for konkret eksempel.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from andenside import cer

# Hvor meget den normaliserede afstand maa vaere, som andel af den laengste af
# de to linjers laengde, foer parringen opgiver og siger "uparret". Sat
# rundhaandet ligesom `maal.MAKS_AFVIGELSE`, fordi en for stram graense bare
# ville skjule daarlige match som "ingen linje fundet" i stedet for at vise
# dem.
MAKS_AFVIGELSE = 0.5


def _normaliseret(tekst: str) -> str:
    """Fælles normalisering foer sammenligning: uden versaler, diakritika og
    tegnsaetning. Parringen skal se paa INDHOLD, ikke paa smaa laesefejl --
    de maales i forvejen af `cer`, og skal ikke ogsaa forstyrre parringen."""
    return cer.normalize(
        tekst,
        ignore_case=True,
        ignore_diacritics=True,
        ignore_punctuation=True,
    )


def _afstand(a: str, b: str) -> float:
    """Normaliseret Levenshtein-afstand mellem 0 og 1 (0 = ens)."""
    if not a and not b:
        return 0.0
    return cer.levenshtein(a, b) / max(len(a), len(b))


def _par_linjer(
    facit_linjer: list[str], model_linjer: list[str], maks_afvigelse: float
) -> list[int | None]:
    """Finder for hver facit-linje den bedste LEDIGE modellinje.

    Returnerer en liste lige saa lang som `facit_linjer`, med modellinjens
    indeks eller `None` for uparret. Ét-til-ét: naar en modellinje er brugt,
    kan den ikke bruges igen (se modulets docstring om hvorfor).

    Behandles i facits egen raekkefoelge, saa resultatet er deterministisk
    for et givent input -- der itereres aldrig over en mængde eller ordbog.
    """
    model_norm = [_normaliseret(m) for m in model_linjer]
    ledige = list(range(len(model_linjer)))
    resultat: list[int | None] = []
    for facit_linje in facit_linjer:
        f_norm = _normaliseret(facit_linje)
        bedste_idx: int | None = None
        bedste_afstand = float("inf")
        for idx in ledige:
            d = _afstand(f_norm, model_norm[idx])
            if d < bedste_afstand:
                bedste_afstand = d
                bedste_idx = idx
        if bedste_idx is not None and bedste_afstand <= maks_afvigelse:
            resultat.append(bedste_idx)
            ledige.remove(bedste_idx)
        else:
            resultat.append(None)
    return resultat


def _laengste_voksende_delfoelge(tal: list[int]) -> int:
    """Laengden af den laengste STRENGT voksende delfoelge.

    Patience-sortering, O(n log n). `haler[k]` er det mindst mulige
    sluttal for en voksende delfoelge af laengde k+1 set indtil videre --
    et standardtrick, men her ekstra vigtigt at faa rigtigt, fordi hele
    omrokerings-tallet hviler paa det (linjer_i_alt - LIS).
    """
    haler: list[int] = []
    for t in tal:
        i = bisect_left(haler, t)
        if i == len(haler):
            haler.append(t)
        else:
            haler[i] = t
    return len(haler)


@dataclass(frozen=True)
class Omrokering:
    """Resultatet af at sammenligne facits raekkefoelge med modellens."""

    antal_flyttede: int    # linjer_parret - laengste voksende delfoelge
    linjer_parret: int      # facit-linjer der fik en modellinje
    linjer_uparret: int     # facit-linjer uden rimeligt modstykke
    model_positioner: tuple[int, ...]  # parrede linjers position i modellen, i facit-raekkefoelge
    # Parrede linjer, hvor modellens tekst er ens med facits efter samme
    # normalisering, parringen selv bruger. Under forankringen kom "andel
    # noejagtig rigtige linjer" fra `Maaltal.andel_identiske`, hvor ét stykke
    # var én parret linje. Sidemaalingen maaler hele siden i ét straek, saa dér
    # er ét stykke én SIDE -- et andet tal. Her er det oprindelige.
    linjer_identiske: int = 0


def maal_omrokering(
    facit_linjer: list[str],
    model_linjer: list[str],
    *,
    maks_afvigelse: float = MAKS_AFVIGELSE,
) -> Omrokering:
    """Taeller, hvor mange af facits linjer staar i en anden indbyrdes
    raekkefoelge end i modellens tekst.

    Uparrede linjer (intet rimeligt modstykke i modellen) taeller IKKE som
    omrokerede -- de kan ikke vaere "paa forkert plads" naar de slet ikke er
    fundet. De opgoeres for sig i `linjer_uparret`.
    """
    par = _par_linjer(facit_linjer, model_linjer, maks_afvigelse)
    positioner = [idx for idx in par if idx is not None]
    lis = _laengste_voksende_delfoelge(positioner)
    identiske = sum(
        1
        for facit_linje, idx in zip(facit_linjer, par)
        if idx is not None
        and _normaliseret(facit_linje) == _normaliseret(model_linjer[idx])
    )
    return Omrokering(
        antal_flyttede=len(positioner) - lis,
        linjer_parret=len(positioner),
        linjer_uparret=len(par) - len(positioner),
        model_positioner=tuple(positioner),
        linjer_identiske=identiske,
    )
