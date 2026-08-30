"""Én sammenligning af hele siden, i raekkefoelge, uden soegning.

Erstatter forankringen i `maal.py` (stage 03, plan af 2026-08-30). Forankringen
soegte hver facit-linje frem i modellens tekst fra sidste traef og fremefter. Paa
`273107_001864` staar "ingen Snue" TO gange i facit; soegningen efter linje 1
fandt det ordrette traef i linje 26, flyttede soegepunktet dertil, og de 24
mellemliggende linjer var derefter uden for raekkevidde. 26 af 29 linjer tabt --
nok til at vende hele forsoegets rangorden.

Her er der kun én vej gennem siden. Et gentaget ord kan ikke flytte noget, og
hele facit er altid med i naevneren: "daekning" og "rabat" findes ikke som
begreber laengere.

Den eneste afvigelse fra en almindelig redigeringsafstand er jokermaerket:
hvor facit siger `[?]`, kunne transskribenten ikke laese stedet, og modellen maa
skrive noget dér uden at det koster -- men kun op til `JOKER_LOFT` tegn
INDHOLD. Loftet findes, fordi en joker uden loft er praecis den ladeport,
forankringen var: den lader modellen springe vilkaarligt langt frem i sin egen
tekst gratis.

Maalt ved siden af, ikke fratrukket: hvor mange tegn modellen faktisk lagde i
hvert jokerfelt. Det er det eneste sted, tilboejeligheden til at digte kan ses.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass

from andenside import cer

# Jokermaerket i facit. Samme streng som `maal.MAERKE`; gentaget her frem for
# importeret, saa dette modul kan staa alene naar `maal.py` skaeres til i trin 3.
MAERKE = "[?]"

# Hvor mange tegn INDHOLD ét jokerfelt maa sluge gratis. Blanktegn taeller ikke
# med: normaliseringen aeder facits egne mellemrum omkring maerket, saa jokeren
# ogsaa maa sluge modellens. Talte de med, ville det effektive loft vaere 13 ved
# det typiske "ikke [?] suspect" og 15 ved "ikke[?]suspect" -- altsaa afhaengigt
# af skriverens typografi frem for af aftalen. Tallet er hentet fra
# materialet: 15 ligger over 99. percentil for ét ord i oevefacit (14 tegn), saa
# et `[?]`, der daekkede ét ord -- ogsaa et langt sammensat -- altid slipper
# igennem. 250 af de 354 maerker staar alene paa deres linje. Skriver modellen
# mere, koster overskuddet 1 pr. tegn (lead, 2026-08-30).
JOKER_LOFT = 15


@dataclass(frozen=True)
class SideMaal:
    """Tal for ÉN side under ÉN variant.

    `facit_tegn` er altid hele sidens kendte tekst. Jokermaerkerne taeller ikke
    med -- de er ikke kendt sandhed, og at lade dem staa i naevneren ville goere
    tallet paenere, jo mere ulaeseligt materialet var.
    """

    tegnafstand: int
    facit_tegn: int
    ordafstand: int
    facit_ord: int
    # Tegn modellen lagde i hvert jokerfelt, i sidens raekkefoelge. Det er
    # gab-tallet, og det viser ALT -- mellemrum med -- fordi gabet skal foere en
    # laeser hen til stedet.
    joker_tegn: tuple[int, ...]
    # Samme felter, men kun ikke-blanktegn. Det er DEM, loftet maales paa.
    joker_indhold: tuple[int, ...]

    @property
    def cer(self) -> float:
        return self.tegnafstand / self.facit_tegn if self.facit_tegn else 0.0

    @property
    def wer(self) -> float:
        return self.ordafstand / self.facit_ord if self.facit_ord else 0.0

    @property
    def jokere(self) -> int:
        return len(self.joker_tegn)

    @property
    def joker_tegn_i_alt(self) -> int:
        return sum(self.joker_tegn)

    @property
    def joker_overskud(self) -> int:
        """Indholdstegn ud over loftet -- den del af jokerne der faktisk kostede."""
        return sum(max(0, n - JOKER_LOFT) for n in self.joker_indhold)


# --------------------------------------------------------------------------
# Facit deles ved maerket FOER normalisering
# --------------------------------------------------------------------------

def _forbered_facit(
    facit: str, options: dict
) -> tuple[str, list[int], list[str], list[int]]:
    """Deler facit ved `[?]` og normaliserer hvert stykke for sig.

    Raekkefoelgen er ikke til forhandling: `normalize()` med
    `ignore_punctuation` fjerner firkantparenteserne, saa `[?]` bliver til den
    tomme streng og maerket forsvinder sporloest. Deles der foerst bagefter, er
    tre af de seks varianter uden jokere overhovedet.

    Returnerer facits tegn og ord uden maerkerne, plus de positioner (i tegn og
    i ord) hvor et jokerfelt ligger. Positionerne kan gentages: to maerker i
    traek giver to jokerfelter samme sted, hvert med sit eget loft.

    Bivirkning, som er tilsigtet: blanktegnene omkring maerket falder vaek med
    det, saa "ikke [?] suspect" bliver til "ikkesuspect". Jokerfeltet daekker
    altsaa stedet MED dets afgraensende mellemrum, og modellen kan skrive dem
    igen uden at betale -- se `_sluge_tegn`, hvor de derfor heller ikke taeller
    mod loftet.

    Prisen er, at naevneren bliver ét tegn mindre pr. maerke. Det er BEVIDST
    ikke rettet: skulle mellemrummet med i `ref`, skulle det ligge enten foer
    eller efter jokergraensen, og saa ville en model, der skriver sit gaet uden
    mellemrum omkring, blive straffet for det. Vi ville altsaa bytte en
    typografi-afhaengighed i loftet for en anden i taelleren. Tabet er 354
    maerker mod oevefacits knap 180.000 tegn -- under en femtedel procent, ens
    for alle seks varianter og i modellens disfavoer, altsaa ikke noget, der
    kan vende en rangorden.
    """
    stykker = [cer.normalize(s, **options) for s in facit.split(MAERKE)]

    ref = "".join(stykker)
    tegn_graenser: list[int] = []
    p = 0
    for stykke in stykker[:-1]:
        p += len(stykke)
        tegn_graenser.append(p)

    # Ordene splittes stykke for stykke. Splittes den sammenklaebede `ref` i
    # stedet, smelter ordet foer et maerke sammen med ordet efter.
    ref_ord: list[str] = []
    ord_graenser: list[int] = []
    for nr, stykke in enumerate(stykker):
        ref_ord.extend(stykke.split())
        if nr < len(stykker) - 1:
            ord_graenser.append(len(ref_ord))

    return ref, tegn_graenser, ref_ord, ord_graenser


def _graense_antal(laengde: int, graenser: list[int]) -> list[int]:
    """Hvor mange jokerfelter der ligger ved hver position 0..laengde."""
    antal = [0] * (laengde + 1)
    for p in graenser:
        antal[p] += 1
    return antal


# --------------------------------------------------------------------------
# Jokerfeltet som en operation paa én DP-raekke
# --------------------------------------------------------------------------

def _indholdstegn(hyp: str) -> list[int]:
    """`nb[j]` = antal ikke-blanktegn i `hyp[:j]`. Loftet maales paa den, saa et
    mellemrum hverken fylder i jokeren eller koster noget."""
    nb = [0] * (len(hyp) + 1)
    for j, tegn in enumerate(hyp, start=1):
        nb[j] = nb[j - 1] + (not tegn.isspace())
    return nb


def _sluge_tegn(raekke, nb: list[int], hyp: str, loft: int = JOKER_LOFT) -> list[int]:
    """Lader ét jokerfelt sluge tegn af modelteksten og relakserer raekken.

    `ud[j]` = billigste vej til modelposition `j`, naar jokeren maa aede et
    vilkaarligt stykke, der ender ved `j`. De foerste `loft` INDHOLDSTEGN er
    gratis, hvert indholdstegn derover koster 1. Blanktegn er altid gratis.

    Gratis-delen er et min over det vindue, hvor der er hoejst `loft`
    indholdstegn tilbage til `j`; `nb` er voksende, saa vinduets start findes
    med et binaert opslag. Den kostbare del falder ud af den loebende
    `ud[j-1] + kost`: er man naaet til `j-1` paa nogen maade, kan ét tegn mere
    altid koebes for 1 -- eller for 0, hvis det er et mellemrum.
    """
    m = len(raekke) - 1
    ud = [0] * (m + 1)
    for j in range(m + 1):
        bedst = min(raekke[bisect_left(nb, nb[j] - loft):j + 1])
        if j:
            videre = ud[j - 1] + (0 if hyp[j - 1].isspace() else 1)
            if videre < bedst:
                bedst = videre
        ud[j] = bedst
    return ud


def _sluge_ord(raekke: list[int], hyp_ord: list[str], loft: int = JOKER_LOFT) -> list[int]:
    """Samme operation paa ord.

    Loftet er stadig de 15 TEGN. Der findes ikke et selvstaendigt ordloft, og at
    opfinde ét ville vaere endnu et tal uden daekning i materialet. Jokeren
    sluger derfor gratis de ord, hvis bogstaver tilsammen holder sig inden for
    loftet; hvert ord derudover koster 1 ordfejl.
    """
    m = len(raekke) - 1
    ud = [0] * (m + 1)
    for j in range(m + 1):
        bedst = raekke[j]
        brugt = 0
        jm = j
        while jm > 0:
            brugt += len(hyp_ord[jm - 1])
            if brugt > loft:
                break
            jm -= 1
            if raekke[jm] < bedst:
                bedst = raekke[jm]
        if j and ud[j - 1] + 1 < bedst:
            bedst = ud[j - 1] + 1
        ud[j] = bedst
    return ud


def _kilden_til(foer, efter, j: int, nb: list[int], loft: int = JOKER_LOFT) -> int:
    """Hvor i modelteksten jokerfeltet begyndte, naar det endte ved `j`.

    Ved lige billige veje vaelges det MINDSTE `jm`, altsaa det stoerste slugte
    stykke. To grunde: valget skal vaere fast, ellers kunne to koersler af
    uaendret kode give hvert sit gab-tal -- og skriver modellen mere end loftet,
    kan overskuddet ligge enten i jokeren eller som indsaettelser i den
    omgivende tekst til samme pris. Gabet er det, der skal ses efter i haanden,
    saa den tvivl skal falde ud til at vise ALT, modellen lagde paa stedet.
    """
    for jm in range(0, j + 1):
        if foer[jm] + max(0, nb[j] - nb[jm] - loft) == efter[j]:
            return jm
    return j  # kan ikke naas: jm == j er altid en gyldig vej


# --------------------------------------------------------------------------
# Maalingen
# --------------------------------------------------------------------------

def _tegnmaaling(
    ref: str, graenser: list[int], hyp: str
) -> tuple[int, list[int], list[int]]:
    """Redigeringsafstand ref->hyp med jokerfelter, plus hvad hver joker slugte.

    Returnerer afstanden, gabene (alle tegn) og de samme gab talt i
    indholdstegn -- gabet skal foere en laeser hen til stedet, loftet skal
    straffe, og de to tal maa derfor ikke vaere det samme.

    Almindelig tabel-DP. Hver raekke gemmes som `array("i")`: en side paa 2.500
    tegn mod et svar paa 3.000 er 7,5 mio. celler, og som Python-lister ville
    tabellen fylde et par hundrede megabyte alene i heltalsobjekter.
    """
    n, m = len(ref), len(hyp)
    antal = _graense_antal(n, graenser)
    nb = _indholdstegn(hyp)

    raekke: list[int] = list(range(m + 1))
    # (ref-position, raekken foer jokeren, raekken efter). Ét lag pr. jokerfelt,
    # i sidens raekkefoelge -- ogsaa naar to maerker staar samme sted.
    lag: list[tuple[int, list[int], list[int]]] = []
    for _ in range(antal[0]):
        foer, raekke = raekke, _sluge_tegn(raekke, nb, hyp)
        lag.append((0, foer, raekke))

    tabel: list[array] = [array("i", raekke)]
    for i in range(1, n + 1):
        forrige = raekke
        tegn = ref[i - 1]
        ny = [forrige[0] + 1]
        for j in range(1, m + 1):
            ny.append(min(
                forrige[j] + 1,
                ny[j - 1] + 1,
                forrige[j - 1] + (tegn != hyp[j - 1]),
            ))
        for _ in range(antal[i]):
            foer, ny = ny, _sluge_tegn(ny, nb, hyp)
            lag.append((i, foer, ny))
        raekke = ny
        tabel.append(array("i", ny))

    # Tilbagesporing. Kun for at faa fat i jokerfelternes laengder -- afstanden
    # selv staar i sidste celle. `aktuel` er den raekke, positionen (i, j) hoerer
    # til lige nu: ved en jokergraense skiftes den ud med raekken FOER jokeren,
    # for det er den, det normale skridt nedad blev regnet fra.
    slugt = [0] * len(graenser)
    indhold = [0] * len(graenser)
    i, j = n, m
    aktuel = tabel[n]
    lag_nr = len(lag) - 1
    while i > 0 or j > 0:
        while lag_nr >= 0 and lag[lag_nr][0] == i:
            _, foer, efter = lag[lag_nr]
            jm = _kilden_til(foer, efter, j, nb)
            slugt[lag_nr] = j - jm
            indhold[lag_nr] = nb[j] - nb[jm]
            j, aktuel = jm, foer
            lag_nr -= 1
        if i == 0 and j == 0:
            break
        forrige = tabel[i - 1] if i else None
        if i and j and aktuel[j] == forrige[j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j, aktuel = i - 1, j - 1, forrige
        elif i and aktuel[j] == forrige[j] + 1:
            i, aktuel = i - 1, forrige
        else:
            j -= 1

    return tabel[n][m], slugt, indhold


def _ordmaaling(ref_ord: list[str], graenser: list[int], hyp_ord: list[str]) -> int:
    """Samme maaling paa ord. Ingen tilbagesporing -- gabene opgoeres i tegn."""
    n, m = len(ref_ord), len(hyp_ord)
    antal = _graense_antal(n, graenser)

    raekke = list(range(m + 1))
    for _ in range(antal[0]):
        raekke = _sluge_ord(raekke, hyp_ord)
    for i in range(1, n + 1):
        forrige = raekke
        ordet = ref_ord[i - 1]
        ny = [forrige[0] + 1]
        for j in range(1, m + 1):
            ny.append(min(
                forrige[j] + 1,
                ny[j - 1] + 1,
                forrige[j - 1] + (ordet != hyp_ord[j - 1]),
            ))
        for _ in range(antal[i]):
            ny = _sluge_ord(ny, hyp_ord)
        raekke = ny
    return raekke[m]


def maal_side(facit: str, model: str, **options) -> SideMaal:
    """Maaler modellens fulde sidetekst mod facits fulde sidetekst, i raekkefoelge.

    `options` er dem, `cer.normalize()` tager, saa en variant maales med
    `maal_side(facit, model, **cer.VARIANTER["arbejdstal"])`.
    """
    ref, tegn_graenser, ref_ord, ord_graenser = _forbered_facit(facit, options)
    hyp = cer.normalize(model, **options)
    hyp_ord = hyp.split()

    tegnafstand, slugt, indhold = _tegnmaaling(ref, tegn_graenser, hyp)
    return SideMaal(
        tegnafstand=tegnafstand,
        facit_tegn=len(ref),
        ordafstand=_ordmaaling(ref_ord, ord_graenser, hyp_ord),
        facit_ord=len(ref_ord),
        joker_tegn=tuple(slugt),
        joker_indhold=tuple(indhold),
    )
