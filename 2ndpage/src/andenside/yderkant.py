"""Sidens YDRE kant -- den modsatte af falsen.

`skraa.py` renser falssiden. Paa den modsatte kant ligger enten bogsnittet
(bogblokkens sammenpressede sidekanter set fra siden -- brunt, uden laesbar
tekst) eller et blad laengere inde i bindet, som er faldet fladt ud og
blevet fotograferet med. I sidste tilfaelde staar der FREMMED haandskrift
langs kanten, som falsbeskaeringen aldrig kan naa.

Maalt paa oevemaengden (`stages/04_billedforberedelse/output/yderkant_facit.csv`):
7 af 118 sider har fremmed tekst dér; 19 har et synligt udragende blad.

## Hvad der maales, og hvorfor ikke noget andet

**Ikke gennemsnittet pr. kolonne.** Maalt 2026-08-28 paa `37554_001492`:
middelvaerdien ligger fladt paa 202-210 hen over baade vores side og det
fremmede blad -- blaekket droerner kanten. Ubrugelig.

**Papirets grundlyshed** -- en hoej percentil pr. kolonne -- viser derimod
bladets skygge som et fald paa 5-25 gráatoneniveauer, mens en tekstlinje
ikke rører percentilen, saa laenge papiret ses mellem bogstaverne.

**Baandvist, ikke over hele hoejden.** Bladet ligger skaevt: paa
`37554_001492` vandrer skyggen fra x=1134 foroven til x=1172 forneden.
Maalt over hele hoejden smøres de 38 px ud, og faldet halveres. Samme
erkendelse som for falsen -- derfor genbruges baandgeometrien fra `skraa`.

## Kanten er den samme i begge tilfaelde

Sveepet 2026-08-28 viste, at ogsaa en side UDEN udragende blad har et lille
fald dér, hvor vores eget papir slipper. Det er sidens egen kant, og den er
det rigtige sted at skaere uanset hvad der ligger udenfor. Detektionen er
derfor ÉN regel (forsoeg A). Spoergsmaalet om, hvorvidt der ligger et
fremmed blad (forsoeg B), afgøres bagefter paa BREDDEN af det lyse baelte
uden for kanten -- ikke paa, om der overhovedet kommer papir igen. Ogsaa
bogsnittet giver nogle faa lyse pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from andenside.bogryg import SoegeVindue, smooth
from andenside.masterlist import Side
from andenside.skraa import ANTAL_BAAND, MARGEN, OVERLAP

YDRE_ANDEL = 0.30      # hvor stor en del af bredden der soeges i
PERCENTIL = 85         # papirets grundlyshed; blaek ligger under
MIN_FALD = 5.0         # mindste troevaerdige fald i graatoneniveauer
FALD_SPAND = 12        # ... som skal ske inden for saa faa kolonner
# 12, ikke 25: det er HAELDNINGEN, der skiller en kant fra belysning.
# Bladets skygge paa 37554_001492 falder ~10 niveauer paa 5 px (2,0/px);
# billedernes jaevne afdaempning ud mod kanten falder 60 niveauer paa 300 px
# (0,2/px). Med spand 25 er graensen 0,20/px -- altsaa praecis paa
# afdaempningen, og detektionen skaerer midt paa vores egen side. Med 12 er
# graensen 0,42/px: stadig fem gange under bladets skygge.
BUFFER_ANDEL = 0.006   # flyttes UDAD, saa intet af vores eget papir ryger
# Halveret fra 1,2 % 2026-08-28 efter leads gennemsyn: bufferen gav for
# meget af det fremmede blad tilbage. Den er ren margen nu -- kanten meldes
# allerede i faldets bund, altsaa paa selve soemmen mellem vores papir og
# naboens, saa der er ikke laengere en indad-skaevhed at kompensere for.
# Paa 1300 px er det ~8 px.
BLAD_MIN_BREDDE = 45   # lyst baelte bredere end dette = et fremmed blad
BLAD_MIN_NIVEAU = 0.85 # ... og mindst saa lyst som vores eget papir
LINJE_TOLERANCE = 10   # px et baand maa afvige fra den rette kant
MAX_HAELDNING = 90     # px kanten kan flytte sig over hele sidens hoejde
# 90: falsens maalte haeldning naaede 88 px paa oevemaengden, og en
# yderkant kan ikke haelde mere end falsen paa samme opslag.
SAMME_KANT = 40        # px: linjer taettere end dette regnes for SAMME kant
SOEM_GULV = 6.0        # mindste soem-dybde, foer en linje overhovedet taeller
# Sidens kant er en FYSISK ting: papiret slipper, og kanten kaster en smal
# skygge. En linje trukket hen over aabent papir goer ikke. Maalt 2026-08-29
# paa de linjer, lead doemte forkerte: 3,0 og 5,0. De rigtige kanter maaler
# 10-163 -- paa naer 273107_001866, hvis rigtige kant kun naar 5-7. Der er
# altsaa OVERLAP ved 5, og gulvet ligger lige over det hoejeste maalte
# falske. Marginen er tynd; det er loesningens svageste led.
SOEM_HALV = 8          # halv bredde af det baand, soemmen soeges i
SOEM_OMKRING = 40      # ... og af naboskabet, papirets niveau maales i
MIN_STOETTE = 6        # saa mange baand skal bekraefte en kant, foer den taeller
# 6 af 24: paa `37554_001496` bekraefter kun 9 baand vores egen kant, mens
# alle 24 ser bladets. Saettes gulvet hoejere, tabes netop den side.
# Sider, der kun lige klarer gulvet, bliver til gengaeld mærket usikre.


@dataclass(frozen=True)
class YdreBeskaering:
    """Hvad der skete ude ved yderkanten."""

    billede: str
    recto_verso: str
    bredde_foer: int
    bredde_efter: int
    haeldning_px: int
    baand_med_kant: int
    baand_i_alt: int

    @property
    def sikker(self) -> bool:
        # Under halvdelen af baandene fandt en kant -> graensen hviler paa
        # for lidt, og siden skal ses efter med oejnene i stedet.
        return self.baand_med_kant * 2 >= self.baand_i_alt


@dataclass(frozen=True)
class BladFund:
    """Ligger der et fremmed blad uden for kanten? (forsoeg B)"""

    er_blad: bool
    baelte_bredde: int   # hvor bredt det lyse baelte uden for kanten er
    niveau_min: float    # laveste vaerdi i selve kantens skygge
    niveau_efter: float  # papirets niveau uden for kanten
    baand_med_blad: int
    baand_i_alt: int


def ydre_vindue(side: Side, bredde: int, *, andel: float = YDRE_ANDEL) -> SoegeVindue:
    """Afgraenser yderkanten -- spejlet af `bogryg.soegevindue`.

    Recto har falsen til venstre, altsaa yderkanten til HOEJRE; verso
    omvendt. `retning` siger hvilken kant vinduet ligger ved, praecis som i
    `bogryg`, saa gennemloebet nedenfor kan skrives éns for begge.
    """
    vindue_bredde = int(bredde * andel)
    if side.recto_verso == "recto":
        return SoegeVindue(start=bredde - vindue_bredde, slut=bredde, retning="fra_hoejre")
    if side.recto_verso == "verso":
        return SoegeVindue(start=0, slut=vindue_bredde, retning="fra_venstre")
    raise ValueError(
        f"{side.image_name}: recto/verso er '{side.recto_verso}', kan ikke afgoere yderkanten"
    )


def papir_profil(graa: np.ndarray, y0: int, y1: int, *, percentil: int = PERCENTIL,
                 step: int = 3) -> list[float]:
    """Papirets grundlyshed pr. kolonne i baandet `y0:y1`.

    Percentilen, ikke gennemsnittet: blaek skal ikke kunne tages for en
    kant. `step` springer raekker over -- percentilen er stabil nok til det,
    og det er den dyreste del af regnestykket.
    """
    return smooth(np.percentile(graa[y0:y1:step, :], percentil, axis=0).tolist(), window=5)


def _kant_i_profil(profil: list[float], vindue: SoegeVindue, *,
                   min_fald: float = MIN_FALD, spand: int = FALD_SPAND) -> int | None:
    """Foerste betydelige FALD paa vejen udad -- dér slipper vores papir.

    Kanten meldes i faldets BUND. Faldets begyndelse ligger op til
    `spand` kolonner inde paa vores egen side, og et snit dér barberer
    ordenderne (paavist af lead 2026-08-28).

    Faldet skal ske inden for `spand` kolonner. Uden det krav ville
    billedernes jaevne afdaempning ud mod kanten -- 60 niveauer fordelt over
    300 px paa nogle sider -- taelle som en kant midt paa vores egen side.
    """
    kandidater = _kandidater_i_profil(profil, vindue, min_fald=min_fald, spand=spand)
    return kandidater[0] if kandidater else None


def _kandidater_i_profil(profil: list[float], vindue: SoegeVindue, *,
                         min_fald: float = MIN_FALD,
                         spand: int = FALD_SPAND) -> list[int]:
    """ALLE betydelige fald paa vejen udad, inderst foerst.

    Der er som regel mere end ét: vores egen sidekant, dernaest hvert
    udragende blads kant, til sidst baggrunden. Hvilket af dem der er
    VORES, kan et enkelt baand ikke afgøre -- det kraever, at baandene ses
    under ét (`_bedste_linje`).
    """
    if vindue.retning == "fra_hoejre":
        raek = range(vindue.start, vindue.slut)      # udad = mod hoejre
        udad = 1
    else:
        raek = range(vindue.slut - 1, vindue.start - 1, -1)   # udad = mod venstre
        udad = -1

    ud: list[int] = []
    sidste = None
    for x in raek:
        slut = x + udad * spand
        if not (0 <= slut < len(profil)):
            slut = len(profil) - 1 if udad > 0 else 0
        lav, hoej = min(x, slut), max(x, slut)
        vaerdier = profil[lav : hoej + 1]
        if not vaerdier:
            continue
        if profil[x] - min(vaerdier) >= min_fald:
            # Kanten laegges i faldets BUND, ikke ved dets begyndelse.
            # Begyndelsen ligger op til `spand` kolonner INDE paa vores
            # egen side, og lead paaviste 2026-08-28, at snittet derfor
            # barberede tre-fire bogstaver af ordenderne. Hellere lade
            # en flig af naboen staa end at klippe vores egen skrift:
            # den fremmede strimmel er 90-140 px bred, saa de faa
            # kolonner koster ikke dens frasortering.
            x = min(range(lav, hoej + 1), key=lambda i: profil[i])
            # kun ét kandidatpunkt pr. fald -- ellers ville hver eneste
            # kolonne i et bredt fald blive talt med som sit eget
            if sidste is None or abs(x - sidste) > spand:
                ud.append(x)
            sidste = x
    return ud


def soem_dybde(graa: np.ndarray, skaering: float, haeldning: float, *,
               halv: int = SOEM_HALV, omkring: int = SOEM_OMKRING) -> float:
    """Hvor dyb en fordybning en linje ligger i, maalt ned gennem hele siden.

    Papirets eget niveau tages som 85-percentilen i et naboskab omkring
    linjen, og derfra traekkes det moerkeste punkt taet paa linjen. Et snit
    paa sidens kant ligger i en maerkbar fordybning; et snit midt paa
    papiret goer ikke.

    Returnerer -1 hvis linjen forlader billedet.
    """
    hoejde, bredde = graa.shape
    dybder = []
    for y in range(int(hoejde * 0.10), int(hoejde * 0.90), 9):
        x = int(round(skaering + haeldning * y / hoejde))
        if not (0 <= x < bredde):
            return -1.0
        a, b = max(0, x - omkring), min(bredde, x + omkring + 1)
        naboskab = graa[y, a:b]
        if naboskab.size < 20:
            continue
        lokal = graa[y, max(0, x - halv) : min(bredde, x + halv + 1)]
        if lokal.size:
            dybder.append(float(np.percentile(naboskab, 85) - lokal.min()))
    return float(np.median(dybder)) if dybder else -1.0


def _bedste_linje(
    baand: list[tuple[int, list[int]]], hoejde: int, vindue: SoegeVindue, *,
    tolerance: int = LINJE_TOLERANCE, max_haeldning: int = MAX_HAELDNING,
    min_stoette: int = MIN_STOETTE,
    graa: "np.ndarray | None" = None,
) -> list[int | None]:
    """Vaelger den rette linje, flest baand kan enes om -- og den inderste af dem.

    Hvert baand melder flere kandidater: vores egen sidekant, hvert
    udragende blads kant, baggrunden. Set alene kan et baand ikke vide,
    hvilken der er vores; set under ét kan de, for en sidekant er RET.

    Maalt paa `37554_001494`: vores egen skygge er saa svag, at nogle baand
    springer den over og finder det naeste blads kant 90 px laengere ude.
    Uden dette trin zigzaggede snittet mellem de to.

    Blandt de linjer, der har stoette nok, vaelges den INDERSTE -- ikke den
    med flest stoetter. Det er afgørende: et blad ligger altid uden for
    vores egen side, aldrig inden for, saa den inderste rette kant er
    vores. Maalt paa `37554_001496` findes vores egen kant kun i 9 af 24
    baand, mens bladets ses i alle 24; "flest baand vinder" ville derfor
    vaelge bladets kant og lade den fremmede tekst blive staaende.

    Naar faa baand baerer linjen, bliver siden til gengaeld mærket usikker
    laengere oppe -- det er dét, `baand_med_kant` taeller.
    """
    udad = 1 if vindue.retning == "fra_hoejre" else -1
    stoettende = [k for _, k in baand]

    ys = [float(y) for y, _ in baand]
    # Den samme fysiske kant foreslaas af hvert eneste baand, der ser den, og
    # for hver haeldning. Linjen kendes entydigt af sit skaeringspunkt ved
    # y=0 og sin haeldning, saa gentagelserne kan springes over. Det er her
    # tiden ligger: uden det proeves ~11.000 linjer pr. side, med det under
    # en tiendedel. (Numpy hjaelper ikke -- tabellerne er 24x5, og
    # kaldsomkostningen er stoerre end regnestykket. Maalt, ikke gaettet.)
    set_af_linjer: set[tuple[int, int]] = set()
    kandidat_linjer: list[tuple[float, int, float, float, float]] = []

    for y_anker, kandidater in baand:
        for x0 in kandidater:
            for haeldning in range(-max_haeldning, max_haeldning + 1, 2):
                skaering = x0 - haeldning * y_anker / hoejde
                noegle = (round(skaering), haeldning)
                if noegle in set_af_linjer:
                    continue
                set_af_linjer.add(noegle)

                traef: list[float | None] = []
                for y, kand in zip(ys, stoettende):
                    forudsagt = skaering + haeldning * y / hoejde
                    naermeste = min(kand, key=lambda k: abs(k - forudsagt), default=None)
                    traef.append(
                        naermeste if naermeste is not None
                        and abs(naermeste - forudsagt) <= tolerance else None
                    )
                stoette = sum(1 for t in traef if t is not None)
                if stoette < min_stoette:
                    continue
                inderhed = udad * (sum(t for t in traef if t is not None) / stoette)
                kandidat_linjer.append((inderhed, stoette, skaering, haeldning, 0.0))

    if not kandidat_linjer:
        return [None] * len(baand)

    # Kandidaterne samles i KANTER: linjer taettere end `SAMME_KANT` paa
    # hinanden beskriver den samme fysiske kant. Kun den bedst stoettede
    # linje pr. kant gaar videre -- ellers skulle ~1.000 linjer soem-maales.
    kandidat_linjer.sort(key=lambda l: l[0])
    kanter: list[list[tuple[float, int, float, float, float]]] = []
    for l in kandidat_linjer:
        if kanter and l[0] - kanter[-1][0][0] <= SAMME_KANT:
            kanter[-1].append(l)
        else:
            kanter.append([l])

    # Hver kant skal BEVISE sin soem. Det er kravet, der skiller sidens
    # rigtige kant fra en svag, ret skygge inde paa papiret -- og uden det
    # vandt skyggen paa 273105_001569 og 273103_001437, blot fordi den laa
    # inderst. Det er en fysisk egenskab, ikke en taerskel valgt paa faelt.
    if graa is not None:
        bekraeftede = []
        for gruppe in kanter:
            _, stoette, skaering, haeldning, y_anker = max(gruppe, key=lambda l: l[1])
            if soem_dybde(graa, skaering, haeldning) >= SOEM_GULV:
                bekraeftede.append(gruppe)
        if not bekraeftede:
            return [None] * len(baand)   # hellere afstaa end skaere paa et gaet
        kanter = bekraeftede

    # FOERST inderste kant -- for et blad ligger altid uden for vores side.
    # DEREFTER den bedst stoettede linje langs netop den kant.
    bedste = max(kanter[0], key=lambda l: l[1])

    _, _, skaering, haeldning, _ = bedste
    valgt: list[int | None] = []
    for y, kand in zip(ys, stoettende):
        forudsagt = skaering + haeldning * y / hoejde
        naermeste = min(kand, key=lambda k: abs(k - forudsagt), default=None)
        valgt.append(int(naermeste) if naermeste is not None
                     and abs(naermeste - forudsagt) <= tolerance else None)
    return valgt


def _baand(hoejde: int, *, antal: int = ANTAL_BAAND, overlap: float = OVERLAP):
    """Baandgeometrien fra `skraa.baandkanter` -- samme vinduer, samme sted."""
    y0, y1 = int(hoejde * MARGEN), int(hoejde * (1 - MARGEN))
    skridt = max(1, (y1 - y0) // antal)
    halv = max(1, int(skridt * overlap / 2))
    for i in range(antal):
        midte = y0 + skridt * i + skridt // 2
        yield midte, max(y0, midte - halv), min(y1, midte + halv)


def baandkanter_ydre(img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
                     min_fald: float = MIN_FALD) -> list[tuple[int, int | None]]:
    """Yderkantens x maalt i et glidende vindue ned gennem siden.

    Returnerer `(vinduets midte i y, kantens x)`. `None` betyder, at
    vinduet ikke rummede en troevaerdig kant.
    """
    vindue = ydre_vindue(side, img.width)   # rejser ValueError ved ukendt r/v
    graa = np.asarray(img.convert("L"), dtype=float)
    baand: list[tuple[int, list[int]]] = []
    for midte, ya, yb in _baand(img.height, antal=antal):
        if yb - ya < 2:
            baand.append((midte, []))
            continue
        baand.append((midte, _kandidater_i_profil(papir_profil(graa, ya, yb),
                                                  vindue, min_fald=min_fald)))

    # Baandene ses under ét: kun de kandidater, der ligger paa den samme
    # rette kant, faar lov at bestemme snittet.
    valgt = _bedste_linje(baand, img.height, vindue, graa=graa)
    return [(y, x) for (y, _), x in zip(baand, valgt)]


def ydre_graense(img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
                 buffer_andel: float = BUFFER_ANDEL,
                 kanter: list[tuple[int, int | None]] | None = None) -> list[int]:
    """Yderkantens x for HVER raekke, interpoleret mellem baandene.

    Bufferen flyttes UDAD -- vaek fra vores egen tekst -- saa et snit lige
    paa graensen ikke kan barbere enderne af hoeje eller dybe bogstavtraek.
    """
    vindue = ydre_vindue(side, img.width)
    raa = baandkanter_ydre(img, side, antal=antal) if kanter is None else kanter
    fundne = [(y, x) for y, x in raa if x is not None]
    if not fundne:
        return []

    buffer_px = max(1, int(img.width * buffer_andel))
    udad = 1 if vindue.retning == "fra_hoejre" else -1
    ys = np.array([y for y, _ in fundne], dtype=float)
    xs = np.array([x for _, x in fundne], dtype=float)
    x = np.interp(np.arange(img.height, dtype=float), ys, xs)
    return (np.rint(x).astype(int) + udad * buffer_px).tolist()


def beskaer_ydre(img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
                 buffer_andel: float = BUFFER_ANDEL,
                 min_fald: float = MIN_FALD) -> tuple[Image.Image, YdreBeskaering]:
    """Beskaerer langs yderkanten og returnerer `(billede, maaling)`.

    Findes ingen kant, returneres billedet uaendret og maalingen mærkes
    usikker -- stage-kontrakten kraever, at usikre tilfaelde mærkes frem for
    at blive skaaret paa slump.
    """
    vindue = ydre_vindue(side, img.width)
    kanter = baandkanter_ydre(img, side, antal=antal, min_fald=min_fald)
    med_kant = sum(1 for _, x in kanter if x is not None)
    graense = ydre_graense(img, side, antal=antal, buffer_andel=buffer_andel, kanter=kanter)

    if not graense:
        return img.copy(), YdreBeskaering(
            billede=side.image_name, recto_verso=side.recto_verso,
            bredde_foer=img.width, bredde_efter=img.width,
            haeldning_px=0, baand_med_kant=0, baand_i_alt=len(kanter),
        )

    # Billedets egen farvetilstand bevares -- en graa udgave ville vaere en
    # skjult aendring af netop det, forsoeget skal isolere.
    data = np.asarray(img).copy()
    kol = np.arange(img.width)[None, :]
    g = np.asarray(graense)[:, None]
    udenfor = kol >= g if vindue.retning == "fra_hoejre" else kol <= g
    if data.ndim == 3:
        udenfor = udenfor[:, :, None]
    data = np.where(udenfor, 255, data).astype(np.uint8)
    arbejde = Image.fromarray(data, mode=img.mode)

    if vindue.retning == "fra_hoejre":
        beskaaret = arbejde.crop((0, 0, min(img.width, max(graense) + 1), img.height))
    else:
        beskaaret = arbejde.crop((max(0, min(graense)), 0, img.width, img.height))

    fundne = [x for _, x in kanter if x is not None]
    return beskaaret, YdreBeskaering(
        billede=side.image_name, recto_verso=side.recto_verso,
        bredde_foer=img.width, bredde_efter=beskaaret.width,
        haeldning_px=max(fundne) - min(fundne),
        baand_med_kant=med_kant, baand_i_alt=len(kanter),
    )


def har_fremmed_blad(img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
                     min_bredde: int = BLAD_MIN_BREDDE,
                     min_niveau: float = BLAD_MIN_NIVEAU) -> BladFund:
    """Ligger der et fremmed blad uden for kanten? (forsoeg B)

    Afgøres paa BREDDEN af det lyse baelte uden for kanten. Ikke paa, om
    der overhovedet kommer papir igen: sveepet 2026-08-28 viste, at ogsaa
    et rent bogsnit giver nogle faa lyse pixels lige uden for kanten, saa
    et "kommer papiret igen"-krav ville sige ja til alt.

    Recall vejer tungere end precision: et overset blad sender fremmed
    tekst videre til modellen, mens et falsk ja kun koster et snit, der
    alligevel skal bestaa kantdetektionens egne krav.
    """
    vindue = ydre_vindue(side, img.width)
    graa = np.asarray(img.convert("L"), dtype=float)
    udad = 1 if vindue.retning == "fra_hoejre" else -1

    bredder, minima, efter = [], [], []
    for _, ya, yb in _baand(img.height, antal=antal):
        if yb - ya < 2:
            continue
        profil = papir_profil(graa, ya, yb)
        kant = _kant_i_profil(profil, vindue)
        if kant is None:
            continue

        # Vores eget papirs niveau, maalt lige INDEN for kanten -- ikke i
        # kanten selv, hvor udglatningen allerede har traukket den ned.
        inden_for = profil[max(0, kant - 30) : kant + 1] if udad > 0 else profil[kant : kant + 31]
        eget = max(inden_for) if inden_for else profil[kant]
        lyst = eget * min_niveau
        moerkt = eget * 0.40   # herunder er det bogsnit eller baggrund

        # Gaa udad, til baggrunden naas. Det lyse baelte undervejs er det,
        # der skiller et fremmed blad fra et bogsnit: bladet giver ~190 px
        # papir, bogsnittet kun en snip paa ~20.
        strimmel: list[float] = []
        x = kant
        while 0 <= x + udad < len(profil):
            x += udad
            if profil[x] < moerkt:
                break
            strimmel.append(profil[x])

        bredder.append(sum(1 for v in strimmel if v >= lyst))
        minima.append(min(strimmel) if strimmel else profil[kant])
        efter.append(max(strimmel) if strimmel else profil[kant])

    if not bredder:
        return BladFund(False, 0, 0.0, 0.0, 0, antal)

    med_blad = sum(1 for b in bredder if b >= min_bredde)
    return BladFund(
        er_blad=med_blad * 2 >= len(bredder),
        baelte_bredde=int(np.median(bredder)),
        niveau_min=float(np.median(minima)),
        niveau_efter=float(np.median(efter)),
        baand_med_blad=med_blad,
        baand_i_alt=len(bredder),
    )
