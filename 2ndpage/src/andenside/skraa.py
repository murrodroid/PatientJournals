"""Baandvis beskaering: snittet foelger falsen i stedet for at vaere lodret.

Baggrund, maalt 2026-08-27. Siden krummer ind mod falsen, og skriveren skrev
helt ud. Et LODRET snit maa derfor vaelge mellem to fejl: skaere tekst af i
den ende af siden, hvor skriften naar laengst ud, eller tage naboopslaget med
i den anden. Paa 273104_001639 er de nederste linjer skaaret over, mens de
oeverste er rene -- og et snit langt nok ude til at redde de nederste
traekker naboens side med foroven.

Et baandvist snit har ikke det dilemma: det kan ligge taet, hvor teksten
stopper tidligt, og laengere ude, hvor den loeber langt ud.

**Snittet foelger FALSEN, ikke teksten.** Falsen er en fysisk graense og
ligger stille. Tekstens yderkant hopper fra linje til linje, og et snit, der
fulgte den, ville blive uroligt og afhaenge af blaekdetektion -- to
usikkerheder oveni hinanden. Falsen findes med samme spring-logik som
`bogryg.find_snitpunkt`, bare inden for hvert baand for sig.

Resultatet er stadig et almindeligt rektangulaert billede. Det, der ligger paa
den forkerte side af graensen, males hvidt, og billedet trimmes til det
yderste, der er beholdt. Modellen ser altsaa en helt normal side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from andenside.bogryg import MIN_STIGNING, smooth, soegevindue
from andenside.masterlist import Side

ANTAL_BAAND = 24       # antal maalepunkter ned gennem siden
OVERLAP = 2.5          # hvert vindues hoejde, malt i skridt mellem punkterne
MARGEN = 0.03          # top og bund udelades -- affotograferingens skygge
LINJE_TOLERANCE = 25   # px et baand maa afvige fra flertallets rette linje
# 25, ikke mindre: maalt paa leveringens 307 sider afviger falsen 5 px fra
# en ret linje i median og 11 px ved 90-percentilen. En tolerance paa 25
# rummer altsaa al virkelig krumning med god margen. De sider, lead kaldte
# gaaet galt, afveg 245-412 px -- ti gange for meget til at vaere en fals.
MIND_ENIGE = 8         # saa mange baand skal blive tilbage, ellers er siden usikker
BUFFER_ANDEL = 0.005   # flyttes VAEK fra vores egen tekst, ind mod falsen
# 0,5 %, valgt af lead 2026-08-30 efter at have set samme sider ved 2,0 %,
# 1,0 % og 0,5 % med det bortskaarne tonet
# (`stages/04_billedforberedelse/output/buffer_sammenligning/`).
#
# Vaerdien har vaeret 1 %, saa 2 %, og nu 0,5 %. Historien er vaerd at kende:
# den blev sat OP til 2 % den 2026-08-27, fordi 1 % ikke reddede de nederste
# linjer paa 273104_001639, hvor siden krummer ind mod falsen og skriveren
# skrev helt ud. Den bekymring blev afloest af, at baandene siden er blevet
# renset for udskridende maalinger (`fjern_udskridende`): snitlinjen ligger
# nu dér, hvor falsen faktisk er, og ikke trukket skaev af en enkelt gal
# maaling, saa der er ikke laengere brug for en stor buffer til at daekke
# over uroen. Lead saa 273104_001639 ved 0,5 % og godtog den.
#
# Retningen er ikke symmetrisk: en stor buffer lader naboens strimmel staa
# (stoej, prompten kan bede modellen se bort fra), en for lille barberer
# vores egne ordender (en fejl i transskriptionen). Gaa derfor kun lavere
# efter et nyt visuelt gennemsyn.


@dataclass(frozen=True)
class SkraaBeskaering:
    """Hvad der skete -- og hvor skaev falsen viste sig at vaere."""

    billede: str
    recto_verso: str
    bredde_foer: int
    bredde_efter: int
    haeldning_px: int      # forskel mellem yderste og inderste baandkant
    baand_med_kant: int
    baand_i_alt: int

    afvigelse_px: int = 0   # stoerste afstand fra et baand til den rette fals

    @property
    def sikker(self) -> bool:
        # To krav, ikke ét. Det gamle taalte kun, om baandene fandt NOGET,
        # og sagde derfor ja til alle 9 sider, hvor snittet paa leveringen
        # skar tvaers gennem teksten -- dér fandt alle 24 baand en kant, de
        # var bare ikke enige om hvor. Baandene skal ogsaa ligge paa samme
        # rette linje.
        return (self.baand_med_kant * 2 >= self.baand_i_alt
                and self.baand_med_kant >= MIND_ENIGE
                and self.afvigelse_px <= LINJE_TOLERANCE)


def _graatoner(img: Image.Image) -> "np.ndarray":
    """Billedet som graatone-array.

    Konverteringen koster ~13 ms pr. kald, saa den laves ÉN gang pr. side og
    sendes videre -- ikke én gang pr. baand.
    """
    return np.asarray(img.convert("L"))


def _profil_i_baand(graa: "np.ndarray", y0: int, y1: int, *, step: int = 3,
                    taerskel: int = 180) -> list[float]:
    return smooth((graa[y0:y1:step, :] < taerskel).mean(axis=0).tolist())


def _kant_i_profil(profil: list[float], vindue) -> int | None:
    """Falsens naere kant: stoerste spring opad langs soegeretningen."""
    if vindue.retning == "fra_hoejre":
        raek = range(vindue.start + 1, vindue.slut)
        forrige = -1
    else:
        raek = range(vindue.slut - 2, vindue.start - 1, -1)
        forrige = 1
    bedste_x, bedste = None, 0.0
    for x in raek:
        spring = profil[x] - profil[x + forrige]
        if spring > bedste:
            bedste, bedste_x = spring, x
    return bedste_x if bedste >= MIN_STIGNING else None


def baandkanter(
    img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
    overlap: float = OVERLAP,
) -> list[tuple[int, int | None]]:
    """Falsens kant maalt i et GLIDENDE vindue ned gennem siden.

    Returnerer `(vinduets midte i y, kantens x)`. `None` betyder, at vinduet
    ikke rummede en troevaerdig fals -- fx fordi der ikke er skrevet noget i
    den hoejde.

    Vinduerne OVERLAPPER (hvert er `overlap` gange saa hoejt som skridtet
    mellem dem). To ting vindes ved det: hver enkelt maaling bygger paa flere
    raekker og bliver derfor mindre foelsom over for et stykke uden blaek, og
    graensen bliver glat i stedet for at faa et knaek ved hver baandgraense.
    Med adskilte baand kunne snitkanten ses "trappe" paa sider med skaev fals.
    """
    vindue = soegevindue(side, img.width)   # rejser ValueError ved ukendt r/v
    graa = _graatoner(img)
    y0, y1 = int(img.height * MARGEN), int(img.height * (1 - MARGEN))
    skridt = max(1, (y1 - y0) // antal)
    halv = max(1, int(skridt * overlap / 2))

    ud = []
    for i in range(antal):
        midte = y0 + skridt * i + skridt // 2
        ya = max(y0, midte - halv)
        yb = min(y1, midte + halv)
        if yb - ya < 2:
            ud.append((midte, None))
            continue
        ud.append((midte, _kant_i_profil(_profil_i_baand(graa, ya, yb), vindue)))
    return ud


def fjern_udskridende(
    kanter: list[tuple[int, int | None]], *, tolerance: int = LINJE_TOLERANCE,
    mindst: int = MIND_ENIGE,
) -> tuple[list[tuple[int, int | None]], int]:
    """Kaster de baand, der ikke ligger paa flertallets rette linje.

    Falsen er maalt til at afvige hoejst 11 px fra en ret linje paa 90 % af
    leveringens 307 sider. Et baand, der peger 400 px vaek, har altsaa ikke
    fundet falsen, men noget andet moerkt -- en klat, en skygge, en
    fingerkant. Interpolationen i `fals_graense` spaender frit mellem
    baandene, saa ét saadant baand traekker snittet med sig og skaerer
    tvaers gennem siden. Det skete paa 9 af leveringens 307 sider.

    Den vaerste kastes ad gangen og linjen laegges paa ny, indtil alle
    tilbagevaerende ligger inden for `tolerance`. Ét gennemloeb ville ikke
    raekke: en enkelt vild maaling traekker selve linjen skaev, saa flere
    rigtige baand ser ud til at afvige.

    Returnerer `(kanter, stoerste afvigelse tilbage)`.
    """
    fundne = [(y, x) for y, x in kanter if x is not None]
    if len(fundne) < 3:
        return kanter, 0

    beholdt = dict(fundne)
    while len(beholdt) > mindst:
        ys = np.array(list(beholdt), dtype=float)
        xs = np.array(list(beholdt.values()), dtype=float)
        haeldning, skaering = np.polyfit(ys, xs, 1)
        afvig = np.abs(xs - (haeldning * ys + skaering))
        if afvig.max() <= tolerance:
            break
        del beholdt[int(ys[int(afvig.argmax())])]

    ys = np.array(list(beholdt), dtype=float)
    xs = np.array(list(beholdt.values()), dtype=float)
    if len(beholdt) >= 3:
        haeldning, skaering = np.polyfit(ys, xs, 1)
        rest = int(round(float(np.abs(xs - (haeldning * ys + skaering)).max())))
    else:
        rest = 0
    return [(y, x if y in beholdt else None) for y, x in kanter], rest


def fals_graense(
    img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
    buffer_andel: float = BUFFER_ANDEL,
    kanter: list[tuple[int, int | None]] | None = None,
) -> list[int]:
    """Falsens x for HVER raekke i billedet, interpoleret mellem baandene.

    Bufferen flyttes vaek fra vores egen tekst, praecis som i `bogryg`.
    """
    vindue = soegevindue(side, img.width)
    # Kun baand med en troevaerdig fals bruges. Huller behoever ingen
    # saerbehandling: interpolationen nedenfor spaender hen over dem, og
    # raekker uden for yderste kendte baand faar dets vaerdi.
    raa = baandkanter(img, side, antal=antal) if kanter is None else kanter
    raa, _ = fjern_udskridende(raa)
    kanter = [(y, x) for y, x in raa if x is not None]
    if not kanter:
        return []

    buffer_px = max(1, int(img.width * buffer_andel))
    retning = 1 if vindue.retning == "fra_hoejre" else -1

    # Lineaer interpolation mellem baandenes midter. `np.interp` holder
    # yderste kendte vaerdi uden for spaendet, praecis som en raekke uden et
    # baand paa begge sider skal have.
    ys = np.array([y for y, _ in kanter], dtype=float)
    xs = np.array([x for _, x in kanter], dtype=float)
    alle = np.arange(img.height, dtype=float)
    x = np.interp(alle, ys, xs)
    return (np.rint(x).astype(int) + retning * buffer_px).tolist()


def beskaer_langs_fals(
    img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
    buffer_andel: float = BUFFER_ANDEL,
) -> tuple[Image.Image, SkraaBeskaering]:
    """Beskaerer langs falsen og returnerer `(billede, maaling)`.

    Alt paa den forkerte side af graensen males hvidt, og billedet trimmes
    til det yderste, der er beholdt. Resultatet er et almindeligt rektangel.
    """
    vindue = soegevindue(side, img.width)
    # Baandkanterne beregnes ÉN gang og genbruges. Tidligere kaldte denne
    # funktion baade baandkanter og fals_graense, og sidstnaevnte beregnede
    # dem forfra -- altsaa tre gange det samme arbejde pr. side.
    kanter = baandkanter(img, side, antal=antal)
    kanter, afvigelse = fjern_udskridende(kanter)
    med_kant = sum(1 for _, x in kanter if x is not None)
    graense = fals_graense(img, side, antal=antal, buffer_andel=buffer_andel,
                           kanter=kanter)

    if not graense:
        maaling = SkraaBeskaering(
            billede=side.image_name, recto_verso=side.recto_verso,
            bredde_foer=img.width, bredde_efter=img.width,
            haeldning_px=0, baand_med_kant=0, baand_i_alt=len(kanter),
            afvigelse_px=afvigelse,
        )
        return img.copy(), maaling

    # Billedets egen farvetilstand bevares -- en graa udgave ville vaere en
    # skjult aendring af det, modellen faar at se.
    #
    # Udhvidningen laves som én maske i stedet for pixel for pixel: for hver
    # raekke sammenlignes kolonneindekset med raekkens egen graense.
    data = np.asarray(img).copy()
    kol = np.arange(img.width)[None, :]
    g = np.asarray(graense)[:, None]
    udenfor = kol >= g if vindue.retning == "fra_hoejre" else kol < g
    if data.ndim == 3:
        udenfor = udenfor[:, :, None]
    data = np.where(udenfor, 255, data).astype(np.uint8)
    arbejde = Image.fromarray(data, mode=img.mode)

    if vindue.retning == "fra_hoejre":
        yderste = min(img.width, max(graense))
        beskaaret = arbejde.crop((0, 0, yderste, img.height))
    else:
        yderste = max(0, min(graense))
        beskaaret = arbejde.crop((yderste, 0, img.width, img.height))

    fundne = [x for _, x in kanter if x is not None]
    maaling = SkraaBeskaering(
        billede=side.image_name,
        recto_verso=side.recto_verso,
        bredde_foer=img.width,
        bredde_efter=beskaaret.width,
        haeldning_px=(max(fundne) - min(fundne)) if fundne else 0,
        baand_med_kant=med_kant,
        baand_i_alt=len(kanter),
    )
    return beskaaret, maaling
