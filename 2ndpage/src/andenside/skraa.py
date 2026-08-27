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

from PIL import Image

from andenside.bogryg import MIN_STIGNING, smooth, soegevindue
from andenside.masterlist import Side

ANTAL_BAAND = 10
MARGEN = 0.03          # top og bund udelades -- affotograferingens skygge
BUFFER_ANDEL = 0.02    # flyttes vaek fra vores egen tekst
# 2%, ikke 1%: maalt paa 273104_001639, hvor skriften loeber ud i papirets
# krumning, redder 1% ikke de nederste linjer. Den baandvise graense kan kun
# koebe forskellen mellem baandene (dér 25 px) -- resten skal bufferen give.


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

    @property
    def sikker(self) -> bool:
        # Under halvdelen af baandene fandt en fals -> graensen er gaettet
        # ud fra for lidt, og siden skal ses efter med oejnene.
        return self.baand_med_kant * 2 >= self.baand_i_alt


def _profil_i_baand(img: Image.Image, y0: int, y1: int, *, step: int = 3,
                    taerskel: int = 180) -> list[float]:
    graa = img.convert("L")
    px = graa.load()
    bredde = graa.size[0]
    ud = []
    for x in range(bredde):
        moerke = total = 0
        for y in range(y0, y1, step):
            total += 1
            if px[x, y] < taerskel:
                moerke += 1
        ud.append(moerke / total)
    return smooth(ud)


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
    img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND
) -> list[tuple[int, int | None]]:
    """Falsens kant i hvert vandret baand: `(baandets midte i y, kantens x)`.

    `None` betyder, at baandet ikke rummede en troevaerdig fals -- fx fordi
    der ikke er skrevet noget i den hoejde.
    """
    vindue = soegevindue(side, img.width)   # rejser ValueError ved ukendt r/v
    y0, y1 = int(img.height * MARGEN), int(img.height * (1 - MARGEN))
    hoejde = max(1, (y1 - y0) // antal)

    ud = []
    for i in range(antal):
        ya = y0 + i * hoejde
        yb = y1 if i == antal - 1 else ya + hoejde
        kant = _kant_i_profil(_profil_i_baand(img, ya, yb), vindue)
        ud.append(((ya + yb) // 2, kant))
    return ud


def fals_graense(
    img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
    buffer_andel: float = BUFFER_ANDEL,
) -> list[int]:
    """Falsens x for HVER raekke i billedet, interpoleret mellem baandene.

    Bufferen flyttes vaek fra vores egen tekst, praecis som i `bogryg`.
    """
    vindue = soegevindue(side, img.width)
    # Kun baand med en troevaerdig fals bruges. Huller behoever ingen
    # saerbehandling: interpolationen nedenfor spaender hen over dem, og
    # raekker uden for yderste kendte baand faar dets vaerdi.
    kanter = [(y, x) for y, x in baandkanter(img, side, antal=antal) if x is not None]
    if not kanter:
        return []

    buffer_px = max(1, int(img.width * buffer_andel))
    retning = 1 if vindue.retning == "fra_hoejre" else -1

    graense = []
    for y in range(img.height):
        foer = [k for k in kanter if k[0] <= y]
        efter = [k for k in kanter if k[0] > y]
        if foer and efter:
            (y0, x0), (y1, x1) = foer[-1], efter[0]
            x = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
        else:
            x = (foer or efter)[-1 if foer else 0][1]
        graense.append(int(round(x)) + retning * buffer_px)
    return graense


def beskaer_langs_fals(
    img: Image.Image, side: Side, *, antal: int = ANTAL_BAAND,
    buffer_andel: float = BUFFER_ANDEL,
) -> tuple[Image.Image, SkraaBeskaering]:
    """Beskaerer langs falsen og returnerer `(billede, maaling)`.

    Alt paa den forkerte side af graensen males hvidt, og billedet trimmes
    til det yderste, der er beholdt. Resultatet er et almindeligt rektangel.
    """
    vindue = soegevindue(side, img.width)
    kanter = baandkanter(img, side, antal=antal)
    med_kant = sum(1 for _, x in kanter if x is not None)
    graense = fals_graense(img, side, antal=antal, buffer_andel=buffer_andel)

    if not graense:
        maaling = SkraaBeskaering(
            billede=side.image_name, recto_verso=side.recto_verso,
            bredde_foer=img.width, bredde_efter=img.width,
            haeldning_px=0, baand_med_kant=0, baand_i_alt=len(kanter),
        )
        return img.copy(), maaling

    # Billedets egen farvetilstand bevares -- en graa udgave ville vaere en
    # skjult aendring af det, modellen faar at se.
    arbejde = img.copy()
    tegn = arbejde.load()
    hvid = 255 if arbejde.mode == "L" else (255,) * len(arbejde.getbands())
    if vindue.retning == "fra_hoejre":
        for y in range(img.height):
            for x in range(max(0, graense[y]), img.width):
                tegn[x, y] = hvid
        yderste = min(img.width, max(graense))
        beskaaret = arbejde.crop((0, 0, yderste, img.height))
    else:
        for y in range(img.height):
            for x in range(0, min(img.width, graense[y])):
                tegn[x, y] = hvid
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
