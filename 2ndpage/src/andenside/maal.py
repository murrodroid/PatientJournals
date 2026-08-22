"""Maaleapparatet: forankring af facit i modellens tekst, og tallene der falder ud.

Hele stagen hviler paa ÉN handling -- find facits kendte tekst i modellens
tekst -- og laeser tre ting af resultatet (rod-CONTEXT.md 2026-08-21):

    forankr(facit_linje, modeltekst) -> stumper, gab

- De fundne **stumper** er kendt sandhed og maales (tegn- og ordfejl).
- **Gabet** mellem to fundne stumper er det, modellen skrev, hvor facit siger
  `[?]`. Dets laengde er hallucinations-signalet.
- Samme gab er modellens bud paa stedet, hvis det senere skal forelaegges.

Grundreglen er stadig beslutning 38: kan en linje ikke forankres, gaar HELE
linjen ud af maalingen. Forankringen er en forbedring oven paa den, med en
defineret vej tilbage -- ikke en anden maade at maale paa.

## Valg, der ikke stod i kontrakten, men foelger af den

De to vigtigste staar her; de oevrige tre (orddelingen afgjort paa facits
linjer, fuldside-kontrollen, og at linjetrofasthed maales frem for antages)
er dokumenteret ved den kode, der baerer dem. Alle fem er beslutning 39-43 i
rod-CONTEXT.md.

**Soegningen taaler laesefejl inde i stumpen.** Kraevede den ordret traef,
ville hver forankret stump per definition have nul fejl, og tallet ville
maale, hvor tit modellen var perfekt -- ikke hvor god den er. Derfor et
naermeste-udsnit-soeg med en oevre graense for afvigelsen (`MAKS_AFVIGELSE`).
Graensen er en knap: saettes den lavere, falder daekningen, og de tilbagevaerende
linjer er de letteste -- altsaa et pænere og mere misvisende tal. Derfor er
den sat rundhaandet, og `selvtest.md` viser, hvordan tallene flytter sig med den.

**Linjeparringen ER forankringen.** Kontraktens punkt 3 kraever, at linjerne
parres, foer de sammenlignes, saa maalingen ikke skrider efter det foerste
afvigende linjebrud. Det sker gratis her: hver facit-linje soeges i modellens
raa tekst uden hensyn til dens linjeskift. Om modellen saa foelger sidens
linjer eller laver sine egne -- den ubeviste antagelse i beslutning 35 --
bliver et resultat, vi maaler (`uden_linjeskift_indeni`, `egen_modellinje`),
i stedet for noget maalingen afhaenger af.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field
from functools import lru_cache

from andenside import cer
from andenside.facit import saml_orddeling

MAERKE = "[?]"

# Stumper under fem tegn bruges ikke til forankring (122 af de 647 i facit).
# De kan forankre hvor som helst og ville give falsk tryghed.
MINDSTE_STUMP = 5

# Hvor meget en stump maa afvige og stadig regnes for fundet, som andel af
# stumpens laengde. 0,4 er rundhaandet med vilje: den, der saenker den, goer
# tallet pænere ved at smide de svaereste linjer ud.
MAKS_AFVIGELSE = 0.4

# Foerste soegning sker i et vindue lige efter forrige stump. Rammer den ikke,
# soeges resten af siden igennem. Vinduet er ren fart -- ikke en regel om, hvor
# teksten maa staa: rammer vinduet ved siden af, finder trin 3 traeffet
# alligevel. Det koster kun tid, aldrig et fund.
VINDUE_EKSTRA = 80


# --------------------------------------------------------------------------
# Soegeform: en foldet udgave af teksten med vej tilbage til de raa positioner
# --------------------------------------------------------------------------

@lru_cache(maxsize=100_000)
def _fold(tegn: str) -> str:
    """Ét tegn foldet. Cachet, fordi `strip_diacritics` kalder
    `unicodedata.normalize` to gange pr. tegn -- og hele siden foldes én gang
    pr. facit-linje, saa de faa hundrede forskellige tegn i materialet ellers
    bliver slaaet op millioner af gange."""
    return cer.strip_diacritics(tegn.lower())


@lru_cache(maxsize=512)
def _soegeform(tekst: str) -> tuple[str, tuple[int, ...]]:
    """Folder versaler, tyske omlyde og mellemrum, og husker hvor hvert tegn
    kom fra i den raa tekst.

    Cachet paa teksten: sidens modeltekst er den samme for alle sidens linjer,
    og uden cachen foldes den forfra hver gang.

    Tegnsaetning bevares her, selvom `lempeligst`-varianten kaster den vaek.
    Grunden er positionerne: droppes kommaet i ", og Canylen", begynder det
    fundne udsnit foerst ved o'et, og kommaet havner i gabet ved siden af --
    hvor det ville blive laest som noget, modellen fandt paa. Soegningen taaler
    alligevel forskelle, saa den har ikke brug for filteret.
    """
    ud: list[str] = []
    kort: list[int] = []
    sidst_var_mellemrum = True
    for i, tegn in enumerate(tekst):
        if tegn.isspace():
            if not sidst_var_mellemrum:
                ud.append(" ")
                kort.append(i)
                sidst_var_mellemrum = True
            continue
        for c in _fold(tegn):
            ud.append(c)
            kort.append(i)
        sidst_var_mellemrum = False
    while ud and ud[-1] == " ":
        ud.pop()
        kort.pop()
    return "".join(ud), tuple(kort)


def _naermeste_udsnit(naal: str, hoestak: str) -> tuple[int, int, int]:
    """Det udsnit af `hoestak`, der ligger taettest paa `naal`.

    Returnerer `(start, slut, afstand)`. Levenshtein med fri begyndelse og
    slutning i hoestakken: raekke 0 er nul hele vejen, saa et traef maa
    begynde hvor som helst uden at betale for teksten foran.

    Ved lige billige veje vaelges diagonal foer sletning foer indsaettelse, og
    det tidligste slutpunkt vinder. Uden den faste raekkefoelge kunne to
    koersler af uaendret kode give hver sit resultat.
    """
    n, m = len(naal), len(hoestak)
    if n == 0:
        return 0, 0, 0
    if m == 0:
        return 0, 0, n

    forrige = [0] * (m + 1)
    forrige_start = list(range(m + 1))
    for i in range(1, n + 1):
        nu = [i] + [0] * m
        nu_start = [0] * (m + 1)
        for j in range(1, m + 1):
            diagonal = forrige[j - 1] + (naal[i - 1] != hoestak[j - 1])
            slet = forrige[j] + 1
            indsaet = nu[j - 1] + 1
            bedst = min(diagonal, slet, indsaet)
            nu[j] = bedst
            if bedst == diagonal:
                nu_start[j] = forrige_start[j - 1]
            elif bedst == slet:
                nu_start[j] = forrige_start[j]
            else:
                nu_start[j] = nu_start[j - 1]
        forrige, forrige_start = nu, nu_start

    slut = min(range(m + 1), key=lambda j: (forrige[j], j))
    return forrige_start[slut], slut, forrige[slut]


def _find_stump(naal_raa: str, hoestak_raa: str, fra: int, maks_afvigelse: float):
    """Finder `naal_raa` i `hoestak_raa[fra:]`. Returnerer `(start, slut)` i
    raa koordinater, eller `None` hvis afvigelsen er for stor.
    """
    naal, _ = _soegeform(naal_raa)
    if not naal:
        return None
    hoestak, kort = _soegeform(hoestak_raa)
    if not hoestak:
        return None

    # Hvor i soegeformen svarer `fra` til? Foerste tegn med raa position >= fra.
    # `kort` vokser monotont, saa der kan soeges binaert.
    start_i_soegeform = bisect_left(kort, fra)
    rest = hoestak[start_i_soegeform:]
    if not rest:
        return None

    graense = maks_afvigelse * len(naal)

    def til_raa(a: int, b: int) -> tuple[int, int]:
        a += start_i_soegeform
        b += start_i_soegeform
        raa_start = kort[a]
        raa_slut = kort[b - 1] + 1
        # Udsnittet kan begynde eller slutte midt i mellemrum, fordi
        # soegeformen har slaaet dem sammen. Skær dem af, saa det maalte
        # stykke er den tekst, der faktisk staar der.
        while raa_start < raa_slut and hoestak_raa[raa_start].isspace():
            raa_start += 1
        while raa_slut > raa_start and hoestak_raa[raa_slut - 1].isspace():
            raa_slut -= 1
        return raa_start, raa_slut

    # 1) Ordret traef er baade det hyppigste og det billigste.
    truffet = rest.find(naal)
    if truffet != -1:
        return til_raa(truffet, truffet + len(naal))

    # 2) Naermeste udsnit i et vindue lige efter forrige stump.
    vindue = rest[: len(naal) + VINDUE_EKSTRA]
    a, b, afstand = _naermeste_udsnit(naal, vindue)
    if afstand <= graense and b > a:
        return til_raa(a, b)

    # 3) Resten af siden, hvis vinduet ikke raktes. Betales kun ved uheld.
    if len(vindue) < len(rest):
        a, b, afstand = _naermeste_udsnit(naal, rest)
        if afstand <= graense and b > a:
            return til_raa(a, b)
    return None


# --------------------------------------------------------------------------
# Forankring af én facit-linje
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Stump:
    """Et stykke kendt facit-tekst mellem to ulaeselighedsmaerker."""

    tekst: str
    start: int          # position i facit-linjen
    slut: int
    for_kort: bool      # under MINDSTE_STUMP -- bruges aldrig til forankring
    fundet: bool = False
    model_tekst: str = ""
    model_start: int = -1
    model_slut: int = -1


@dataclass(frozen=True)
class Gab:
    """Det modellen skrev, hvor facit siger `[?]`.

    `facit_mellem` er facits egen tekst mellem de to fundne stumper -- typisk
    `[?]`, men kan rumme en stump, der var for kort til at forankre. Den staar
    med, saa gabet ikke laeses som rent opdigt, naar det ikke er det.
    """

    facit_mellem: str
    model_tekst: str
    model_start: int
    model_slut: int


@dataclass(frozen=True)
class LinjeFund:
    nr: int
    facit: str
    stumper: tuple[Stump, ...] = ()
    gab: tuple[Gab, ...] = ()

    @property
    def forankret(self) -> bool:
        return any(s.fundet for s in self.stumper)

    @property
    def svaer(self) -> bool:
        return MAERKE in self.facit

    @property
    def fundne(self) -> tuple[Stump, ...]:
        return tuple(s for s in self.stumper if s.fundet)

    @property
    def facit_maalt(self) -> str:
        """Facits tekst paa denne linje, som den indgaar i maalingen."""
        return " ".join(s.tekst for s in self.fundne)

    @property
    def model_maalt(self) -> str:
        return " ".join(s.model_tekst for s in self.fundne)

    @property
    def model_udsnit(self) -> tuple[int, int]:
        """Linjens samlede udstraekning i modelteksten, gab iberegnet."""
        fundne = self.fundne
        if not fundne:
            return (-1, -1)
        return (fundne[0].model_start, fundne[-1].model_slut)


def _del_i_stumper(facit_linje: str) -> list[Stump]:
    """Deler linjen ved ulaeselighedsmaerkerne. Tomme stykker falder ud."""
    stumper: list[Stump] = []
    pos = 0
    graenser = [(m.start(), m.end()) for m in re.finditer(re.escape(MAERKE), facit_linje)]
    for start, slut in graenser + [(len(facit_linje), len(facit_linje))]:
        raa = facit_linje[pos:start]
        tekst = raa.strip()
        if tekst:
            forskydning = len(raa) - len(raa.lstrip())
            stumper.append(
                Stump(
                    tekst=tekst,
                    start=pos + forskydning,
                    slut=pos + forskydning + len(tekst),
                    for_kort=len(tekst) < MINDSTE_STUMP,
                )
            )
        pos = slut
    return stumper


def forankr(
    facit_linje: str,
    modeltekst: str,
    *,
    nr: int = 0,
    fra: int = 0,
    maks_afvigelse: float = MAKS_AFVIGELSE,
) -> LinjeFund:
    """Finder facit-linjens kendte stumper i modellens tekst.

    Stumperne soeges fra venstre mod hoejre, hver efter den forriges traef, saa
    en gentaget vending ikke kan forankre bagud og lave et negativt gab.

    En stump, der ikke findes, er ikke en fejl -- den er bare ikke fundet.
    Findes ingen af linjens stumper, er linjen uforankret og gaar helt ud af
    maalingen (beslutning 38).
    """
    stumper = _del_i_stumper(facit_linje)
    resultat: list[Stump] = []
    pos = fra
    for stump in stumper:
        if stump.for_kort:
            resultat.append(stump)
            continue
        traef = _find_stump(stump.tekst, modeltekst, pos, maks_afvigelse)
        if traef is None:
            resultat.append(stump)
            continue
        start, slut = traef
        resultat.append(
            Stump(
                tekst=stump.tekst,
                start=stump.start,
                slut=stump.slut,
                for_kort=False,
                fundet=True,
                model_tekst=modeltekst[start:slut],
                model_start=start,
                model_slut=slut,
            )
        )
        pos = slut

    # Gab: kun mellem to stumper der BEGGE er fundet. Er der en ubrugelig
    # stump imellem, staar dens tekst i `facit_mellem`, saa laeseren kan se,
    # at gabet ikke er rent ulaeseligt.
    gab: list[Gab] = []
    fundne = [s for s in resultat if s.fundet]
    for venstre, hoejre in zip(fundne, fundne[1:]):
        gab.append(
            Gab(
                facit_mellem=facit_linje[venstre.slut : hoejre.start].strip(),
                model_tekst=modeltekst[venstre.model_slut : hoejre.model_start],
                model_start=venstre.model_slut,
                model_slut=hoejre.model_start,
            )
        )

    return LinjeFund(nr=nr, facit=facit_linje, stumper=tuple(resultat), gab=tuple(gab))


# --------------------------------------------------------------------------
# Fladning: samme orddelings-beslutninger paa begge sider
# --------------------------------------------------------------------------

def deler_ord(linjer: list[str]) -> list[bool]:
    """For hvert linjeskift: deler det ét ord over to linjer?

    Reglen er facits egen (beslutning 21): bindestreg sidst paa en linje deler
    kun et ord, naar naeste linje fortsaetter med lille bogstav -- materialet
    bruger nemlig ogsaa bindestreg som punktum ("enkelte Rhonchi-" efterfulgt
    af en ny saetning).

    Beslutningen traeffes paa FACITS linjer og bruges paa begge sider. Ellers
    ville "Infektions-" / "sygdomme." blive samlet i facit og staa som to ord
    hos en model, der skrev "Infektionssygdomme" i ét stykke -- og modellen
    ville blive straffet for at have laest rigtigt.
    """
    ud = []
    for nu, naeste in zip(linjer, linjer[1:]):
        s = nu.strip()
        n = naeste.strip()
        ud.append(s.endswith("-") and n[:1].islower())
    return ud


def flad(linjer: list[str], delinger: list[bool]) -> str:
    """Samler linjerne til én streng efter de givne orddelings-beslutninger.

    Skal give samme resultat som `facit.saml_orddeling`, naar delingerne er
    regnet ud af de samme linjer. `tests/test_maaling.py` holder de to op mod
    hinanden paa hele facit, saa de ikke kan skride fra hinanden.
    """
    if not linjer:
        return ""
    ud = [linjer[0].strip()]
    for i, naeste in enumerate(linjer[1:]):
        stykke = naeste.strip()
        if i < len(delinger) and delinger[i]:
            kerne = ud[-1]
            # Bindestregen falder kun vaek, naar der staar et bogstav foran
            # den. Staar den efter fx et ulaeselighedsmaerke, ved vi ikke,
            # hvad der blev delt -- saa bliver den staaende.
            if kerne.endswith("-") and len(kerne) > 1 and kerne[-2].isalpha():
                kerne = kerne[:-1]
            ud[-1] = kerne + stykke
        else:
            ud.append(stykke)
    return " ".join(d for d in ud if d)


# --------------------------------------------------------------------------
# Maaling af en hel side
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SideMaaling:
    image_name: str
    linjer: tuple[LinjeFund, ...]

    fladet: dict[str, cer.Maaltal] = field(default_factory=dict)
    pr_linje: dict[str, cer.Maaltal] = field(default_factory=dict)
    fuldside: dict[str, cer.Maaltal] | None = None

    facit_tegn_i_alt: int = 0
    facit_tegn_maalt: int = 0
    linjer_i_alt: int = 0
    linjer_maalt: int = 0
    svaere_linjer: int = 0
    svaere_linjer_reddet: int = 0

    model_tegn_i_alt: int = 0
    model_tegn_daekket: int = 0

    uden_linjeskift_indeni: int = 0
    egen_modellinje: int = 0

    @property
    def daekning(self) -> float:
        """Andel af facits tegn, tallene ovenfor faktisk er maalt paa.

        Naevneren taeller `[?]`-maerkerne med som tre tegn hver. Det er ikke
        rigtigt -- teksten bag et maerke har en ukendt laengde, saa ingen
        naevner er rigtig -- men det er den forsigtige regning: den lader
        daekningen se lidt daarligere ud, end den er, i stedet for omvendt.
        Paa hele facit er der 498 maerker mod knap 90.000 tegn, saa det
        flytter under to procentpoint.
        """
        return self.facit_tegn_maalt / self.facit_tegn_i_alt if self.facit_tegn_i_alt else 0.0

    @property
    def model_tegn_uforankret(self) -> int:
        """Modeltekst uden modstykke i facit. Det groveste opdigtnings-signal."""
        return max(0, self.model_tegn_i_alt - self.model_tegn_daekket)

    @property
    def gab(self) -> tuple[Gab, ...]:
        return tuple(g for linje in self.linjer for g in linje.gab)


def _tegn(tekst: str) -> int:
    """Tegn uden mellemrum -- saa layout ikke taeller med som indhold."""
    return len("".join(tekst.split()))


def maal_side(
    image_name: str,
    facit_linjer: list[str],
    modeltekst: str,
    *,
    maks_afvigelse: float = MAKS_AFVIGELSE,
) -> SideMaaling:
    """Maaler én side. Facits linjer forankres fortloebende i modellens tekst."""
    fund: list[LinjeFund] = []
    pos = 0
    for nr, linje in enumerate(facit_linjer):
        f = forankr(linje, modeltekst, nr=nr, fra=pos, maks_afvigelse=maks_afvigelse)
        if f.forankret:
            pos = f.model_udsnit[1]
        fund.append(f)

    # Fladet maaling: kun de forankrede linjer, paa begge sider.
    facit_stykker = [f.facit_maalt for f in fund if f.forankret]
    model_stykker = [f.model_maalt for f in fund if f.forankret]
    delinger = deler_ord(facit_stykker)
    facit_fladet = flad(facit_stykker, delinger)
    model_fladet = flad(model_stykker, delinger)

    fladet = {
        navn: cer.maal_par(facit_fladet, model_fladet, **valg)
        for navn, valg in cer.VARIANTER.items()
    }

    pr_linje: dict[str, cer.Maaltal] = {navn: cer.NUL for navn in cer.VARIANTER}
    for f in fund:
        if not f.forankret:
            continue
        for navn, valg in cer.VARIANTER.items():
            pr_linje[navn] = pr_linje[navn] + cer.maal_par(f.facit_maalt, f.model_maalt, **valg)

    # Kontroltal: paa sider helt uden ulaeselighedsmaerker kan hele siden
    # sammenlignes direkte, uden forankring. Er de to tal langt fra hinanden,
    # pynter forankringen paa resultatet, og det skal ses.
    fuldside = None
    if not any(f.svaer for f in fund):
        hele_facit = saml_orddeling("\n".join(facit_linjer))
        hele_model = saml_orddeling(modeltekst)
        fuldside = {
            navn: cer.maal_par(hele_facit, hele_model, **valg)
            for navn, valg in cer.VARIANTER.items()
        }

    # Linjetrofasthed: falder facit-linjen inden for én af modellens linjer,
    # og faar hver facit-linje sin egen? Svaret paa beslutning 35, maalt.
    linjeskift = [i for i, c in enumerate(modeltekst) if c == "\n"]

    def modellinje(pos_: int) -> int:
        return sum(1 for b in linjeskift if b < pos_)

    uden_skift = 0
    egen_linje = 0
    forrige_modellinje = -1
    for f in fund:
        if not f.forankret:
            continue
        a, b = f.model_udsnit
        if not any(a <= p < b for p in linjeskift):
            uden_skift += 1
        nu = modellinje(a)
        if nu != forrige_modellinje:
            egen_linje += 1
        forrige_modellinje = nu

    daekket = sum(_tegn(s.model_tekst) for f in fund for s in f.fundne)
    daekket += sum(_tegn(g.model_tekst) for f in fund for g in f.gab)

    return SideMaaling(
        image_name=image_name,
        linjer=tuple(fund),
        fladet=fladet,
        pr_linje=pr_linje,
        fuldside=fuldside,
        facit_tegn_i_alt=_tegn(saml_orddeling("\n".join(facit_linjer))),
        facit_tegn_maalt=_tegn(facit_fladet),
        linjer_i_alt=len(fund),
        linjer_maalt=sum(1 for f in fund if f.forankret),
        svaere_linjer=sum(1 for f in fund if f.svaer),
        svaere_linjer_reddet=sum(1 for f in fund if f.svaer and f.forankret),
        model_tegn_i_alt=_tegn(modeltekst),
        model_tegn_daekket=daekket,
        uden_linjeskift_indeni=uden_skift,
        egen_modellinje=egen_linje,
    )


@dataclass(frozen=True)
class SaetMaaling:
    """Alle maalte sider rullet op til ét tal pr. variant."""

    sider: tuple[SideMaaling, ...]

    def _sum(self, felt: str) -> dict[str, cer.Maaltal]:
        ud = {navn: cer.NUL for navn in cer.VARIANTER}
        for side in self.sider:
            for navn in cer.VARIANTER:
                ud[navn] = ud[navn] + getattr(side, felt)[navn]
        return ud

    @property
    def fladet(self) -> dict[str, cer.Maaltal]:
        return self._sum("fladet")

    @property
    def pr_linje(self) -> dict[str, cer.Maaltal]:
        return self._sum("pr_linje")

    @property
    def fuldside(self) -> dict[str, cer.Maaltal]:
        """Kontroltallet, kun over de sider der slet ingen [?] har."""
        ud = {navn: cer.NUL for navn in cer.VARIANTER}
        for side in self.sider:
            if side.fuldside is None:
                continue
            for navn in cer.VARIANTER:
                ud[navn] = ud[navn] + side.fuldside[navn]
        return ud

    @property
    def sider_med_fuldsidekontrol(self) -> int:
        return sum(1 for s in self.sider if s.fuldside is not None)

    @property
    def daekning(self) -> float:
        i_alt = sum(s.facit_tegn_i_alt for s in self.sider)
        maalt = sum(s.facit_tegn_maalt for s in self.sider)
        return maalt / i_alt if i_alt else 0.0

    @property
    def linjedaekning(self) -> float:
        i_alt = sum(s.linjer_i_alt for s in self.sider)
        maalt = sum(s.linjer_maalt for s in self.sider)
        return maalt / i_alt if i_alt else 0.0

    @property
    def gab(self) -> tuple[tuple[str, Gab], ...]:
        return tuple((s.image_name, g) for s in self.sider for g in s.gab)


def maal_saet(
    poster: list[dict],
    modeltekster: dict[str, str],
    *,
    felt: str = "alt_linjer",
    maks_afvigelse: float = MAKS_AFVIGELSE,
) -> SaetMaaling:
    """Maaler alle sider, der har baade facit og et modelsvar.

    `felt` er `alt_linjer` som standard: modellen bliver bedt om at laese hele
    siden og bliver udtrykkeligt ikke bedt om at afgoere, hvad der er streget
    ud (beslutning 24). Maaltes der mod `rettet_linjer`, ville den blive
    straffet 33 steder for at goere netop det, vi bad om.

    Sider gennemloebes i sorteret raekkefoelge, saa to koersler giver samme
    rapport.
    """
    sider = []
    for post in sorted(poster, key=lambda p: p["image_name"]):
        navn = post["image_name"]
        if navn not in modeltekster:
            continue
        sider.append(
            maal_side(navn, post[felt], modeltekster[navn], maks_afvigelse=maks_afvigelse)
        )
    return SaetMaaling(sider=tuple(sider))
