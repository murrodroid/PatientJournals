"""Maaleapparatet: ÉN sammenligning af hele siden, i raekkefoelge, uden soegning.

Indtil 2026-08-31 hvilede stagen paa `forankr()`: hver facit-linje blev SOEGT
frem i modellens tekst, fra sidste traef og fremefter. Den mekanisme er fjernet.
Paa `273107_001864` staar "ingen Snue" to gange i facit selv; soegningen efter
linje 1 foretrak det ordrette traef nede i linje 26, flyttede soegepunktet
dertil, og de 24 mellemliggende linjer var derefter uden for raekkevidde. 26 af
29 linjer tabt -- nok til at vende hele wordpicking-forsoegets rangorden. Stage
03 blev laast med den udtrykkelige betingelse, at den genaabnes, hvis der viser
sig et hul ved de foerste rigtige tal. Hullet var der.

Her sammenlignes facits fulde tekst med modellens fulde tekst i ét straek, fra
top til bund. Selve maaleoperationen ligger i `sidemaaling.py`; dette modul
laegger facit til rette, koerer de seks varianter, og samler tallene.

## Hvad der foelger af det

- **Der er kun én vej gennem siden.** Et gentaget ord kan ikke flytte noget.
- **"Daekning" og "rabat" findes ikke laengere.** Hele facit er altid i
  naevneren. Dermed forsvinder ogsaa den faelde, der fik den forkerte
  konklusion igennem to gange paa én dag: et hovedtal maalt paa kun de fundne
  linjer gav rabat til netop den variant, der afveg mest.
- **Gabene falder mere direkte ud end foer.** Et gab er den modeltekst, der
  blev stillet op mod et `[?]`-jokerfelt -- ikke laengere noget, der skal
  udledes af to fundne stumper. Gab-filen er kontraktbundet (rod-CONTEXT
  2026-08-21) og er samtidig arbejdslisten over steder, facit maaske kan
  rettes.

## Tre ting, der IKKE er en simpel oversaettelse

**Den strenge maaling** (beslutning 44) udelader hele linjer med et `[?]`.
Uden forankringen findes der ingen "modellens modstykke til denne linje" at
udelade paa modelsiden. Derfor: hele linjen erstattes af ét jokermaerke, som
faar lov at sluge lige saa meget, som linjen selv indeholdt, plus
standardloftet. Maalingen bliver dermed den samme ene gennemgang af siden, blot
med de svaere linjer gjort gratis. Naevneren falder med netop de linjers tegn --
en FAST udeladelse, ens for alle varianter, ikke den glidende rabat.

**`pr_linje` er fjernet, ikke bygget om.** Den summerede en maaling pr. parret
linje, og parringen VAR forankringen. Den kunne genskabes fra `orden.py`s
linjeparring, men saa ville et hovedtal arve den parrings kendte svaghed
(graadigt venstre-mod-hoejre valg mellem naesten ens linjer). Planen af
2026-08-30 skrev "bygges om"; det er fravalgt med aabne oejne.

**`fuldside` er fjernet, ikke bygget om.** Den sammenlignede hele siden direkte
paa sider helt uden `[?]` -- som kontrol MOD forankringen. Nu er hele siden
altid maalt direkte, saa kontrollen og det den kontrollerede er blevet det
samme tal. At beholde den ville vaere at rapportere hovedtallet to gange.

Linjetrofastheden (beslutning 35, og forudsaetningen for at aflevere
`PageLine`-poster videre) maales nu af `orden.py`: hvor mange facit-linjer der
har et genkendeligt modstykke, og hvor mange der staar i en anden raekkefoelge.
"""
from __future__ import annotations

from dataclasses import dataclass

from andenside import cer, orden, sidemaaling
from andenside.facit import saml_orddeling
from andenside.sidemaaling import JOKER_LOFT, MAERKE

# Hvor mange ord af facit der staar paa hver side af et gab, saa laeseren kan
# finde stedet paa siden igen. Gabet i sig selv er tit ét ord langt og ville
# ellers vaere umuligt at placere.
GAB_KONTEKST_ORD = 4


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


def _tegn(tekst: str) -> int:
    """Tegn uden mellemrum -- saa layout ikke taeller med som indhold."""
    return len("".join(tekst.split()))


# --------------------------------------------------------------------------
# Den strenge maaling: svaere linjer erstattes af ét jokermaerke hver
# --------------------------------------------------------------------------

def streng_facit(facit_linjer: list[str]) -> tuple[str, list[int]]:
    """Facit, hvor hver linje med et `[?]` er erstattet af ét jokermaerke.

    Returnerer den fladede tekst og ét loft pr. erstattet linje.

    Loftet er linjens eget indhold, hverken mere eller mindre. Grunden til ikke
    at give jokeren frit lejde: et loft paa uendelig lader modellen springe
    vilkaarligt langt frem i sin egen tekst gratis, og saa kan en model, der
    springer en hel blok over, faa sine rigtige fejl slugt af jokeren ved siden
    af. Loftet siger: du maa skrive lige saa meget, som der stod paa den linje,
    vi ikke kan laese.

    Der laegges IKKE `JOKER_LOFT` oveni linjen. De 15 tegn er udledt af, hvor
    langt ét ord er i materialet, og for en HEL linje er linjens egen laengde
    den tilsvarende udledning; laegges de to sammen, taelles samme begrundelse
    to gange, og den strenge maaling bliver mildere, end den giver sig ud for.

    Men MAERKET SELV er ikke tekst. `[?]` staar i stedet for indhold, hvis
    laengde ingen kender -- taelles de tre tegn som tre bogstaver, er loftet
    tre tegn for lavt hver eneste gang, og en model, der skriver et helt
    almindeligt ord dér hvor et menneske gav op, betaler for det. Maalt
    2026-09-01: det gav 3,21 % paa "facit mod sig selv" i selvtesten, hvor
    kravet er nul. Maerket taeller derfor med `JOKER_LOFT` -- den samme
    rummelighed, et `[?]` faar inde i en linje.

    Orddelings-beslutningerne traeffes paa de OPRINDELIGE linjer og bruges paa
    de erstattede. En bindestreg sidst paa en erstattet linje er forsvundet med
    linjen, saa `flad` samler alligevel ikke hen over den; beslutningen tages
    paa forhaand for at holde reglen ét sted.
    """
    delinger = deler_ord(facit_linjer)
    erstattet = [MAERKE if MAERKE in linje else linje for linje in facit_linjer]
    lofter = [_tegn(linje.replace(MAERKE, "x" * JOKER_LOFT))
              for linje in facit_linjer if MAERKE in linje]
    return flad(erstattet, delinger), lofter


# --------------------------------------------------------------------------
# Fra sidemaalingens tal til de Maaltal, rapporten regner med
# --------------------------------------------------------------------------

def _som_maaltal(maal: sidemaaling.SideMaal) -> cer.Maaltal:
    """Ét `SideMaal` som et `Maaltal`, saa sider kan laegges sammen.

    `stykker` er nu SIDER, ikke linjer. Under forankringen var et stykke en
    parret linje, og `andel_identiske` betoed "andel helt rigtige linjer". Det
    tal findes stadig, men kommer nu fra linjeparringen i `orden.py` og staar
    for sig i rapporten -- det maa ikke forveksles med dette.
    """
    return cer.Maaltal(
        tegnafstand=maal.tegnafstand,
        facit_tegn=maal.facit_tegn,
        ordafstand=maal.ordafstand,
        facit_ord=maal.facit_ord,
        stykker_maalt=1,
        stykker_identiske=int(maal.tegnafstand == 0),
    )


@dataclass(frozen=True)
class Gab:
    """Det modellen skrev, hvor facit siger `[?]`.

    `facit_foer` og `facit_efter` er facits egne ord omkring maerket. De staar
    med, fordi gabet skal kunne findes igen paa siden med det blotte oeje --
    modelteksten alene er tit ét ord og kan staa hvor som helst.
    """

    facit_foer: str
    facit_efter: str
    model_tekst: str

    @property
    def indholdstegn(self) -> int:
        return _tegn(self.model_tekst)


def _gab_kontekst(facit_fladet: str) -> list[tuple[str, str]]:
    """Facits ord lige foer og lige efter hvert jokermaerke, i raekkefoelge."""
    ud: list[tuple[str, str]] = []
    stykker = facit_fladet.split(MAERKE)
    for nr in range(len(stykker) - 1):
        foer = " ".join(stykker[nr].split()[-GAB_KONTEKST_ORD:])
        efter = " ".join(stykker[nr + 1].split()[:GAB_KONTEKST_ORD])
        ud.append((foer, efter))
    return ud


# --------------------------------------------------------------------------
# Maaling af en hel side
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SideMaaling:
    """Tal for ÉN side. Hele siden er altid maalt -- der er ingen daekning."""

    image_name: str

    # Hovedtallet og den strenge maaling, hver i de seks varianter
    # (beslutning 26 -- alle staar side om side, ingen vaelges efter,
    # hvilken der klaeder resultatet bedst).
    fladet: dict[str, cer.Maaltal]
    rene: dict[str, cer.Maaltal]

    # Raekkefoelgen som sit eget tal (lead 2026-08-30). Maalingen ovenfor er
    # streng om orden; DENNE viser, hvor meget af fejlen der er omrokering.
    omrokering: orden.Omrokering

    gab: tuple[Gab, ...]
    joker_tegn_i_alt: int
    joker_overskud: int

    facit_tegn_i_alt: int
    linjer_i_alt: int
    svaere_linjer: int
    rene_linjer_i_alt: int
    rene_tegn_i_alt: int
    model_tegn_i_alt: int

    @property
    def identiske_linjer(self) -> int:
        return self.omrokering.linjer_identiske


def maal_side(
    image_name: str,
    facit_linjer: list[str],
    modeltekst: str,
) -> SideMaaling:
    """Maaler én side: hele facit mod hele modelteksten, i raekkefoelge."""
    delinger = deler_ord(facit_linjer)
    facit_fladet = flad(facit_linjer, delinger)

    # Orddelingen samles ogsaa paa modelsiden (beslutning 42). Uden det maales
    # facits "Infektions-"/"sygdomme." som ét ord, mens modellens samme to
    # linjer staar som to -- og modellen ville blive straffet for en forskel,
    # der kun er facits egen typografi. Reglen er indholdsbaaret (bindestreg
    # sidst paa linjen, lille bogstav paa den naeste), saa den giver samme
    # svar, hvad enten modellen skrev ordet delt eller samlet.
    model_fladet = saml_orddeling(modeltekst)

    fladet = {
        navn: _som_maaltal(sidemaaling.maal_side(facit_fladet, model_fladet, **valg))
        for navn, valg in cer.VARIANTER.items()
    }

    # Den strenge maaling (beslutning 44, lead 2026-08-23): linjer med et `[?]`
    # slet ikke med. Den var oprindeligt et vaern mod forankringens glidende
    # rabat, og den rabat findes ikke laengere -- men den bevares, fordi netop
    # den fremgangsmaade er konventionen i HTR (Transkribus udelader hele
    # linjen ved ulaeselige steder). Beholdes tallet, kan vores resultater
    # sammenlignes med anden forskning.
    strengt, strenge_lofter = streng_facit(facit_linjer)
    rene = {
        navn: _som_maaltal(
            sidemaaling.maal_side(
                strengt, model_fladet, lofter=list(strenge_lofter), **valg
            )
        )
        for navn, valg in cer.VARIANTER.items()
    }

    # Gabene tages fra arbejdstallet: det er den variant, der laeses efter i
    # haanden, og maerkerne staar samme sted i alle seks.
    arbejds = sidemaaling.maal_side(
        facit_fladet, model_fladet, **cer.VARIANTER["arbejdstal"]
    )
    kontekst = _gab_kontekst(facit_fladet)
    gab = tuple(
        Gab(facit_foer=foer, facit_efter=efter, model_tekst=tekst)
        for (foer, efter), tekst in zip(kontekst, arbejds.joker_tekst)
    )

    rene_linjer = [linje for linje in facit_linjer if MAERKE not in linje]

    return SideMaaling(
        image_name=image_name,
        fladet=fladet,
        rene=rene,
        omrokering=orden.maal_omrokering(facit_linjer, modeltekst.split("\n")),
        gab=gab,
        joker_tegn_i_alt=arbejds.joker_tegn_i_alt,
        joker_overskud=arbejds.joker_overskud,
        facit_tegn_i_alt=_tegn(saml_orddeling("\n".join(facit_linjer))),
        linjer_i_alt=len(facit_linjer),
        svaere_linjer=sum(1 for linje in facit_linjer if MAERKE in linje),
        rene_linjer_i_alt=len(rene_linjer),
        rene_tegn_i_alt=_tegn(flad(rene_linjer, deler_ord(rene_linjer))),
        model_tegn_i_alt=_tegn(modeltekst),
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
    def rene(self) -> dict[str, cer.Maaltal]:
        """Den strenge maaling: kun linjer helt uden ulaeselige steder."""
        return self._sum("rene")

    @property
    def andel_af_facit_i_rene(self) -> float:
        """Hvor stor en del af facits tegn den strenge maaling overhovedet ser.

        Resten ligger paa linjer med mindst ét `[?]`. Det er en FAST
        udeladelse: den er den samme for alle varianter, fordi den kun
        afhaenger af facit. Den maa ikke laeses som den glidende, variant-
        afhaengige daekning, forankringen havde -- det var netop den, der
        gav rabat til den variant, som afveg mest.
        """
        i_alt = sum(s.facit_tegn_i_alt for s in self.sider)
        rene = sum(s.rene_tegn_i_alt for s in self.sider)
        return rene / i_alt if i_alt else 0.0

    @property
    def linjer_i_alt(self) -> int:
        return sum(s.linjer_i_alt for s in self.sider)

    @property
    def linjer_parret(self) -> int:
        return sum(s.omrokering.linjer_parret for s in self.sider)

    @property
    def linjer_omrokeret(self) -> int:
        return sum(s.omrokering.antal_flyttede for s in self.sider)

    @property
    def identiske_linjer(self) -> int:
        return sum(s.identiske_linjer for s in self.sider)

    @property
    def joker_tegn_i_alt(self) -> int:
        return sum(s.joker_tegn_i_alt for s in self.sider)

    @property
    def joker_overskud(self) -> int:
        return sum(s.joker_overskud for s in self.sider)

    @property
    def gab(self) -> tuple[tuple[str, Gab], ...]:
        return tuple((s.image_name, g) for s in self.sider for g in s.gab)


def maal_saet(
    poster: list[dict],
    modeltekster: dict[str, str],
    *,
    felt: str = "alt_linjer",
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
        sider.append(maal_side(navn, post[felt], modeltekster[navn]))
    return SaetMaaling(sider=tuple(sider))
