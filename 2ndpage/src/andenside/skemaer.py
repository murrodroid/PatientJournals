"""Skemavarianterne til stage 06's forsoeg -- wordpicking mod full scan.

Baggrund. Kollegaens `FrontPage`-skema er gennemsyret af stedanvisninger:
`ward` staar "strictly in the uppermost top-left corner", `severity` kan
"overlap with the margin in the far left side of the page, make sure to check
there". Hans `TextPage` har til sammenligning én stedanvisning i alt. Det er
den sondring, forsoeget skal maale: hjaelper det at fortaelle modellen HVOR
tingene staar, naar siden er loebende prosa i stedet for navngivne felter?

Litteraturen peger den anden vej. Struktureret output maales til at forringe
fri generering -- JSON-syntaksen konkurrerer med indholdet om modellens
opmaerksomhed. Derfor er `ren_tekst` med som variant uden skema overhovedet:
det er den ene ende af skalaen, og uden den ved vi ikke, hvilken RETNING mere
struktur traekker tallet.

**Feltbeskrivelserne sendes til modellen.** Gemini laegger `Field(description=)`
ind i det skema, modellen ser. En beskrivelse er altsaa prompttekst, ikke
dokumentation, og skal skrives som saadan. Det er ogsaa grunden til, at vores
egen bare kopi af `TextPage` var en fejl og ikke et valg: vi koerte et andet
skema end det, kollegaens app koerer.

Alt hvad der staar i beskrivelserne under `bar` og `beskrevet` er ordret
kollegaens (`src/patientjournals/config/schemas.py`). Roer dem ikke -- de er
maalestokken, ikke vores bidrag.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# V0 `bar` -- som vi hidtil har koert. Bevares, saa de foerste tal stadig kan
# genskabes, men det er IKKE kollegaens skema.
# ---------------------------------------------------------------------------

class BarPageLine(BaseModel):
    text: str
    metadata: str | None = None


class BarTextPage(BaseModel):
    page_lines: list[BarPageLine]


# ---------------------------------------------------------------------------
# V1 `beskrevet` -- kollegaens skema, ordret, med hans feltbeskrivelser.
# Det er det, hans app faktisk sender.
# ---------------------------------------------------------------------------

class BeskrevetPageLine(BaseModel):
    text: str = Field(
        description="This includes all text described that isn't written "
                    "seperated from the line."
    )
    metadata: str | None = Field(
        None,
        description="Contains the metadata of the line, can describe dates, "
                    "temperatures, etc. Most commonly written in the "
                    "left-side, before normal text is written.",
    )


class BeskrevetTextPage(BaseModel):
    page_lines: list[BeskrevetPageLine] = Field(
        description="This is meant for each line on the page, seperated by "
                    "linebreaks."
    )


# ---------------------------------------------------------------------------
# V4 `linjefelter` -- aegte wordpicking: linjen delt i sine dele.
#
# Bemaerk hvad optaellingen af alle 2.586 oevelinjer siger: datomargen rammer
# 3,2 %, temperatur/puls 3,6 %, talkolonner 4,8 %. Over 80 % af linjerne er
# prosa uden nogen af delene. Felterne er derfor `None` det meste af tiden, og
# det er MENINGEN -- forsoeget skal vise, om det at navngive de faa hjaelper
# eller forstyrrer de mange.
# ---------------------------------------------------------------------------

class DeltPageLine(BaseModel):
    dato: str | None = Field(
        None,
        description="A date written in the left margin, before the ruled line "
                    "or fold, in the form day/month such as '19/12'. Only the "
                    "date itself. Most lines have no margin date; leave this "
                    "null when there is none.",
    )
    maalinger: str | None = Field(
        None,
        description="Clinical measurements written as bare numbers at the "
                    "start of the line, usually a morning and an evening "
                    "temperature separated by a slash, such as '39.5/39.4', "
                    "sometimes followed by a pulse. Transcribe the digits and "
                    "separators exactly as written, including whether the "
                    "decimal mark is a comma or a period. Leave null when the "
                    "line has no such numbers.",
    )
    text: str = Field(
        description="The running text of the line, after any margin date and "
                    "any leading measurements have been taken out. This is "
                    "late 19th-century Danish medical prose and is the main "
                    "content of the page. Never leave this empty for a line "
                    "that has writing in it.",
    )


class DeltTextPage(BaseModel):
    page_lines: list[DeltPageLine] = Field(
        description="One entry for every distinct line of writing on the "
                    "page, in top-to-bottom order."
    )


# ---------------------------------------------------------------------------
# V5 `usikkerhed` -- pick paa det hyppigste strukturelle element vi HAR.
#
# `[?]` staar paa 11,5 % af oevelinjerne -- tre gange saa hyppigt som
# datomargenen. Skal noget picks paa en tekstside, er det dette. Feltet er
# ikke et transskriptionsmaal; det leverer ind i stage 07's uenighedsmarkering
# og i gennemsynsbyrden.
# ---------------------------------------------------------------------------

class UsikkerPageLine(BaseModel):
    text: str = Field(
        description="This includes all text described that isn't written "
                    "seperated from the line."
    )
    metadata: str | None = Field(
        None,
        description="Contains the metadata of the line, can describe dates, "
                    "temperatures, etc. Most commonly written in the "
                    "left-side, before normal text is written.",
    )
    usikker: str | None = Field(
        None,
        description="The words in this line that you could not read with "
                    "confidence, copied exactly as you wrote them in 'text', "
                    "separated by ' | '. Report a word here whenever a "
                    "different reading seems roughly as likely as the one you "
                    "chose. Leave null when you are confident about the whole "
                    "line. Do not change 'text' because of this field: still "
                    "give your best reading there.",
    )


class UsikkerTextPage(BaseModel):
    page_lines: list[UsikkerPageLine] = Field(
        description="This is meant for each line on the page, seperated by "
                    "linebreaks."
    )


# `ren_tekst` har med vilje ingen model: varianten sender intet skema.
SKEMAER: dict[str, type[BaseModel] | None] = {
    "bar": BarTextPage,
    "beskrevet": BeskrevetTextPage,
    "linjefelter": DeltTextPage,
    "usikkerhed": UsikkerTextPage,
    "ren_tekst": None,
}


# ---------------------------------------------------------------------------
# Fra skemasvar til én tekst
#
# Maaleapparatet vil have sammenhaengende tekst. Hvordan et skemasvar foldes
# ud til tekst er derfor et VALG, og det valg kan afgoere forsoeget, hvis man
# ikke passer paa:
#
# Kollegaens app laegger margendatoen i `metadata` og ikke i `text`. Facit har
# den inline i linjen. Foldes `metadata` ikke ind, taeller hver margendato som
# manglende tekst -- maalt til 22 tegn paa otte sider. `linjefelter` samler
# derimod sine dele igen, saa datoen ER med.
#
# Sammenlignes de to uden videre, vinder `linjefelter` delvist paa den
# forskel alene, og ikke paa den feltopdeling, forsoeget skulle afgoere. Derfor
# er `med_metadata` et argument og ikke en fast regel: samme svar kan foldes
# ud paa begge maader uden et nyt modelkald, og begge tal rapporteres.
# ---------------------------------------------------------------------------

# Raekkefoelgen linjens dele laeses i. Den foelger sidens egen: margendatoen
# staar yderst til venstre, saa maalingerne, saa broedteksten.
DELE_I_LAESEORDEN = ("dato", "metadata", "maalinger", "text")


def saml_linje(post: dict, *, med_metadata: bool) -> str:
    """Én linjes felter samlet til den tekst, linjen faktisk bestaar af.

    `med_metadata` styrer kun de felter, kollegaens app holder UDE af `text`
    (`metadata`). `linjefelter`s egne dele er altid med -- de er selve
    varianten, ikke et sidespor.
    """
    if "text" not in post:
        raise ValueError(f"linjen mangler feltet 'text': {sorted(post)}")
    dele = []
    for felt in DELE_I_LAESEORDEN:
        if felt == "metadata" and not med_metadata:
            continue
        vaerdi = post.get(felt)
        if vaerdi is None:
            continue
        vaerdi = str(vaerdi).strip()
        if vaerdi:
            dele.append(vaerdi)
    return " ".join(dele)


def tekst_af_svar(raat: dict, *, med_metadata: bool = False) -> str:
    """Hele svaret som én tekst, uanset hvilken skemavariant der blev brugt.

    `ren_tekst` har ingen linjeposter -- der er teksten allerede tekst.
    """
    if "ren_tekst" in raat:
        return raat["ren_tekst"]
    sider = raat.get("page_lines")
    if not sider:
        raise ValueError(
            "svaret har ingen linjer. Det er en fejlet koersel, ikke en tom side"
        )
    return "\n".join(saml_linje(p, med_metadata=med_metadata) for p in sider)
