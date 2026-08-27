"""Modellaget: kaldet til gemini-3.1-pro og oversaettelsen af svaret.

Projektet afleverer en prompt og et skema til kollegaens app -- ikke
koerselskode. Det her er derfor med vilje tyndt: nok til at maale, om prompten
virker, og ikke mere.

**Noeglen har sit EGET sted for dette projekt** -- se `NOEGLEFIL` nedenfor.
Modulet kender med vilje ingen anden noeglefil: mangler projektets egen,
fejler koerslen i stedet for at lede videre til en tilfaeldig noegle, der
maatte ligge paa maskinen.

Filen laeses ved koersel. Vi kender ikke feltnavnet indeni, og den skal ikke aabnes af et
menneske eller en agent for at finde ud af det -- opslaget finder selv et felt,
der ligner en Gemini-noegle, og fejler tydeligt med en liste over de feltnavne,
der ER i filen, hvis der ikke er et. Selve vaerdien staar aldrig i en fejl,
en log eller et bogholderi. Paa sigt gaar kaldet gennem kollegaens Vertex/GCS.

**Svaret.** Modellen svarer efter kollegaens `TextPage`-skema: én post pr.
linje med `text` og `metadata`. Maaleapparatet vil have én sammenhaengende
tekst. Den oversaettelse ligger her og er testet, fordi et tab dér ligner et
daarligt modelsvar i tallet.

Skemasvaret loeser samtidig stage 03's advarsel: der kommer ingen "Her er
transskriptionen:" eller markdown-hegn ud af et skemabundet svar, saa der er
intet at rense -- og dermed intet at skulle notere i bogholderiet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Projektets EGEN noeglefil, bevidst adskilt fra andre noeglefiler paa
# maskinen. Ligger uden for repoet, saa et uagtsomt `git add -A` ikke kan
# publicere den.
NOEGLEFIL = Path(
    os.environ.get("ANDENSIDE_NOEGLEFIL", r"C:\Work\2ndpage_keys.json")
)

# Feltnavne, der peger paa en Gemini-noegle. Roert i den raekkefoelge, saa en
# fil med flere udbydere ikke giver os OpenAI's noegle ved et uheld.
GEMINI_ORD = ("gemini", "genai", "google")
NOEGLE_ORD = ("api_key", "apikey", "key", "noegle", "secret", "token")


class NoegleFejl(RuntimeError):
    """Rejses, naar der ikke kan findes en brugbar Gemini-noegle."""


def _ligner_gemini(navn: str) -> bool:
    lavt = navn.lower()
    return any(ord_ in lavt for ord_ in GEMINI_ORD)


def _ligner_noegle(navn: str) -> bool:
    lavt = navn.lower()
    return any(ord_ in lavt for ord_ in NOEGLE_ORD)


def _feltnavne(data: Any, praefiks: str = "") -> list[str]:
    """Alle feltnavne i strukturen -- til fejlbeskeden, aldrig vaerdierne."""
    if not isinstance(data, dict):
        return []
    navne = []
    for navn, vaerdi in data.items():
        fuldt = f"{praefiks}{navn}"
        navne.append(fuldt)
        navne.extend(_feltnavne(vaerdi, f"{fuldt}."))
    return navne


def find_noegle(data: dict) -> str:
    """Finder Gemini-noeglen i en indlaest noeglefil-struktur.

    Taaler baade et fladt felt (`{"gemini": "..."}`) og en gruppering pr.
    udbyder (`{"providers": {"gemini": {"api_key": "..."}}}`).
    """
    fundet = _soeg(data, gemini_kontekst=False)
    if fundet is None:
        raise NoegleFejl(
            "ingen Gemini-noegle fundet i noeglefilen. Felterne i filen er: "
            + ", ".join(_feltnavne(data) or ["(ingen)"])
            + ". Giv feltet et navn, der rummer 'gemini', 'genai' eller 'google'."
        )
    return fundet


def _soeg(data: Any, *, gemini_kontekst: bool) -> str | None:
    if isinstance(data, str):
        # En streng taeller kun, hvis vejen hertil allerede pegede paa Gemini.
        return data.strip() if gemini_kontekst and data.strip() else None
    if not isinstance(data, dict):
        return None

    # Foerst felter, hvis eget navn peger paa Gemini.
    for navn, vaerdi in data.items():
        if _ligner_gemini(navn):
            fundet = _soeg(vaerdi, gemini_kontekst=True)
            if fundet:
                return fundet
    # Derefter -- inde i en Gemini-gruppe -- et felt, der ligner en noegle.
    if gemini_kontekst:
        for navn, vaerdi in data.items():
            if _ligner_noegle(navn):
                fundet = _soeg(vaerdi, gemini_kontekst=True)
                if fundet:
                    return fundet
    # Til sidst videre ned i grupper, der ikke selv naevner en udbyder.
    for navn, vaerdi in data.items():
        if isinstance(vaerdi, dict) and not _ligner_gemini(navn):
            fundet = _soeg(vaerdi, gemini_kontekst=gemini_kontekst)
            if fundet:
                return fundet
    return None


def hent_noegle(sti: Path = NOEGLEFIL) -> str:
    """Læser noeglefilen og finder Gemini-noeglen.

    Filens indhold gaar udelukkende herind -- det logges ikke, gemmes ikke og
    vises ikke.
    """
    if not sti.exists():
        raise NoegleFejl(
            f"projektets noeglefil findes ikke: {sti}. "
            f"Opret den med et felt, hvis navn rummer 'gemini', 'genai' eller "
            f"'google'. Stien kan flyttes med ANDENSIDE_NOEGLEFIL."
        )
    try:
        data = json.loads(sti.read_text(encoding="utf-8"))
    except json.JSONDecodeError as fejl:
        raise NoegleFejl(f"noeglefilen kunne ikke laeses som JSON: {fejl.msg}") from None
    return find_noegle(data)


def tekst_af_sider(sider: list[dict]) -> str:
    """Samler modellens linje-poster til én tekst, som maaleapparatet vil have.

    `metadata` (margendatoer) kommer IKKE med. De taeller med i facit som
    tekst, men de staar dér, hvor de staar paa siden -- vi maa ikke lime dem
    ind et vilkaarligt sted. Mangler de i svaret, er det en promptsag.
    """
    if not sider:
        raise ValueError(
            "modellen returnerede et tomt svar (nul linjer). Det er en fejlet "
            "koersel, ikke en tom side"
        )
    linjer = []
    for nummer, post in enumerate(sider, start=1):
        if "text" not in post:
            raise ValueError(
                f"linje {nummer} i svaret mangler feltet 'text' -- skemaet blev "
                f"ikke fulgt"
            )
        linjer.append(post["text"])
    return "\n".join(linjer)


# ---------------------------------------------------------------------------
# Selve kaldet
#
# Skemaet er kollegaens `TextPage`/`PageLine` (se
# references/app_interface_upstream.md). Vi genskaber det her i stedet for at
# importere hans pakke, fordi projektet afleverer en prompt og et skema til
# HANS app -- ikke omvendt. Felterne skal matche hans, ellers maaler vi noget
# andet, end han kommer til at koere.
# ---------------------------------------------------------------------------

class PageLine(BaseModel):
    text: str
    metadata: str | None = None


class TextPage(BaseModel):
    page_lines: list[PageLine]


def transskriber(
    billede: Path,
    prompt: str,
    *,
    model: str = "gemini-3.1-pro",
    temperatur: float = 0.0,
    noegle: str | None = None,
) -> tuple[str, dict]:
    """Sender ét billede til modellen og returnerer `(tekst, raat_svar)`.

    `tekst` er linjerne samlet, som maaleapparatet vil have dem. `raat_svar`
    er skemaets egen struktur -- den gemmes i bogholderiet, saa margendatoer
    og linjeopdeling ikke gaar tabt, bare fordi maalingen ikke bruger dem.
    """
    from google import genai
    from google.genai import types

    klient = genai.Client(api_key=noegle or hent_noegle())
    svar = klient.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=billede.read_bytes(),
                mime_type=f"image/{billede.suffix.lstrip('.').lower()}",
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=temperatur,
            response_mime_type="application/json",
            response_schema=TextPage,
        ),
    )
    raat = json.loads(svar.text)
    return tekst_af_sider(raat["page_lines"]), raat
