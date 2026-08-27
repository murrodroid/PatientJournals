"""Bogholderi omkring en modelkoersel.

Stage 05's kontrakt: "Gem altid det raa modelsvar sammen med model,
promptversion, indstillinger og dato, saa en koersel kan genfindes og
genkoeres."

Et raat modelsvar uden sin opsaetning er vaerdiloest. Blev tallet daarligt,
kan vi ikke vide, om det var prompten, billedvarianten eller modellen -- og vi
kan ikke koere den om. Derfor er de to ting bundet sammen i én mappe, og der
findes ingen vej til at gemme det ene uden det andet.

Svaret gemmes RAAT. Stage 03 advarer om, at en indledning som "Her er
transskriptionen:" taelles som digtning, men beslutningen om at rense hoerer
til maalingen -- bogholderiet skal bevare, hvad modellen faktisk skrev.

Der er ingen noegle nogen steder i det, der skrives her. Bogholderiet skal
kunne deles og committes uden eftersyn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

VARIANTER = ("helt_opslag", "beskaaret")

OPSAETNING_FIL = "opsaetning.json"
SVAR_MAPPE = "svar"


@dataclass(frozen=True)
class Opsaetning:
    """Alt det, der skal til for at genkende og genkoere en koersel."""

    model: str
    promptversion: str
    prompt: str
    variant: str
    temperatur: float
    noter: str = ""

    @property
    def prompt_aftryk(self) -> str:
        """Kort fingeraftryk af selve promptteksten.

        Promptversionen er et NAVN, mennesker giver. Aftrykket afsloerer, at
        teksten bag navnet har flyttet sig -- ellers kan to koersler se ens ud
        i regnskabet og alligevel have brugt hver sin prompt.
        """
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:12]

    def kontrollér(self) -> None:
        if not self.model.strip():
            raise ValueError("opsaetningen mangler et model-navn -- koerslen "
                             "ville ikke kunne genfindes bagefter")
        if not self.promptversion.strip():
            raise ValueError("opsaetningen mangler en promptversion")
        if not self.prompt.strip():
            raise ValueError("opsaetningen mangler selve promptteksten")
        if self.variant not in VARIANTER:
            raise ValueError(
                f"ukendt variant {self.variant!r} -- kendte er {', '.join(VARIANTER)}"
            )

    def som_dict(self, *, dato: str) -> dict:
        return {
            "dato": dato,
            "model": self.model,
            "promptversion": self.promptversion,
            "prompt_aftryk": self.prompt_aftryk,
            "variant": self.variant,
            "temperatur": self.temperatur,
            "noter": self.noter,
            "prompt": self.prompt,
        }


def _mappenavn(opsaetning: Opsaetning, dato: str) -> str:
    stempel = dato.replace(":", "").replace("-", "").replace("T", "_")
    return f"{stempel}_{opsaetning.variant}"


def gem_koersel(
    rod: Path, opsaetning: Opsaetning, svar: dict[str, str], *, dato: str | None = None
) -> Path:
    """Gemmer én koersel og returnerer mappen, den ligger i.

    To koersler skriver aldrig oven i hinanden -- heller ikke to inden for
    samme sekund. Datoen skelner dem i praksis; taelleren er der, fordi tabte
    data er en vaerre fejl end et grimt mappenavn.
    """
    if not isinstance(opsaetning, Opsaetning):
        raise TypeError(
            "gem_koersel kraever en Opsaetning -- et raat svar uden opsaetning "
            "kan ikke gemmes"
        )
    opsaetning.kontrollér()

    dato = dato or datetime.now().replace(microsecond=0).isoformat()
    grund = rod / _mappenavn(opsaetning, dato)
    mappe = grund
    tael = 2
    while mappe.exists():
        mappe = grund.with_name(f"{grund.name}_{tael}")
        tael += 1

    (mappe / SVAR_MAPPE).mkdir(parents=True)
    (mappe / OPSAETNING_FIL).write_text(
        json.dumps(opsaetning.som_dict(dato=dato), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for billede, tekst in sorted(svar.items()):
        (mappe / SVAR_MAPPE / f"{billede}.txt").write_text(tekst, encoding="utf-8")
    return mappe


def laes_koersel(mappe: Path) -> tuple[Opsaetning, dict[str, str]]:
    """Læser en gemt koersel tilbage: opsaetningen og de raa svar."""
    raa = json.loads((mappe / OPSAETNING_FIL).read_text(encoding="utf-8"))
    opsaetning = Opsaetning(
        model=raa["model"],
        promptversion=raa["promptversion"],
        prompt=raa["prompt"],
        variant=raa["variant"],
        temperatur=raa["temperatur"],
        noter=raa.get("noter", ""),
    )
    svar = {
        sti.stem: sti.read_text(encoding="utf-8")
        for sti in sorted((mappe / SVAR_MAPPE).glob("*.txt"))
    }
    return opsaetning, svar


def find_koersler(rod: Path) -> list[Path]:
    """Alle gemte koersler under `rod`, nyeste foerst."""
    if not rod.exists():
        return []
    mapper = [m for m in rod.iterdir() if (m / OPSAETNING_FIL).exists()]
    return sorted(
        mapper,
        key=lambda m: json.loads(
            (m / OPSAETNING_FIL).read_text(encoding="utf-8")
        )["dato"],
        reverse=True,
    )
