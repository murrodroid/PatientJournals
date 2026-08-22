"""Tegn- og ordfejl med de fem varianter.

Overtaget fra `J-Hoffi/StadsCER` (`stadscer/cer.py`, tilgaaet 2026-08-21) som
stage 03's kontrakt foreskriver: koden er allerede dansk-tilpasset og i brug
paa 32 korrekturlaeste hospitalsprotokolsider. Aendringer i forhold til kilden:

- `compare()` og `compare_all_variants()` er IKKE overtaget. De matcher linjer
  paa id, og hos os har model og facit ikke faelles linje-id'er. Vores egen
  parring ligger i `maal.py`.
- `Metrics` hedder `Maaltal` og kan laegges sammen, saa sider kan rulles op
  til et samlet tal uden at gennemsnitte rater.
- Docstrings er skrevet om til dansk efter projektets sprogregel.

**CER** = samlet Levenshtein-afstand paa tegn delt med antal tegn i facit.
**WER** = det samme paa ord. WER er strengere -- et enkelt forkert bogstav
goer hele ordet forkert -- men taettere paa, hvad en soegning oplever.

Begge laegges sammen over alt materiale under ét, ikke som gennemsnit af de
enkelte linjers rater. Ellers vejer en linje paa tre tegn lige saa tungt som
en paa firs.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Tegnsaetning som forekommer i protokollerne. Bevidst en fast liste frem for
# unicodedata-kategorier, saa resultatet ikke skifter med Python-version.
PUNCTUATION = set(".,;:!?()[]{}«»\"'`´-–—/\\*†‡§¶&%")

# De tyske former er samme bogstav som de danske i dette materiale. Skriverne
# veksler mellem dem, og modellen goer det samme. Maalt over StadsCERs 50
# sider er ø/ö den hyppigste enkeltforveksling overhovedet (131 tilfaelde) --
# ren ortografisk variation, ikke en laesefejl.
TYSKE_FORMER = {
    "ä": "æ", "Ä": "Æ",
    "ö": "ø", "Ö": "Ø",
    "ü": "y", "Ü": "Y",
}


def strip_diacritics(text: str) -> str:
    """Fjerner accenter, men bevarer æ, ø og å som selvstaendige bogstaver --
    de er ikke diakritika paa dansk, de er bogstaver i alfabetet.

    De tyske omlyde foldes til deres danske modstykker (ö->ø, ä->æ, ü->y) frem
    for at blive strippet til o/a/u. Ellers ville netop den variation, filteret
    er til for at se bort fra, taelle som en laesefejl.
    """
    keep = {"æ", "ø", "å", "Æ", "Ø", "Å"}
    out = []
    for char in text:
        if char in keep:
            out.append(char)
            continue
        if char in TYSKE_FORMER:
            out.append(TYSKE_FORMER[char])
            continue
        decomposed = unicodedata.normalize("NFD", char)
        out.append("".join(c for c in decomposed if not unicodedata.combining(c)))
    return unicodedata.normalize("NFC", "".join(out))


def normalize(
    text: str,
    ignore_case: bool = False,
    ignore_diacritics: bool = False,
    ignore_punctuation: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    if collapse_whitespace:
        text = " ".join(text.split())
    if ignore_case:
        text = text.lower()
    if ignore_diacritics:
        text = strip_diacritics(text)
    if ignore_punctuation:
        text = "".join(c for c in text if c not in PUNCTUATION)
        text = " ".join(text.split())
    return text


def levenshtein(a, b) -> int:
    """Redigeringsafstand. Virker paa baade strenge (tegn) og lister (ord)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def align(a, b) -> list[tuple[str, int, int, int, int]]:
    """Minimal redigeringssekvens som opcodes, samme format som difflib:
    `(tag, i1, i2, j1, j2)` med tags equal/replace/delete/insert.

    Findes for at fejlanalysen kan bygge paa PRAECIS den samme opdeling, som
    `levenshtein()` taeller. `difflib.SequenceMatcher` maksimerer sammen-
    haengende matchblokke og garanterer ikke minimal afstand -- den kan derfor
    rapportere flere operationer, end CER-tallet er beregnet ud fra.

    Ved lige billige veje vaelges ALTID i denne raekkefoelge: diagonal ->
    sletning -> indsaettelse. Den faste raekkefoelge er ikke pynt: uden den
    kunne to koersler af uaendret kode give hver sin fejlrapport.

    En `replace`-blok har altid samme laengde paa begge sider, fordi hver
    substitution forbruger ét element fra hver. Kaldere kan regne med det.
    """
    n, m = len(a), len(b)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )

    ops: list[tuple[str, int, int, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            tag = "equal" if a[i - 1] == b[j - 1] else "replace"
            ops.append((tag, i - 1, i, j - 1, j))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(("delete", i - 1, i, j, j))
            i -= 1
        else:
            ops.append(("insert", i, i, j - 1, j))
            j -= 1
    ops.reverse()

    merged: list[tuple[str, int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in ops:
        if merged and merged[-1][0] == tag:
            prev = merged[-1]
            merged[-1] = (tag, prev[1], i2, prev[3], j2)
        else:
            merged.append((tag, i1, i2, j1, j2))
    return merged


@dataclass(frozen=True)
class Maaltal:
    """Tegn- og ordfejl for ét udsnit af materialet under én variant.

    Kan laegges sammen, saa linjer ruller op til sider og sider til hele
    saettet, uden at der gennemsnittes over rater undervejs.
    """

    tegnafstand: int
    facit_tegn: int
    ordafstand: int
    facit_ord: int
    stykker_maalt: int
    stykker_identiske: int

    @property
    def cer(self) -> float:
        return self.tegnafstand / self.facit_tegn if self.facit_tegn else 0.0

    @property
    def wer(self) -> float:
        return self.ordafstand / self.facit_ord if self.facit_ord else 0.0

    @property
    def andel_identiske(self) -> float:
        return self.stykker_identiske / self.stykker_maalt if self.stykker_maalt else 0.0

    def __add__(self, other: "Maaltal") -> "Maaltal":
        return Maaltal(
            self.tegnafstand + other.tegnafstand,
            self.facit_tegn + other.facit_tegn,
            self.ordafstand + other.ordafstand,
            self.facit_ord + other.facit_ord,
            self.stykker_maalt + other.stykker_maalt,
            self.stykker_identiske + other.stykker_identiske,
        )


NUL = Maaltal(0, 0, 0, 0, 0, 0)


# Varianterne der rapporteres. Alle staar side om side; ingen af dem maa
# vaelges efter, hvilken der klaeder resultatet bedst (stage 03, punkt 1).
# `raa` er leverancens tal. `arbejdstal` er beslutning 26 -- uden versaler OG
# uden tegnsaetning -- og staar med i tabellen frem for at blive regnet ud i
# hovedet af de to enkeltfiltre, som ikke kan laegges sammen.
VARIANTER: dict[str, dict] = {
    "raa": {},
    "uden_versaler": {"ignore_case": True},
    "uden_diakritika": {"ignore_diacritics": True},
    "uden_tegnsaetning": {"ignore_punctuation": True},
    "arbejdstal": {"ignore_case": True, "ignore_punctuation": True},
    "lempeligst": {
        "ignore_case": True,
        "ignore_diacritics": True,
        "ignore_punctuation": True,
    },
}


def maal_par(facit: str, model: str, **options) -> Maaltal:
    """Tegn- og ordfejl mellem ét facit-stykke og modellens modstykke."""
    ref = normalize(facit, **options)
    hyp = normalize(model, **options)
    ref_ord, hyp_ord = ref.split(), hyp.split()
    return Maaltal(
        tegnafstand=levenshtein(ref, hyp),
        facit_tegn=len(ref),
        ordafstand=levenshtein(ref_ord, hyp_ord),
        facit_ord=len(ref_ord),
        stykker_maalt=1,
        stykker_identiske=int(ref == hyp),
    )
