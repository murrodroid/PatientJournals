# AGENTS.md — arbejdsregler for 2ndpage

Dette projekt følger **ICM (Interpretable Context Methodology)**: folderstrukturen
er den primære agent-arkitektur. Metodikken stammer fra `magresprot_xmltools` —
se `references/icm_metodik.md` for den fulde beskrivelse.

## Læserækkefølge

1. Læs rod-`CONTEXT.md` for routing, principper og trufne beslutninger.
2. Læs `PROGRESS.md` for hvad der er gjort, og hvad der er næste skridt.
3. Læs den aktuelle stages `stages/<NN>_<navn>/CONTEXT.md`.
4. Læs kun de referencefiler, den stage-kontrakt udpeger. Ikke mere.

## Regler

- **Ny funktionalitet begynder med en test eller en udtrykkelig testkontrakt** —
  men kun for deterministisk logik (parsere, billedbehandling, målekode).
  Modelforsøg logges i stedet som resultater i stagens `output/`.
  Se `_config/tdd.md`.
- **Stage-output er menneskelæsbare filer** i stagens `output/`: CSV, JSONL,
  Markdown, PNG. Plain text først.
- **Videre til næste stage kun efter menneskelig gennemgang** af forrige stages
  output. Agenten må ikke selv erklære en stage færdig.
- **Ingen fulde kørsler uden go.** Pilot på få sider → gennemgang → brugeren
  beslutter skalering. Det gælder både API-kald og billedbehandling.
- **Alt med eksterne bivirkninger er dry-run som standard**; `--yes` udfører.
- Enhver rettelse skal være et generelt mønster, ikke en hardkodet undtagelse
  for et bestemt billede eller en bestemt patient.

## Hvad projektet leverer

Til sidst afleveres til kollegaens app i `murrodroid/PatientJournals`:
en **prompt**, et **skema** og et **bevis** for at det virker (måletal mod
facit). Ikke kørselskode — hans pakke har allerede klienter og batch-spor.

## Sprog

Dokumentation, kontekstfiler og commit-beskeder skrives på dansk.
Kode, variabelnavne og bibliotekskald står på engelsk.
