# ICM-metodikken (fra magresprot_xmltools) + genbrugeligt baseline-maskineri

Kortlagt 2026-08-18 af udforsknings-agent i `c:\Work\magresprot`. OBS: den lokale
klon var 42 commits bag `origin/main`; baseline-koden blev læst via `git show
origin/main:<sti>` uden at røre arbejdstræet.

## 1. ICM — Interpretable Context Methodology

Navnet står i magresprots `AGENTS.md:3`: *"Interpretable Context Methodology
(ICM): folderstrukturen er den primære agent-arkitektur"* (også `README.md:17`).

### Mappelayout

| Element | Rolle |
|---|---|
| `CLAUDE.md` (rod) | Én linje: `@AGENTS.md` — AGENTS.md er eneste kilde |
| `AGENTS.md` | Routing-regler for agenter (workflow, se nedenfor) |
| `CONTEXT.md` (rod) | Projektbrede principper + domænefakta + links til hver stages CONTEXT.md |
| `_config/tdd.md` | TDD-regelbog (testniveauer, Definition of Done); input for alle stages |
| `stages/<NN>_<navn>/` | Én mappe pr. nummereret stage med egen `CONTEXT.md` og `output/` |
| `stages/<NN>_<navn>/proposals/`, `references/` | Valgfri stage-lokal research/reference |
| `src/<pakke>/` | Selve Python-pakken |
| `tests/` | Flad, én fil pr. modul/emne + `test_icm_structure.py` der håndhæver metodikken |
| `references/` (rod) | Repo-brede referencedokumenter |
| `work/` | Scratch/research der ikke er stage-leverance (selektivt git-tracket) |

### Den håndhævede seks-afsnits stage-kontrakt

`tests/test_icm_structure.py:19-27` kræver at hver `stages/<stage>/CONTEXT.md`
indeholder disse overskrifter ordret, og at `stages/<stage>/output/` findes:

```python
required_sections = ["## Formål", "## Inputs", "## Process", "## Outputs", "## Test Contract", "## Handoff"]
```

Indholdskonvention (observeret på tværs af alle fire stages):
- **Formål** — ét afsnit.
- **Inputs** — tabel `Type | Sti`; altid inkl. `../../_config/tdd.md` + forrige stages outputs.
- **Process** — nummererede trin; dry-run-først og usikkerhedsflagning nævnes.
- **Outputs** — tabel `Fil | Beskrivelse`; filnavne matcher det der lander i `output/`.
- **Test Contract** — hvad der SKAL testes uafhængigt af model/ekstern tjeneste
  (fx "Modelintegration må ikke være eneste verifikation").
- **Handoff** — én sætning: næste stage + hvad "reviewed" betyder før videre.

Andre strukturtests: `test_icm_root_context_files_exist` (rod-`CLAUDE.md`,
`CONTEXT.md`, `_config/tdd.md` skal findes); `test_stage_order_is_numeric_and_unique`
(stage-mappenavne skal matche en hardcodet `STAGES`-liste — ny stage = opdatér testen).

### Workflow-regler (`AGENTS.md:7-15`)

1. Læs rod-`CONTEXT.md` for routing → læs aktuel stages `CONTEXT.md` → læs kun
   de referencer, stage-kontrakten nævner.
2. Ny funktionalitet starter med test/test-kontrakt.
3. Stage-output skrives som læsbare filer i stagens `output/` ("Plain text
   først": CSV/JSON/Markdown/PAGE XML).
4. Videre til næste stage KUN efter menneskelig review af forrige stages output.

### CLI-konvention

`pyproject.toml [project.scripts]` → ét CLI-entrypoint med argparse-subkommandoer,
én pr. pipeline-handling; alt med eksterne bivirkninger er dry-run-by-default med
`--yes` for at udføre (dokumenteret i `_config/tdd.md:16-17`).

### Bootstrap-tjekliste for nyt projekt

Rod-`CLAUDE.md` (`@AGENTS.md`) + `AGENTS.md` + rod-`CONTEXT.md` + `_config/tdd.md`
+ `stages/01_.../CONTEXT.md` med alle seks overskrifter + `output/`-mappe +
tilpasset `tests/test_icm_structure.py` + `src/<pkg>/` + `pyproject.toml` med
CLI-entrypoint.

## 2. Baseline/HTR-maskineri (kun på `origin/main`, ikke lokalt)

Alt er en trimmet vendor-kopi af kollega-repoet
`CopenhagenCityArchives/python-yolo-segmentation` (commit `16cb5d3`, senere
`f38ca6f`) — dokumenteret i `stages/03_baselines/references/baseline_contract.md`.
Upstreams klyngning/deskew/region-syntese ("regions-from-lines") blev bevidst
IKKE vendoret ind i magresprot ("vi har allerede regionerne").

**Model**: `Riksarkivet/yolov9-lines-within-regions-1` (HuggingFace Hub, lazy
download via `huggingface_hub.hf_hub_download`). Ultralytics `YOLO` importeres
lazy, så modulet kan importeres uden torch/ultralytics.

**Tre moduler** (læses fra magresprots `origin/main`):
1. `src/magresprot_xmltools/baselines_detect.py` — detektor.
   `detect_region_lines(image_bgr, region_polygon, cfg, *, segmenter=None)`:
   region-crop (maskerer udenfor hvidt) → `YoloSegmenter.predict` →
   polygon-simplificering (cv2 `approxPolyDP`) → tilbage til sidekoordinater →
   baseline pr. linjepolygon (modes `underline`/`strike`) → sortér top→bund.
   `BaselineDetectConfig`: `lines_repo`, `conf=0.25`, `imgsz`, `baseline_mode`,
   `min_confidence`. Injectable segmenter = testbar uden model.
2. `src/magresprot_xmltools/baselines.py` — orkestrator `run_stage(...)`: går
   PAGE XML igennem pr. side, cropper, kalder detektoren, skriver
   `baselines.jsonl` (pr. region: `region_id`, `lines[]` med
   polygon/baseline/confidence, `flags`).
3. `src/magresprot_xmltools/pagexml_stage03.py` — injicerer
   `<TextLine><Coords/><Baseline/></TextLine>` i eksisterende PAGE XML.

**CLI**: `magresprot baselines` med `--detect-only`/`--xml-only` (de to trin kan
køres hver for sig), `--conf`, `--imgsz`, `--region-min-confidence`.

**Transkribus-integration**: `src/magresprot_xmltools/transkribus.py` — Legacy
REST (`account.readcoop.eu` OIDC, `client_id=transkribus-api-client`); endpoints
i `references/transkribus.md:84-99`; POST PAGE XML til eksisterende sider via
`/collections/{col}/{doc}/{page}/text` (aldrig re-upload). Metagrapho/Processing
API kun dokumenteret, ikke brugt.

**Miljø**: uv-projekt; `origin/main` tilføjer `ultralytics>=8.4.90`,
`huggingface-hub>=1.22.0`, `shapely>=2.1.2` + research-gruppe med
`opencv-python-headless`.

## Faldgruber for genbrug i 2ndpage

- Magresprot afhænger af `pagexml-tools` som lokal sti-dependency
  (`../pagexml_tools/pagexmltools`) — findes IKKE på denne maskine (kun g129).
  Genbrug af de tre baseline-moduler bør kopiere dem fri af den dependency.
- Uafklaret om fuldside-detektion (uden foruddefinerede regioner) er landet i
  magresprots `baselines_detect.py` eller stadig kun findes i kollega-repoet —
  for 2ndpage (ingen regioner på forhånd) er kollega-repoets fuldside-pipeline
  formentlig det rigtige udgangspunkt.
- Remote-grene `origin/stage03-baseline-eval` og `origin/stage04-text-to-baselines`
  ikke undersøgt.
