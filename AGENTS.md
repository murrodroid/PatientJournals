## Batch-first architecture

Cloud batch jobs are the primary production path for this project. Design and implement features against the batch workflow first:

1. `uv run invoke batch.upload`
2. When `ocr_enabled=True`, optionally prepare the full population eagerly with
   `uv run invoke batch.ocr`; `batch.submit` must automatically prepare any
   missing or stale OCR sidecars in its exact selected cohort before submission
3. `uv run invoke batch.submit`
4. `uv run invoke batch.status --watch`
5. `uv run invoke batch.retrieve --wait`
6. When the submitted job has `model_validation_enabled=True`, treat retrieval as
   a pre-version candidate, then run:
   - `uv run invoke batch.verify --source-run-dir runs/submits/<run>`
   - `uv run invoke batch.status --run-dir runs/verifications/<run> --watch`
   - `uv run invoke batch.verify --retrieve --run-dir runs/verifications/<run> --wait`

Rules:
- A feature is not complete if it only works through `patientjournals.local`. Local generation is a secondary development and recovery path.
- Durable preprocessing output needed by a batch request must live in cloud
  storage. The automatic OCR preflight is an orchestration stage before request
  construction: it may fetch exact selected image generations only for missing
  or stale sidecars, must persist the resulting sidecars and run-scoped
  preflight manifest in cloud storage, and must complete before provider request
  files are built.
- Treat OCR, schema-specialist subagents, and second-pass model verification as
  independent per-job options. The app must expose all three during submission,
  persist their defaults separately, and snapshot the selected values in the
  job's immutable run metadata. Do not infer that enabling one enables another.
- When `ocr_enabled=True`, submission must resolve its final cohort first and
  automatically prepare missing or stale OCR metadata for exactly those pages.
  Valid cached sidecars must be reused, sidecars must be bound to the exact GCS
  object generation, and required-sidecar failures must block extraction before
  any provider batch is submitted. `batch.ocr` remains the eager bulk-preparation
  command. When OCR is false, submission must not require or prepare a sidecar.
- Cloud preprocessing should use bounded provider-native batches and parallelism rather than one remote call per input. Keep batch sizes and concurrency configurable so production runs can stay within provider quotas.
- New operational commands and tests should preserve the sequence upload ->
  cloud preprocessing/preflight -> submit -> retrieve.
- `model_validation_enabled=False` must preserve the original extraction and
  dataset-publication path. When it is enabled, extraction retrieval must not
  create `v001` or place a candidate in the canonical cloud dataset library.
  Candidate pages, image/OCR bindings, requests, responses, patches, failures,
  and summaries must be durable cloud-backed batch artifacts.
- `v001` is the publication boundary, not the raw extraction boundary. Before it
  is created, every expected page must have passed deterministic response and
  strict original-schema validation of the canonical page candidate. Provider
  JSON is first parsed through Pydantic and may be deterministically coerced or
  default-filled before that canonical candidate is written; do not describe the
  later strict gate as proof that the raw provider response required no coercion.
  The versioned deterministic router selects complex/risky pages plus a
  reproducible routine control sample for the configured final-authority model;
  an all-page scope remains selectable. New final-stage verifier runs
  automatically accept sparse corrections only after they apply
  cleanly and the corrected page revalidates against the original extraction
  schema. Publication requires one valid verifier result per selected page, a
  strict deterministic clearance for every unselected page, and no missing,
  failed, duplicate, or selected `unverifiable` pages.
- New verifier submissions must write a create-only, generation-bound
  `final_validation_policy.json` at the fixed policy-anchor namespace. That
  contract—not mutable job-status metadata—is authoritative for the selected
  routing/scope population, automatic acceptance, source/verifier identities,
  effective provider/backend/model/thinking/token settings, exact request-file
  contract, and snapshotted dataset and validation storage prefixes. Generate
  and hash every provider request JSONL before creating the policy, upload the
  request contract content-addressed and create-only, submit those frozen bytes,
  and fail retrieval on any request-contract digest or generation drift.
  Unanchored historical runs are report-only.
  Custom candidate-file submission must still identify its source extraction
  run so a successful verifier has exactly one version ledger.
- Before creating the final-validation policy anchor or submitting any provider
  request, require every candidate to resolve to one portable source-run ID and
  require it to match the supplied source extraction run. Reject missing,
  conflicting, mixed-run, or mismatched provenance; candidates from one run
  must never publish through another run's version ledger.
- Allocate verifier run directories atomically with globally collision-resistant
  portable IDs. A locally unique timestamp or numeric suffix is insufficient
  because the ID also names cloud policy anchors and publication provenance.
- Bind second-pass requests to the extraction input manifest. Refuse validation
  when an image generation, checksum, candidate digest, schema digest, request
  binding, or enabled OCR binding no longer matches. Verification must remain
  usable without OCR and must still consume the exact extraction image bytes.
  Stage immutable verifier image objects when a provider cannot consume a
  generation-qualified reference.
- Verifier outputs are sparse field patches plus short evidence, not a second
  full transcription. Do not pass extraction-model thoughts or hidden reasoning
  to the verifier; retain reproducible provenance and concise evidence only. An
  inaccurate field must produce a minimal corrective patch, not only a warning.
  Every retrieval must write and upload an immutable, content-addressed
  `field_corrections.json`, with the page, field path, original and proposed
  value, evidence, acceptance/application state, inclusion in the complete
  corrected dataset bytes, corrected dataset digest, object generation, and
  artifact SHA-256. Treat `dataset_versions.json` as the publication authority. Keep
  `accepted` distinct from `corrected`: a patch can be schema-valid and accepted
  even when another page blocks publication of the complete dataset.
- Never put machine-local verifier paths in hashed correction/candidate
  artifacts. Use portable run IDs so moving a recovered run does not change
  publication provenance.
- Dataset version records must point to immutable `vNNN_*` files. A mutable
  `current.*` path may be maintained separately for app convenience.
- Version allocation belongs to the shared batch publication layer, not only to
  app state. Direct `batch.verify --retrieve` and app retrieval must consume the
  same replay-safe `dataset_versions.json` ledger, immutable local `vNNN` file,
  and generation-bound GCS `vNNN` object. Replaying one verifier run must reuse
  its version; a genuinely new verifier run may allocate the next version.
  Allocation must consider uncommitted create-only cloud objects as reservations:
  reconcile an exact same-publication orphan through ledger compare-and-swap and
  skip a foreign orphan without overwriting it.
- When model validation is enabled, also pin the first-pass extraction request
  to the input manifest's image generation. Gemini requests use a write-once
  staged copy whose identity is rechecked before candidate creation; Anthropic
  signed URLs include the GCS generation. A non-clean provider finish/stop
  reason must not become a pre-version candidate even when its JSON validates.
- Before writing validation candidates, snapshot the completed extraction
  `metadata.json` and `batch_job.json` to content-addressed, create-only cloud
  objects. Bind their SHA-256, GCS URI, and exact generation into every candidate;
  recovery and verifier submission must fail closed on any digest or generation
  mismatch.
- Failed-page retry jobs must bind each request file directly to its
  content-addressed cloud URI, SHA-256, and GCS generation. Root aggregation
  must distinguish retry attempts by portable retry-run identity even when file
  names repeat, and must verify local bytes or download the exact bound cloud
  generation. `batch.submit --rerun` must restore the original scientific and
  transport settings; never regenerate missing validation-enabled request
  chunks under an existing run identity.
- Prefer provider-independent request preparation before the provider-specific Gemini or Anthropic batch adapters.
- Agent-based transcription is also batch-first. `subagents=False` must preserve the
  single-request-per-page pipeline. `subagents=True` fans each page out into one
  request per top-level schema field. Derive that field list once from the
  selected schema's top-level `properties`, require at least one property, and
  freeze the exact list in run metadata; do not replan it per page. Nested
  descendants stay inside that top-level specialist; scalar top-level fields
  receive specialists too. Every specialist uses the same extraction model and
  extraction thinking level; there is no per-field model/effort selection.
  Retrieval must validate every specialist, join a complete page, validate the
  full schema, run deterministic risk routing, and only then send the selected
  complete pages to the configured final-authority model. Failed-page batch
  recovery may target only failed or missing specialist requests. Request files
  and provider outputs participate in the cloud batch path; joined/failure,
  candidate, and routing JSONL files must be content-addressed, create-only,
  SHA- and generation-bound cloud artifacts. Do not add an in-process planner or
  another sequential merge-model wave to the production cloud batch path.
- Do not claim that the configured final-authority model is necessarily larger,
  different, or independent from the extraction model; that is recommended
  experimental design, not an enforced gate. Verifier model, thinking level, and
  scope may be changed when the second batch is submitted, and the immutable
  final-validation policy records the effective values. The control-sample
  percentage is different: it is frozen before routing and cannot be changed at
  verifier submission.
- Document verifier thinking as provider-specific. Qualitative Gemini models
  receive low/medium/high. Gemini 2.5 budgets Low=512, Medium=2,048, and Maximum
  as `verification_max_output_tokens - 256` (capped by availability; minimum
  maximum 384). Anthropic budgets Low=1,024, Medium=2,048, and Maximum as
  `verification_max_output_tokens - 512` (minimum maximum 1,536). Therefore the
  maximum-token setting affects both reasoning and answer space.

## Pipeline documentation

`PIPELINE.md` is the canonical researcher-facing description of the production
pipeline. Update it in the same change whenever a pipeline stage, job option,
default, model or prompt input, validation/publication gate, evidence binding,
artifact, recovery rule, or versioning behavior changes. Keep it focused on
scientifically relevant behavior and reproducibility rather than module-level
implementation detail. Mark every optional setting with `[OPTIONAL: setting]`,
fixed behavior with `[FIXED POLICY]`, mandatory checks with `[GATE]`, durable
outputs with `[ARTIFACT]`, and throughput-only controls with
`[OPERATIONAL: setting]`. Use `[DERIVED]` for values resolved from another
recorded choice rather than independently selected.

## Prompt ownership

- Keep every non-schema model instruction in `src/patientjournals/config/prompts.py`.
  This includes page prompts, sub-agent role/context, and OCR evidence instructions.
- Keep JSON Schema `Field(description=...)` prompts beside their fields in
  `src/patientjournals/config/schemas.py`.
- Provider adapters and request builders may assemble prompts, but must not embed
  new model-facing prose.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
