# PatientJournals research pipeline

This document describes the production transcription pipeline at the level
needed to design, reproduce, and compare research runs. Cloud batch execution is
the canonical path. The local API path is intended for development and bounded
recovery; it does not provide the complete cloud evidence-binding and immutable
dataset-version workflow described here.

Tags used throughout:

- `[REQUIRED]` — always part of the production path.
- `[OPTIONAL: setting]` — enabled, disabled, or selected independently for a job.
- `[GATE]` — a condition that must pass before the next scientific stage.
- `[FIXED POLICY]` — deliberately not selectable for new jobs.
- `[DERIVED]` — resolved from another recorded choice rather than selected
  independently.
- `[ARTIFACT]` — durable provenance or research output.
- `[OPERATIONAL: setting]` — optional throughput/storage tuning that should not
  intentionally alter transcription semantics.

## Pipeline at a glance

```text
[REQUIRED] upload final processed page bytes to GCS
    |
    +-- [OPTIONAL: ocr_enabled] prepare generation-bound OCR sidecars
    |
[REQUIRED] submit first-pass extraction batch
    |
    +-- [OPTIONAL: subagents] one schema specialist per top-level field
    |                         otherwise one request per page
    |
[REQUIRED] retrieve + deterministic/schema validation
    |
    +-- model_validation_enabled = false --> direct extraction dataset
    |
    `-- [OPTIONAL: model_validation_enabled]
          pre-version candidates + deterministic routing
              |
          [OPTIONAL: verification_scope]
              +-- flagged: risky pages + reproducible routine control sample
              `-- all: every page
              |
          second batch: configured final-authority model for selected pages
                        (blind to extraction thoughts/reasoning)
              |
          automatic sparse corrections + full-population consolidation
              |
          [GATE] one valid terminal candidate per page; no unverifiable pages
              |
          immutable v001, v002, ... dataset publication
```

The page is the primary processing unit. `image_name` is the stable dataset
identity used for coverage, continuation, duplicate handling, and retry logic.
For validation-enabled jobs, the scientific evidence for a page is the exact
processed image object and GCS generation recorded at extraction submission—not
an earlier local source file. Ordinary direct-dataset jobs do not create this
additional generation-bound evidence manifest.

## 1. Input selection and image preparation

`[REQUIRED]` The upload stage discovers images or renders PDF pages, converts
them to RGB, applies configured crop/contrast/resize operations, serializes the
result, and uploads those final bytes. OCR coordinates and all later model
evidence refer to this processed image.

Uploads are write-once: existing page objects are reused, so a later run's
preprocessing settings do not prove how an already stored object was originally
created. The stored object bytes/generation are authoritative. When several GCS
objects resolve to the same
`image_name`, the pipeline deterministically keeps the first and records the
duplicates instead of silently processing the page twice.

Important selections and transformations are:

| Setting | Default | Research effect |
|---|---:|---|
| `[OPTIONAL: upload_source]` | `images` | Choose PDF input, image input, or automatic detection. |
| `[OPTIONAL: target_folder / upload_images_folder]` | configured path | Select the local source population. |
| `[OPTIONAL: input_glob / upload_images_glob]` | `*.png` | Restrict discovered files. |
| `[OPTIONAL: recursive / upload_images_recursive]` | `true` | Include nested source folders. |
| `[OPTIONAL: fp_mode]` | `all` | Include all pages, only front pages, or exclude front pages; `fp_suffix` defines the marker. |
| `[OPTIONAL: fp_suffix]` | `_fp` | Set the filename marker used by `fp_mode`. |
| `[OPTIONAL: batch_input_prefix / batch_input_prefixes]` | empty | Select one or more cloud populations. |
| `[OPTIONAL: batch_restrict_image_names]` | empty | Restrict submission to an explicit basename set. |
| `[OPTIONAL: batch_input_extensions]` | PNG/JPEG/WebP/TIFF | Select accepted cloud image formats. |
| `[OPTIONAL: batch_year_filter]` | empty | Select years using `batch_date_mapping_file`. |
| `[OPTIONAL: batch_date_mapping_file]` | `date_mapping.csv` | Map page identities to years for `batch_year_filter`. |
| `[OPTIONAL: continue_dataset]` | empty | Skip `image_name` values already represented in a prior dataset. |
| `[OPTIONAL: downscale]` | empty | Randomly sample a fraction at CLI submission. This is exploratory only: no seed is currently persisted. |
| `[OPTIONAL: image_settings.max_dim]` | `3000` | Bound the longest processed-image dimension. |
| `[OPTIONAL: image_settings.margins]` | left `150`, others `0` | Crop pixel margins before serialization. |
| `[OPTIONAL: image_settings.contrast_factor]` | `1.1` | Adjust contrast before serialization. |
| `[OPTIONAL: image_settings.output_format]` | `PNG` | Select processed image encoding. |
| `[OPTIONAL: pdf_render_dpi]` | `300` | Set PDF rasterization resolution. |
| `[OPTIONAL: page_number_digits]` | `4` | Set deterministic rendered-page numbering width. |

`[ARTIFACT]` Duplicate-object JSONL/CSV records and the image processing
manifest preserve excluded objects and preprocessing outcomes.

## 2. Positional OCR

`[OPTIONAL: ocr_enabled]` OCR is an independent preprocessing stage. When off,
no OCR sidecar is required and no OCR tokens are sent to either extraction or
verification models. Enabling OCR does not enable subagents or model validation.

When on, Google Vision document OCR processes bounded groups of images in
parallel. The durable sidecar contains line-level text and bounding boxes
normalized to `0..1000`. Model prompts receive a compact representation:

```text
x1,y1,x2,y2|text
```

This minimizes input tokens while retaining page position. Prompts identify OCR
as fallible context; the image remains primary evidence.

Each sidecar is bound to the exact source bucket, object name, GCS generation,
size, checksums, ETag, decoded dimensions, image SHA-256, OCR backend, and OCR
document digest. A cached sidecar is reused only when that identity still
matches.

| Setting | Default | Effect |
|---|---:|---|
| `[OPTIONAL: ocr_enabled]` | `true` | Include or omit OCR as model context. |
| `[OPTIONAL: ocr_backend]` | `google_vision` | Select the OCR implementation; currently this is the supported backend. |
| `[OPTIONAL: ocr_language_hints]` | `da` | Supply OCR language hints. |
| `[OPTIONAL: ocr_required]` | `false` | Local-path policy for missing/failed OCR. |
| `[OPTIONAL: ocr_sidecar_suffix]` | `.ocr.json` | Select the per-image sidecar suffix. |
| `[OPTIONAL: batch_ocr_metadata_required]` | `true` | Make missing or stale sidecars block cloud submission. Validation-enabled jobs always enforce exact OCR bindings when OCR is on. |
| `[OPERATIONAL: batch_ocr_workers]` | `8` | Number of concurrent OCR RPC groups. |
| `[OPERATIONAL: batch_ocr_api_batch_size]` | `16` | Maximum images per provider RPC. |
| `[OPERATIONAL: batch_ocr_api_batch_max_bytes]` | `8,000,000` | Maximum encoded bytes per provider RPC. |
| `[OPERATIONAL: batch_ocr_manifest_object]` | `batch/ocr/metadata_manifest.json` | Cloud manifest location. |
| `[OPTIONAL: batch.ocr --force]` | off | Recompute valid cached sidecars. |
| `[OPTIONAL: batch.ocr --limit]` | none | Prepare only a bounded exploratory subset. |
| `[OPTIONAL: batch.ocr --allow-failures]` | off | Let preprocessing finish with failures; it does not bypass a later required-sidecar gate. |

`[ARTIFACT]` `<image-name>.ocr.json` and
`batch/ocr/metadata_manifest.json` make OCR reusable by later batch workers
without local downloads or repeated OCR calls.

## 3. First-pass extraction batch

`[REQUIRED]` Submission snapshots the effective model, prompt mapping, complete
JSON Schema and schema identity, output settings, optional-stage switches, input
population, and provider request files. Provider-independent requests are built
first and then adapted to Gemini or Anthropic batch APIs.

For model-validation-enabled runs, extraction requests are additionally pinned
to the exact manifest image generation. Gemini uses a write-once staged object
whose identity is rechecked; Anthropic receives a generation-qualified signed
URL.

### Required scientific choices

| Setting | Meaning |
|---|---|
| `[REQUIRED] model` | First-pass transcription model; it must support the selected batch backend. |
| `[REQUIRED] output_schema_name` | Extraction schema identity. |
| `[REQUIRED] input_prompt_name` | Page-level task prompt associated with the schema. |
| `[REQUIRED] dataset_source / run_mode` | Select local or cloud source and batch or secondary local execution. Production research uses cloud batch. |
| `[DERIVED] output_model` | Runtime Pydantic model resolved from `output_schema_name` plus any managed schema override; it is not a separate scientific choice. |

### Optional extraction settings

| Setting | Default | Effect or limitation |
|---|---:|---|
| `[OPTIONAL: output_format]` | `jsonl` | Select JSONL or CSV dataset representation. |
| `[OPTIONAL: csv_sep]` | `$` | Select CSV delimiter. |
| `[OPTIONAL: dataset_file_name]` | `dataset` | Set output basename. |
| `[OPTIONAL: output_schema_version_id]` | empty/built in | Select and snapshot a managed schema version. |
| `[OPTIONAL: output_schema_override]` | none | Supply and snapshot a managed JSON Schema payload. |
| `[OPTIONAL: model_temperature]` | `0.0` | Gemini/local adapter sampling temperature; current Anthropic batch extraction does not send it. |
| `[OPTIONAL: model_max_output_tokens]` | `4096` | Anthropic extraction limit; Gemini batch extraction does not currently apply this field uniformly. |
| `[OPTIONAL: thinking_level]` | `high` | Gemini/local adapter reasoning setting; current Anthropic extraction does not send it. |
| `[OPTIONAL: include_thoughts]` | `false` | Persist Gemini/local adapter thoughts when available. Thoughts are never sent to the final verifier. |
| `[OPTIONAL: include_confidence_scores]` | `false` | Request confidence information from adapters that implement it; current Anthropic extraction does not. |
| `[OPTIONAL: include_response_avg_logprobs]` | `true` | Persist average response log probability when the adapter returns it; current Anthropic extraction does not. |
| `[OPTIONAL: batch_include_response_schema]` | `true` | Send the extraction JSON Schema when supported. |
| `[OPTIONAL: response_mime_type]` | `application/json` | Set structured-response MIME type. |
| `[OPTIONAL: response_schema_field]` | `response_json_schema` | Select the provider schema parameter variant. |
| `[OPERATIONAL: batch_backend]` | `vertex` | Choose Vertex or Gemini Developer API (`mldev`) batch transport. |
| `[OPERATIONAL: batch_num_chunks / num_batches]` | `1` | Split requests into provider jobs. |
| `[OPERATIONAL: batch_input_max_bytes]` | unlimited (`0`) | Reject an oversized request artifact when nonzero. |
| `[OPERATIONAL: anthropic_signed_url_ttl_hours]` | `48` | Set signed-image URL lifetime for Anthropic batches. |

Prompt ownership is intentional: non-schema model prose lives in
`src/patientjournals/config/prompts.py`; field-level schema descriptions remain
beside their fields in `src/patientjournals/config/schemas.py`.

`[ARTIFACT]` The extraction run contains `config_snapshot.py`, `metadata.json`,
`batch_job.json`, request JSONL chunks, and provider job identifiers. For a
validation-enabled run, retrieval makes content-addressed cloud snapshots of the
completed `metadata.json` and `batch_job.json` before it writes any page
candidate; the later candidate provenance records each snapshot's SHA-256, GCS
URI, and exact object generation.

## 4. Schema-specialist decomposition

`[OPTIONAL: subagents]` When false (the default), the extraction batch contains
one request per page. When true, submission dynamically derives one specialist
per top-level `properties` entry in the selected, snapshotted JSON Schema. The
schema must therefore be an object with at least one top-level property. The
derived field list is then frozen for the run in `batch_job.json` as
`specialist_fields`; it is not replanned per page. Scalar and non-nested
top-level fields each receive a specialist, while every nested or nested-nested
descendant remains inside its nearest top-level specialist schema. No
page-specific planner or live in-process agent chooses the partition.

Every specialist receives the same processed page image, optional OCR context,
a short statement that it is a subagent, its single assignment, a compact field
brief, and a one-field schema. For multi-field schemas, the general page prompt
is omitted because the field schema and brief carry the relevant instructions;
this avoids repeating the same tokens for every specialist. A one-field schema
keeps the general page prompt. Retrieval treats missing, unknown, duplicate,
non-clean, or invalid specialist responses as page failures. It deterministically
joins only a complete, non-overlapping field set and then validates the complete
page against the original full schema.

`[DERIVED]` All specialists use the selected first-pass extraction model and its
`thinking_level`; there is no per-field model or reasoning setting. The exact
specialist and final-authority roles are recorded in `pipeline_model_roles`.
When a specialist fails, failed-page batch recovery can target only the missing
or failed specialist request rather than rerunning every field, so a recovered
page can combine successful specialist outputs from different attempts.

The Python join is not a model orchestrator. Cross-field and complexity risks
are handled by the deterministic routing stage, and selected complete pages are
then reviewed by the configured final-authority model. This keeps the 100k+ page
workflow at two asynchronous model waves instead of adding a sequential planner
and merge-model wave. `subagents=false` preserves the original full-schema,
single-request-per-page path; it creates no specialist requests or join artifact.

`[ARTIFACT]` Gemini request JSONL preserves each specialist task directly.
Anthropic request JSONL stores key, MIME type, and image source; its exact
specialist prompt/schema are reconstructed from the snapshotted configuration
and specialist request key. `[ARTIFACT] subagent_combined.jsonl` preserves
successful page joins and `[ARTIFACT] subagent_failures.jsonl` records failed
joins and retry eligibility. Both are uploaded content-addressed and create-only;
their SHA-256, URI, and GCS generation are retained with retrieval results.

## 5. Retrieval and deterministic sweep

`[REQUIRED]` Retrieval downloads output chunks in parallel and checks, in order:

1. JSONL/envelope integrity and known request identity.
2. Provider error state and nonempty response. A clean finish/stop reason is
   additionally required before creating validation candidates and during API
   recovery; the ordinary direct-dataset path retains its existing policy.
3. JSON decoding and Pydantic parsing against the snapshotted extraction schema.
   This first parse may deterministically coerce compatible scalar values, insert
   schema defaults, and serialize dates or other typed values into canonical JSON.
   For validation-enabled jobs, the canonical model dump becomes the page
   candidate. A later strict gate validates that canonical candidate; it does not
   prove that the raw provider JSON required no coercion. Raw response chunks are
   retained when that distinction matters to an analysis.
4. Complete specialist join when `[OPTIONAL: subagents]` is on.
5. Duplicate policy and expected/successful page coverage.

When `[OPTIONAL: model_validation_enabled]` is off, successful retrieval follows
the original direct-dataset path. It does not use the `vNNN` validation ledger.
When validation is on, retrieval creates unflattened page candidates and
pre-validation rows only; it must not publish `v001` or appear as a canonical
validated dataset.

`[GATE]` Before the first validation candidate is written, retrieval requires
the complete extraction `metadata.json` and `batch_job.json` and uploads their
canonical bytes to content-addressed, create-only cloud objects. `[ARTIFACT]`
Every candidate binds both snapshots by SHA-256, GCS URI, and generation. The
verifier downloads those exact generations when local recovery is needed and
refuses a digest or generation mismatch.

### Deterministic routing for the final-authority model

`[REQUIRED when model_validation_enabled=true]` Every successful page candidate
is evaluated once by policy `deterministic-routing-v1`. The router stores rule
identifiers and aggregate counts, never source values or model reasoning. It
performs strict full-schema validation and routes a page to `heavy_review` when
any of these explicit conditions occurs:

- strict schema validation fails;
- live API recovery produced the candidate;
- an empty/whitespace string is present;
- one JSON object or array contains more than 8 entries;
- the page contains more than 40 populated scalar leaves;
- one text leaf exceeds 1,600 characters;
- for the exact built-in `FrontPage` model, discharge precedes admission;
- for the exact built-in `FrontPage` model, serum presence/details disagree; or
- for the exact built-in `FrontPage` model, death and section-diagnosis presence
  disagree.

The three FrontPage semantic rules are Python-model validators and therefore do
not run for a modified managed schema, even when that schema retains the
`FrontPage` name. Managed schemas still receive every general schema, recovery,
empty-value, size, text-length, and sampling rule above.

`[FIXED POLICY]` These rule meanings and thresholds are versioned code, not
per-run tuning knobs. Changing them requires a new routing-policy version so
scientific comparisons remain interpretable.

`[OPTIONAL: verification_control_sample_percent]` Independently selects a stable
SHA-256 sample of otherwise routine pages for heavy review (default `2%`). It is
chosen at extraction-job submission and frozen when retrieval writes the routing
artifact; it cannot be changed after routing without retrieving the extraction
again. `[GATE]` Risk-routed jobs require a value greater than zero so an
all-routine population still exercises the final model; all-page scope ignores
the percentage for selection. The same key and policy seed receive the same sampling decision. For a
nonempty small population where a positive percentage happens to select zero
routine pages, the routine page with the lowest stable sample hash is included.
This control sample measures false negatives in deterministic routing.

`[ARTIFACT] deterministic_routing.jsonl` records candidate digest, route,
strict-schema result, stable ordered rule IDs, aggregate metrics, thresholds,
and sample hash/decision for every candidate. The same routing provenance is
embedded in `page_candidates.jsonl`. Both files are uploaded under
content-addressed, create-only object names, and their SHA-256 and exact GCS
generation are recorded before any final-model batch is submitted. Verifier
submission independently recomputes every decision from the frozen thresholds
and refuses any difference in candidate digest, route, rule, metric, or sample
decision.

### Optional retrieval and recovery settings

These options can change page inclusion or which model produced a recovered
page and therefore belong in experimental records.

| Setting | Default | Effect |
|---|---:|---|
| `[OPTIONAL: batch_duplicate_strategy]` | `first_successful` | Keep the first successful duplicate, or expose all duplicate responses for resolution. |
| `[OPTIONAL: batch.retrieve --duplicate-strategy]` | configured default | Override duplicate handling for one retrieval. Validation-enabled jobs require `first_successful`; `provide_all` is rejected. |
| `[OPTIONAL: require_all_expected_pages]` | `true` | Require coverage of every expected image. |
| `[OPTIONAL: require_all_pages_successful]` | `false` | Require every expected page to produce a successful row. |
| `[OPTIONAL: page_validation_sample_size]` | `5` | Limit how many missing/failed page keys are printed; coverage gates still evaluate the complete key sets. |
| `[OPTIONAL: api_recovery_enabled]` | `true` | Permit live-API recovery of a bounded missing-page set. |
| `[OPTIONAL: api_recovery_max_missing_pages]` | `50` | Maximum missing pages eligible for API recovery. |
| `[OPTIONAL: api_recovery_model]` | `gemini-3.1-pro-preview` | Model used for API recovery. |
| `[OPTIONAL: batch_submit_failed_pages]` | `false` | Submit failed pages as a new batch. |
| `[OPTIONAL: batch.retrieve --allow-partial]` | off | Retrieve terminal partial provider output. This never relaxes final validation publication gates. |
| `[OPTIONAL: batch.retrieve --recover-missing-with-api]` | off | Explicitly run bounded API recovery. |
| `[OPTIONAL: batch.retrieve --ignore-failed]` | off | Permit failed rows in the direct extraction dataset; failed pages cannot enter a validated `vNNN`. |
| `[OPTIONAL: batch.retrieve --submit-failed]` | off | Submit a failed-page retry batch. |
| `[OPTIONAL: failed_retry_num_batches]` | inherited | Split a failed-page retry into multiple jobs. |
| `[OPTIONAL: app api_recovery_threshold]` | `20` | Choose API recovery at or below the threshold and a retry batch above it. |

`[GATE]` For validation-enabled Gemini jobs, live-API recovery reuses the exact
generation-qualified image recorded for the first-pass request—normally its
immutable staged copy—not the current unqualified source object. If OCR was
enabled, recovery separately reloads the manifest-recorded OCR sidecar
generation, verifies its artifact/image/document digests and original source
identity, and then renders that bound OCR context. Missing or changed evidence
fails the page recovery. Non-validation recovery retains the ordinary current
source-object behavior.

Failed-page batch retries preserve each generated request JSONL as a
content-addressed, create-only cloud artifact and attach its URI, SHA-256, and
generation to the exact retry provider job. Aggregation resolves request bytes
by portable retry-run ID and can recover a missing local artifact from that
generation-bound cloud binding. This remains unambiguous when several retry
attempts reuse the same request filename. `[OPTIONAL: batch.submit --rerun]`
restores the original extraction and transport settings before resubmission;
for validation-enabled jobs it refuses to regenerate missing request chunks
under the existing run identity, because newly generated bytes would constitute
a different scientific request population.

`[GATE]` The application marks retrieval `candidate_incomplete` and disables
final-model submission unless every expected page has one successful candidate.
The verifier independently enforces exact equality between the complete
candidate population and extraction input manifest, so CLI submission cannot
bypass this coverage gate. Failed pages must be recovered or resubmitted first.

`require_headers_for_all_rows` and `header_validation_sample_size` are declared
configuration fields but currently have no active batch-retrieval gate. They
must not be treated as scientific validation until an implementation and test
make them effective.

`[ARTIFACT]` Retrieval writes raw output chunks, an image processing manifest
and summary, the flattened dataset or pre-validation rows, and—when validation
is enabled—`page_candidates.jsonl`.

## 6. Exact evidence bindings for final validation

`[REQUIRED when model_validation_enabled=true]` The extraction run persists:

- `[ARTIFACT] input_image_manifest.jsonl` and its metadata file, binding every
  page to image object, generation, size, checksums, and MIME type.
- `[ARTIFACT] extraction_image_bindings.jsonl` and its metadata file, binding
  each first-pass request to those image bytes.
- `[ARTIFACT] validation_bindings.jsonl`, created at verifier submission,
  separately binds each selected heavy-review candidate digest to its verified
  image/OCR evidence.
- When OCR is enabled, the manifest also binds sidecar object/generation/hash,
  OCR image hash, OCR-document hash, backend, and line count.

Validation refuses to proceed if image, OCR, full/selected candidate, schema,
routing, or request-binding identity has changed. `[GATE]` Before the immutable
policy anchor or any provider request is created, every candidate must resolve
to one portable source-run ID, that ID must match the supplied extraction run,
the source metadata and input manifest must have candidate-bound SHA-256s, and
the complete manifest/candidate populations must match. This prevents
candidates from one extraction run from being published through another run's
version ledger. A provider-specific immutable image copy is staged when the
verification API cannot consume a generation-qualified GCS reference.

## 7. Candidate-aware final verifier

`[OPTIONAL: model_validation_enabled]` This separate second model wave sends
one request per selected page. Under the default risk-routed scope, selection is
every `heavy_review` route plus the routine control sample; under all-page scope,
selection is the complete candidate population. Each request includes:

- the exact processed extraction image;
- optional generation-bound positional OCR;
- the complete original extraction schema;
- the unflattened first-pass candidate.

The configured verifier is the final authority for selected pages and is
explicitly asked to change any inaccurate field. It does not receive extraction
thoughts.
Evidence and schema are presented before the candidate to reduce anchoring. Its
structured response is one of `confirmed`, `needs_correction`, or
`unverifiable`. A correction is a minimal, non-overlapping RFC 6902 patch with a
field path, short issue/evidence, replacement value where applicable, and
optional OCR box references—not a second full transcription.

`[ARTIFACT] validation_request_contract.json` is written before any verifier
batch is submitted. It freezes the effective provider/backend, model, thinking
level, maximum output tokens, requested/effective chunk counts, and the ordered
file name, row count, byte count, and SHA-256 of every exact provider request
JSONL. Its canonical bytes are uploaded content-addressed and create-only; the
final policy binds its SHA-256, URI, and GCS generation. Submission sends those
already-frozen request bytes, and retrieval refuses publication if the local or
cloud contract, a request file, or any bound generation differs.

| Setting | Default | Effect |
|---|---:|---|
| `[OPTIONAL: model_validation_enabled]` | `false` | Add or omit the final verifier batch. |
| `[OPTIONAL: verification_model]` | `gemini-3.1-pro-preview` | Select the batch-capable final-authority model. A model different from and more capable than extraction is recommended for independent error detection, but the pipeline does not enforce that relationship. |
| `[OPTIONAL: verification_thinking_level]` | `high` | Select Low, Medium, or Maximum (`high`) reasoning for the verifier. |
| `[OPTIONAL: verification_max_output_tokens]` | `4096` | Bound the provider's combined reasoning/answer envelope; for budget-based adapters this also limits reasoning depth. |
| `[OPERATIONAL: verification_num_chunks]` | `1` | Split verifier requests across batch jobs. |
| `[OPTIONAL: verification_scope]` | `flagged` | `flagged` sends explicit risks plus the stable control sample; `all` sends every page. Both paths consolidate the complete population before publication. |
| `[OPTIONAL: verification_control_sample_percent]` | `2.0` | Percentage of otherwise routine pages sent to the final model in risk-routed mode. |
| `[FIXED POLICY] verification_apply_mode` | `apply_patches` | Corrections are accepted after the machine gates below; there is no report/apply choice for new runs. |

Thinking translation is provider-specific. Gemini models with qualitative
thinking support receive `low`, `medium`, or `high`. Gemini 2.5 uses a token
budget capped by `verification_max_output_tokens - 256`: Low requests 512,
Medium 2,048, and Maximum uses all available budget; the configured maximum must
be at least 384. Anthropic reserves 512 answer tokens: Low requests 1,024
thinking tokens, Medium 2,048, and Maximum uses the remaining budget; its
configured maximum must be at least 1,536. These mappings make both thinking
level and maximum tokens scientific cost/accuracy choices, not throughput-only
controls.

The extraction job snapshots the planned verifier model, thinking level, and
scope. `[OPTIONAL: verification_model / verification_thinking_level /
verification_scope]` The app or `batch.verify --model`, `--thinking-level`, and
`--scope` may select different values when the second batch is actually
submitted. `[FIXED POLICY]` The immutable `final_validation_policy.json` records
those effective second-wave values and is authoritative for publication. In
contrast, `verification_control_sample_percent` is frozen before extraction
retrieval because it changes deterministic routing and cannot be overridden at
verifier submission.

Historical verifier metadata may contain `report_only`; such runs remain
report-only for reproducibility. More generally, any historical run without the
independent immutable policy anchor remains report-only, even if its mutable
`batch_job.json` is later edited. Submit a new verifier run to use the current
fixed policy. A custom `--candidate-file` still requires `--source-run-dir`,
because that extraction run owns the local version mirror and cloud version
ledger.

`[ARTIFACT] final_validation_policy.json` freezes the selected scope, routing
policy and routing artifact SHA-256/URI/generation, full and selected candidate
counts/hashes, input-manifest/schema/request-binding hashes, final
provider/backend/model, thinking level, maximum output tokens, prompt hash, the
immutable exact-request contract, automatic schema-gated correction acceptance,
source/verifier run identities, and dataset/validation storage prefixes at
submission. It is created only after those local gates pass;
no failed preflight can leave a publication-authorizing anchor. Its canonical bytes are
uploaded create-only at a configuration-independent cloud anchor; retrieval
requires the local bytes, SHA-256, URI, and exact GCS generation to match that
anchor at
`_patientjournals/model_validation_policies/<verification-run-id>/final_validation_policy.json`.
This separates publication authority from the mutable operational job status
file. `[FIXED POLICY]` Each verifier run receives a globally collision-resistant,
atomically allocated portable run ID before this anchor is created, preventing
independent workers from sharing policy or publication identity.

## 8. Automatic correction and correction metadata

`[FIXED POLICY]` A correction from a new final verifier run is assumed correct
and becomes the final field value without manual approval only after all three
machine gates pass:

1. The verifier response satisfies the verifier response schema.
2. Every sparse patch applies cleanly to its exact candidate.
3. The resulting complete page satisfies the original extraction schema.

This policy records what was accepted; it is not a claim that model validation
eliminates all transcription error. The immutable audit trail is intended to
support later accuracy estimation and comparison across `vNNN` datasets.

`[ARTIFACT] field_corrections.json` (schema version 2) records one entry for
every source page and every proposed field patch. Model-reviewed pages carry
`confirmed`, `needs_correction`, or `unverifiable`; unselected routine pages
carry `deterministic_cleared`. It includes:

- verifier model/provider, prompt version/hash, and acceptance policy;
- portable source-run and verifier-run identities;
- immutable final-validation policy hash, URI, and generation;
- candidate, extraction schema, input manifest, and validation-binding hashes;
- raw verifier response, parsed-result, failure, and patched-candidate hashes;
- original, post-verifier, and corrected-dataset candidate hashes;
- JSON Pointer, operation, issue, evidence, optional OCR references, and
  original/proposed values;
- separate `proposed`, `accepted`/`applied`, and `corrected` states;
- whether a field entered the complete corrected dataset and that dataset's
  SHA-256.

The distinction matters: a page patch can pass all page-level gates and be
`accepted=true`, while a failure on another page prevents creation of the full
dataset, leaving `corrected=false`. The correction artifact is uploaded under a
content-addressed path with a create-only precondition; its URI, SHA-256, and GCS
generation are retained in the result and version ledger. It attests to the
built corrected dataset bytes and their SHA-256; `dataset_versions.json` is the
atomic authority for whether those exact bytes were published as a final
`vNNN`.

The verification run also preserves `final_validation_policy.json`, selected
`page_candidates.jsonl`, complete `source_page_candidates.jsonl` when selection
is a subset,
`input_image_manifest.jsonl`, `extraction_schema.json`, `batch_job.json`,
`metadata.json`, `validation_bindings.jsonl`, chunked request JSONL, raw response
JSONL, `validation_results.jsonl`, `validation_failures.jsonl`,
`patched_candidates.jsonl`, and `validation_summary.json`.

## 9. Publication gate and `vNNN` datasets

`[GATE]` A final validation dataset is published only when:

- the complete source candidate set exactly covers the extraction manifest;
- the selected set exactly matches the immutable routing/scope decision;
- every selected page has exactly one valid final-model result;
- every unselected page has a strict-schema-valid deterministic clearance;
- there are no missing, duplicate, failed, or selected `unverifiable` pages;
- every accepted patch has applied cleanly; and
- every resulting page has passed the original extraction schema.

If any condition fails, artifacts and accepted page-level corrections remain
auditable, but no final dataset version is created.

Successful retrieval gives the configured final-authority model final say on
selected pages, combines its confirmed/corrected candidates with unchanged
routine candidates, strictly validates the complete population, flattens it
through the existing dataset conversion path, and allocates the next immutable
version:

```text
runs/submits/<source-run>/dataset_versions/v001_model_validation.jsonl
<datasets_gcs_prefix>/<source-run>/validation_versions/v001_model_validation.jsonl
```

The suffix follows the chosen output format. `[ARTIFACT]`
`dataset_versions.json` is the shared local/cloud ledger. It records version ID,
dataset SHA-256, object URI and generation, verifier run identity, candidate and
prompt hashes, a canonical publication-provenance hash, schema/model provenance,
and the correction artifact binding.

Cloud objects use create-only generation preconditions and ledger updates use
compare-and-swap. Allocation considers both committed ledger entries and
uncommitted immutable version objects. A same-publication object left by a crash
is generation- and byte-verified and committed on retry; a foreign orphan keeps
its number reserved and the next publisher skips it without overwriting bytes.
Consequently, a ledger can have a version-number gap after an interrupted
foreign publication. Only ledger entries are final datasets. Replaying the same
verifier run reuses its committed version; the same run producing different
bytes or provenance is rejected. A genuinely new verifier run allocates the
next unreserved number. Research citations should use the version ID, SHA-256,
GCS URI, and generation. Any mutable `current.*` path is only an application
convenience.

## 10. Operational settings

The following options should be recorded for performance reproducibility, but
are not intended to change transcription meaning:

| Settings | Purpose |
|---|---|
| `[OPERATIONAL: batch_size, flush_every]` | Local buffering and artifact flush frequency. |
| `[OPERATIONAL: api_concurrent_tasks]` | Concurrent live-API recovery calls. |
| `[OPERATIONAL: api_max_attempts, api_retry_initial_delay_seconds, api_retry_max_delay_seconds, api_retry_jitter_seconds]` | API retry policy. |
| `[OPERATIONAL: upload_auto_tune, upload_profile]` | Select automatic/light/normal/aggressive upload tuning. |
| `[OPERATIONAL: upload_max_workers, upload_workers, batch_upload_limit]` | Upload concurrency and grouping. |
| `[OPERATIONAL: upload_timeout_seconds, upload_retry_attempts, upload_retry_initial_delay_seconds, upload_retry_max_delay_seconds]` | Upload timeout/retry policy. |
| `[OPERATIONAL: batch_poll_interval_seconds]` | Status polling interval. |
| `[OPERATIONAL: output_root]` | Local root for run, verification, and dataset artifacts (`local_runs_root` in the app). |
| `[OPERATIONAL: batch_job_display_name]` | Human-readable provider job label. |
| `[OPERATIONAL: batch_job_name]` | Explicit provider job ID used by status/retrieval when no run directory is supplied. |
| `[OPERATIONAL: batch_requests_file_name]` | Base name of extraction request JSONL artifacts. |
| `[FIXED POLICY] batch_input_source` | `gcs`; production batch request construction consumes cloud inputs. |
| `[OPTIONAL: batch_use_local_pdf_folders]` | Permit PDF-folder discovery during batch input preparation; this can change the selected population. |
| `[OPTIONAL: batch_auto_upload_missing]` | Upload selected local images absent from the cloud prefix; this can change the selected population. |
| `[OPERATIONAL: gcs_pages_prefix, batch_requests_gcs_prefix, batch_outputs_gcs_prefix, datasets_gcs_prefix, validations_gcs_prefix, schemas_gcs_prefix]` | Cloud artifact namespaces. |
| `[OPERATIONAL: upload_dataset_to_gcs]` | Upload direct-path datasets to cloud storage. Model-validation artifacts and immutable versions are always cloud-backed. |
| `[OPTIONAL: upload_validation_to_gcs]` | Upload browser/manual validation sync artifacts; it does not disable the mandatory cloud backing of final model-verification artifacts. |
| `[OPERATIONAL: gcp_auth_mode, service_account_file]` | Select ADC or service-account authentication and its credential file. Credential contents are never research metadata. |
| `[OPERATIONAL: gcp_project_id, gcp_location, vertex_model_location]` | Select the cloud project and provider regions. |
| `[OPERATIONAL: gcs_bucket_name]` | Select the cloud artifact bucket. |
| `[OPERATIONAL: provider_api_keys / api_key]` | Provider credentials and the legacy Gemini-key alias; secret values must not be written into research artifacts. |

These deployment settings do not alter the intended transcription logic. Safe
identifiers are snapshotted where needed to locate and verify durable artifacts;
credential secrets are not.

## Canonical command sequence

```bash
uv run invoke batch.upload
uv run invoke batch.ocr                         # only if ocr_enabled=true
uv run invoke batch.submit
uv run invoke batch.status --watch
uv run invoke batch.retrieve --wait

# only if model_validation_enabled=true
uv run invoke batch.verify --source-run-dir runs/submits/<run>
uv run invoke batch.status --run-dir runs/verifications/<run> --watch
uv run invoke batch.verify --retrieve --run-dir runs/verifications/<run> --wait
```

The application submits/retrieves the same extraction and verification jobs and
stores the three independent stage choices—OCR, schema specialists, and final
model validation—with each job. It currently preflights prepared OCR sidecars
but does not run `batch.ocr`; cloud OCR preparation remains a separate command.
Persistent app settings only provide defaults. The extraction snapshot is
authoritative for first-pass semantics and deterministic routing; when a second
model wave is submitted, its immutable final-validation policy is authoritative
for the effective verifier model, thinking level, and scope.
