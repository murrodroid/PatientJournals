# Graph Report - PatientJournals  (2026-08-27)

## Corpus Check
- 97 files · ~114,963 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1652 nodes · 4826 edges · 66 communities (55 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 241 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `778bb912`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- JobStore
- tools.py
- WorkflowService
- retry.py
- config/schemas.py
- PatientJournalsApp
- dashboard.py
- upload.py
- jobs.py
- status.py
- test_app_architecture.py
- PatientJournals Conda Environment
- workflows.py
- submit.py
- retrieve_batch
- ui.py
- validation/cli.py
- inspection.py
- ValidatorApp
- process_file
- resolve_model_spec
- patientjournals/tasks.py
- response_parsing.py
- model_client.py
- BrowserValidationSession
- generate.py
- test_subagents.py
- ocr.py
- submit_requests.py
- ImageAccessService
- Journal
- .name
- ocr_context.py
- test_ocr.py
- resolve_batch_run_readiness
- preprocess.py
- .__init__
- OcrDocument
- Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?
- JobRegistry
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- Path
- get_batch_client
- processing_metrics.py
- Journal
- test_validation_sampling.py
- prepare_ocr.py
- patientjournals/__init__.py
- local/__init__.py
- patientjournals
- patientjournals.app.access
- patientjournals.app.catalog
- patientjournals.app.datasets
- patientjournals.app.image_access
- patientjournals.app.settings_store
- retrieve.py
- collect_outputs.py
- read_dataset_preview
- run_layout.py
- FakeBlob
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.
- image_name_from_reference
- recover_dataset_gaps
- Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage.

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 64 edges
2. `AppSettings` - 57 edges
3. `WorkflowService` - 54 edges
4. `PatientJournalsApp` - 44 edges
5. `retrieve_batch()` - 44 edges
6. `submit_batch()` - 40 edges
7. `serializable()` - 33 edges
8. `image_name_from_reference()` - 32 edges
9. `build_storage_bucket()` - 30 edges
10. `collect_outputs()` - 29 edges

## Surprising Connections (you probably didn't know these)
- `test_job_store_persists_background_tasks()` --uses--> `JobStore`  [INFERRED]
  tests/test_app_workflows.py → src/patientjournals/app/job_store.py
- `test_recover_dataset_gaps_only_targets_missing_pages()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_recover_dataset_gaps_reports_zero_row_api_completion()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_failed_page_retry_can_split_into_multiple_chunks()` --calls--> `_submit_failed_pages_as_batch()`  [INFERRED]
  tests/test_batch_retrieve_recovery.py → src/patientjournals/batch/retry.py
- `test_batch_retrieve_request_namespace()` --uses--> `BatchRetrieveRequest`  [INFERRED]
  tests/test_batch_service.py → src/patientjournals/batch/service.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Local Model Provider Support** — readme_local_generation, readme_gemini, readme_openai, readme_anthropic [EXTRACTED 1.00]
- **Web App Service Architecture** — readme_patientjournals_app_access, readme_patientjournals_app_catalog, readme_patientjournals_app_dashboard, readme_patientjournals_app_datasets, readme_patientjournals_app_image_access, readme_patientjournals_app_job_store, readme_patientjournals_app_jobs, readme_patientjournals_app_schemas, readme_patientjournals_app_settings_store, readme_patientjournals_app_task_runner, readme_patientjournals_app_web, readme_patientjournals_app_workflows_workflowservice [EXTRACTED 1.00]
- **Journal Record Model** — visualizations_patientjournals_journal, visualizations_patientjournals_hospital_stay, visualizations_patientjournals_patient, visualizations_patientjournals_diagnoses, visualizations_patientjournals_serum [EXTRACTED 1.00]
- **Quality Assurance Flow** — visualizations_patientjournals_dataset, visualizations_patientjournals_validators, visualizations_patientjournals_validations, visualizations_patientjournals_accuracy [EXTRACTED 1.00]
- **Diagnosis Structure** — visualizations_patientjournals_schema_diagnoses, visualizations_patientjournals_schema_bottom, visualizations_patientjournals_schema_section [EXTRACTED 1.00]
- **Journal Composition** — visualizations_patientjournals_schema_journal, visualizations_patientjournals_schema_hospital_stay, visualizations_patientjournals_schema_patient, visualizations_patientjournals_schema_diagnoses, visualizations_patientjournals_schema_serum [EXTRACTED 1.00]
- **Patient Details** — visualizations_patientjournals_schema_patient, visualizations_patientjournals_schema_age, visualizations_patientjournals_schema_address [EXTRACTED 1.00]
- **Synthetic Dataset Pipeline** — visualizations_patientjournals_front_page_images, visualizations_patientjournals_journal_schema, visualizations_patientjournals_orchestrator, visualizations_patientjournals_preprocessing, visualizations_patientjournals_parallel_api_requests, visualizations_patientjournals_llm, visualizations_patientjournals_dataset [EXTRACTED 1.00]
- **Reproducible Dataset Lineage** — readme_immutable_schema_versioning, readme_image_name_dataset_identity, readme_row_level_provenance, readme_image_processing_measurements [INFERRED 0.85]

## Communities (66 total, 11 thin omitted)

### Community 0 - "JobStore"
Cohesion: 0.05
Nodes (41): Connection, Row, Application services and desktop UI for PatientJournals., _copy_dataset_into_job(), _dataset_files(), JobStore, _json_dumps(), _json_loads() (+33 more)

### Community 1 - "tools.py"
Cohesion: 0.10
Nodes (45): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, create_local_model_client(), _emit(), _input_without_existing() (+37 more)

### Community 2 - "WorkflowService"
Cohesion: 0.06
Nodes (40): BaseHTTPRequestHandler, CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes() (+32 more)

### Community 3 - "retry.py"
Cohesion: 0.13
Nodes (37): _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _append_retry_to_source_metadata(), _build_anthropic_batch_requests_for_retry(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _build_retry_gemini_request_line() (+29 more)

### Community 4 - "config/schemas.py"
Cohesion: 0.06
Nodes (62): FieldConfidenceByPointer, fixture, model_validator, list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), resolve_schema_class() (+54 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.10
Nodes (15): BooleanVar, Canvas, Frame, Label, LabelFrame, Misc, main(), _open_in_file_browser() (+7 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (50): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+42 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (38): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+30 more)

### Community 8 - "jobs.py"
Cohesion: 0.11
Nodes (49): _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows(), _dataset_files_in_run_dir(), _dataset_rows() (+41 more)

### Community 9 - "status.py"
Cohesion: 0.10
Nodes (42): _batch_model_progress(), cancel_batch_run(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job() (+34 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.07
Nodes (36): batch_run_provider(), find_dataset_near(), list_batch_chunks(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return the recorded retrieval results for a run, if it has been retrieved., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.… (+28 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "workflows.py"
Cohesion: 0.15
Nodes (30): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+22 more)

### Community 13 - "submit.py"
Cohesion: 0.05
Nodes (82): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), BatchChunkPlan, BatchCollectOutputsRequest, BatchSubmitPlan, BatchSubmitRequest, BatchSubmitService, Bucket (+74 more)

### Community 14 - "retrieve_batch"
Cohesion: 0.12
Nodes (21): CollectOutputsResult, RetrieveBatchResult, _arg_batch_names(), _effective_duplicate_strategy(), _expected_success_keys(), _extract_anthropic_response_metadata(), _extract_location_from_batch_name(), _failed_retry_num_batches() (+13 more)

### Community 15 - "ui.py"
Cohesion: 0.13
Nodes (20): DuplicateStrategy, build_retrieve_command(), build_submit_command(), build_validation_command(), app_settings_path(), AppSettings, CommandSpec, Path (+12 more)

### Community 16 - "validation/cli.py"
Cohesion: 0.13
Nodes (28): Random, build_validation_datapoints(), choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type() (+20 more)

### Community 17 - "inspection.py"
Cohesion: 0.10
Nodes (37): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, Local data inspection and health checks., collect_files() (+29 more)

### Community 18 - "ValidatorApp"
Cohesion: 0.15
Nodes (7): Button, Entry, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 19 - "process_file"
Cohesion: 0.18
Nodes (14): _api_key_recovery_failure_reason(), _compact_exception_text(), _generate_recovery_response(), _guess_blob_mime_type(), BaseException, Blob, _recover_one_missing_page_via_api_key(), _redact_error_text() (+6 more)

### Community 20 - "resolve_model_spec"
Cohesion: 0.43
Nodes (6): all_registered_models(), _infer_provider_from_model_name(), ModelSpec, ProviderName, registered_google_models(), resolve_model_spec()

### Community 21 - "patientjournals/tasks.py"
Cohesion: 0.09
Nodes (31): Batch upload, submission, status, and retrieval commands., _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show() (+23 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.18
Nodes (27): _chosen_token_logprobs(), _collect_leaf_value_spans(), _collect_logprobs_by_pointer(), confidence_from_avg_logprobs(), _escape_pointer_segment(), extract_field_confidence_by_pointer(), extract_response_avg_logprobs(), extract_response_metadata() (+19 more)

### Community 24 - "model_client.py"
Cohesion: 0.18
Nodes (18): _build_provider_client(), _extract_anthropic_response_text(), _extract_openai_response_text(), _import_anthropic_async_client(), _import_openai_async_client(), LocalGenerationResult, LocalModelClient, _pick_value() (+10 more)

### Community 25 - "BrowserValidationSession"
Cohesion: 0.24
Nodes (4): BrowserValidationSession, Server-side validation state for the browser validator., _score_for_label(), _stringify_value()

### Community 26 - "generate.py"
Cohesion: 0.19
Nodes (21): _anthropic_metadata(), combine_subagent_jsonl_sources(), CombinedSubagentOutputs, Validate specialist results and join them into ordinary page records., _request_key_and_metadata(), generate_data(), Any, BaseModel (+13 more)

### Community 27 - "test_subagents.py"
Cohesion: 0.24
Nodes (11): decode_specialist_request_key(), page_key_from_request_key(), _FakeBlob, _gemini_line(), test_batch_request_fanout_and_disabled_compatibility(), test_combiner_joins_out_of_order_specialist_results(), test_combiner_withholds_page_when_specialist_is_missing(), test_merge_specialists_validates_full_page() (+3 more)

### Community 28 - "ocr.py"
Cohesion: 0.18
Nodes (11): _break_name(), _configured_backend(), detect_configured_ocr_batch(), extract_google_vision_lines(), GoogleVisionOcrBackend, OcrImageInput, Collapse Vision's symbol hierarchy into token-efficient visual lines., Send up to 16 images through one Vision images:annotate RPC. (+3 more)

### Community 29 - "submit_requests.py"
Cohesion: 0.19
Nodes (23): ocr_context_for_blob(), _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_anthropic_manifest_lines(), _build_request_config() (+15 more)

### Community 30 - "ImageAccessService"
Cohesion: 0.38
Nodes (4): ImageAccessService, Any, Path, Short-lived image links for dataset inspection and submission previews.

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - ".name"
Cohesion: 0.18
Nodes (33): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), _list_page_keys(), _blob_extension(), _blob_size() (+25 more)

### Community 33 - "ocr_context.py"
Cohesion: 0.19
Nodes (16): _cache_key(), CloudBlobIdentity, CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), _PendingOcr, prepare_ocr_metadata_for_blob() (+8 more)

### Community 34 - "test_ocr.py"
Cohesion: 0.18
Nodes (16): OcrAttempt, OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., FakeBucket, FakeOcrBackend, _png_bytes(), _symbol(), test_batch_ocr_preparation_creates_generation_bound_reusable_sidecar() (+8 more)

### Community 35 - "resolve_batch_run_readiness"
Cohesion: 0.22
Nodes (13): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks_with_state(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini…, Query the batch API once and aggregate chunk states into a job-level status.… (+5 more)

### Community 36 - "preprocess.py"
Cohesion: 0.15
Nodes (21): Image, Protocol, detect_configured_ocr(), detect_ocr(), image_identity(), OcrBackend, Read canonical dimensions and digest from the exact serialized bytes., Run configured OCR, failing open unless ``ocr_required`` is set. (+13 more)

### Community 37 - ".__init__"
Cohesion: 0.15
Nodes (13): ImageSource, BrowserValidationManager, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Any, Path (+5 more)

### Community 38 - "OcrDocument"
Cohesion: 0.21
Nodes (7): OcrDocument, OCR derived from, and cryptographically bound to, one image payload., Render every line with minimal syntax and no repeated field names., render_ocr_context(), PreparedPage, The exact image payload and OCR context supplied to a model request., test_ocr_prompt_format_contains_all_text_without_json_field_overhead()

### Community 39 - "Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?, Source Nodes

### Community 40 - "JobRegistry"
Cohesion: 0.24
Nodes (10): JobRegistry, list_app_registry_jobs(), list_cloud_batch_jobs(), _primary_request_count_from_payload(), RegisteredJob, _retry_attempt_label(), start_command(), _summary_from_store_record() (+2 more)

### Community 41 - "Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/, Source Nodes

### Community 43 - "analysis.py"
Cohesion: 0.60
Nodes (10): _add_accuracy_scores(), load_validations(), main(), plot_label_distribution(), plot_nested_accuracy(), plot_overall_accuracy(), plot_top_level_accuracy(), DataFrame (+2 more)

### Community 44 - "Path"
Cohesion: 0.24
Nodes (18): _anthropic_custom_id_for_key(), _download_from_mldev_output(), _extract_batch_names_from_payload(), _find_submit_run_dir(), _flush_rows(), _latest_batch_job_file(), _normalize_key(), _output_destinations_from_submit_run() (+10 more)

### Community 45 - "get_batch_client"
Cohesion: 0.24
Nodes (15): _candidate_model_ids(), _check_model_ids(), _ConfigSnapshot, _iter_models(), main(), _norm(), _parse_args(), _print_models() (+7 more)

### Community 46 - "processing_metrics.py"
Cohesion: 0.38
Nodes (11): append_processing_record(), base_image_record(), _counter(), _number(), _numeric_summary(), Any, Path, read_processing_records() (+3 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 49 - "prepare_ocr.py"
Cohesion: 0.24
Nodes (11): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+3 more)

### Community 58 - "retrieve.py"
Cohesion: 0.13
Nodes (27): add_reproducibility_columns(), add_response_metadata_columns(), _await_completion(), _batch_job_state(), _batch_job_successful(), _build_api_key_generation_config(), _dataset_content_type(), _download_from_anthropic_output() (+19 more)

### Community 59 - "collect_outputs.py"
Cohesion: 0.14
Nodes (31): Counter, collect_outputs(), collect_valid_outputs_from_jsonl_sources(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines() (+23 more)

### Community 60 - "read_dataset_preview"
Cohesion: 0.50
Nodes (4): Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns…, read_dataset_preview(), test_read_dataset_preview_csv(), test_read_dataset_preview_jsonl()

### Community 61 - "run_layout.py"
Cohesion: 0.17
Nodes (19): category_root(), classify_legacy_dir(), _created_at_from_name(), document_existing_runs(), iter_all_run_dirs(), Path, Central conventions for the runs/ output folder. All job output lives under a…, Write a README documenting the runs/ layout. Returns its path. (+11 more)

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

### Community 65 - "image_name_from_reference"
Cohesion: 0.20
Nodes (17): copy_dataset_rows_for_image_names(), copy_dataset_rows_for_keys(), load_dataset_image_coverage(), load_dataset_key_coverage(), normalize_dataset_image_name(), _normalize_output_format(), Path, duplicate_image_names() (+9 more)

### Community 66 - "recover_dataset_gaps"
Cohesion: 0.18
Nodes (12): _api_recovery_error_rows(), _api_recovery_error_summary(), Retrieve a submitted batch in-process and record the result on the submit run.…, Resubmit the requests that did not succeed as a fresh batch. Clears the…, Fill in failed/missing pages with synchronous API calls and record results.…, Recover only the pages genuinely missing from the existing dataset via API.…, recover_dataset_gaps(), recover_failed_via_api() (+4 more)

### Community 67 - "Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage., Source Nodes

## Knowledge Gaps
- **46 isolated node(s):** `patientjournals`, `UploadProfile`, `Batch-first architecture`, `graphify`, `Answer` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `CloudBlobIdentity` (2× useful, score=1.999765272)
- `LocalModelClient` (2× useful, score=1.997808494) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JobStore` connect `JobStore` to `WorkflowService`, `recover_dataset_gaps`, `jobs.py`, `test_app_architecture.py`, `workflows.py`, `ui.py`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `JobRegistry`, `workflows.py`, `ui.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `ui.py` to `.name`, `JobStore`, `WorkflowService`, `recover_dataset_gaps`, `PatientJournalsApp`, `jobs.py`, `status.py`, `test_app_architecture.py`, `workflows.py`, `ImageAccessService`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `WorkflowService` (e.g. with `AppHandler` and `ImageAccessService`) actually correct?**
  _`WorkflowService` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `PatientJournalsApp` (e.g. with `JobRegistry` and `AppSettings`) actually correct?**
  _`PatientJournalsApp` has 4 INFERRED edges - model-reasoned connections that need verification._