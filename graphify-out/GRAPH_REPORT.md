# Graph Report - PatientJournals  (2026-08-27)

## Corpus Check
- 92 files · ~110,439 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1587 nodes · 4557 edges · 65 communities (53 shown, 12 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 195 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `452a5471`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- JobStore
- collect_outputs.py
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
- datasets.py
- submit.py
- retrieve.py
- ui.py
- validation/cli.py
- inspection.py
- ValidatorApp
- config/__init__.py
- AppHandler
- test_batch_retrieve_recovery.py
- response_parsing.py
- batch/service.py
- patientjournals/tasks.py
- BrowserValidationSession
- submit_inputs.py
- bucket.py
- ocr.py
- submit_requests.py
- access.py
- Journal
- image_name_from_reference
- ocr_context.py
- test_ocr.py
- resolve_batch_run_readiness
- preprocess.py
- .__init__
- OcrDocument
- test_data_inspection.py
- JobRegistry
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- Path
- get_batch_client
- processing_metrics.py
- Journal
- build_validation_datapoints
- prepare_ocr.py
- patientjournals/__init__.py
- local/__init__.py
- patientjournals
- patientjournals.app.access
- patientjournals.app.catalog
- patientjournals.app.datasets
- patientjournals.app.image_access
- patientjournals.app.settings_store
- _recover_missing_pages_via_api_key
- _api_key_recovery_failure_reason
- _list_input_blobs
- test_run_layout.py
- FakeBlob
- BrowserValidationManager
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 64 edges
2. `AppSettings` - 57 edges
3. `WorkflowService` - 54 edges
4. `PatientJournalsApp` - 44 edges
5. `retrieve_batch()` - 41 edges
6. `submit_batch()` - 40 edges
7. `serializable()` - 33 edges
8. `image_name_from_reference()` - 32 edges
9. `build_storage_bucket()` - 30 edges
10. `normalize_prefix()` - 29 edges

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

## Communities (65 total, 12 thin omitted)

### Community 0 - "JobStore"
Cohesion: 0.06
Nodes (41): Connection, Row, _copy_dataset_into_job(), _dataset_files(), JobStore, _json_dumps(), _json_loads(), Path (+33 more)

### Community 1 - "collect_outputs.py"
Cohesion: 0.05
Nodes (89): Counter, ProgressCallback, collect_outputs(), collect_valid_outputs_from_jsonl_sources(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows() (+81 more)

### Community 2 - "WorkflowService"
Cohesion: 0.14
Nodes (12): _apply_runtime_overrides(), cancel_batch_run(), poll_local_batch_states(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, One-shot API poll mapping each unfinished local batch run_dir to a live status.…, _restore_runtime_overrides(), run_local_draft_direct(), command_override_payload() (+4 more)

### Community 3 - "retry.py"
Cohesion: 0.07
Nodes (56): ocr_context_for_blob(), _generate_recovery_response(), _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _append_retry_to_source_metadata(), _build_anthropic_batch_requests_for_retry(), _build_retry_anthropic_manifest_line() (+48 more)

### Community 4 - "config/schemas.py"
Cohesion: 0.05
Nodes (66): FieldConfidenceByPointer, model_validator, list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), resolve_schema_class(), ModelOption (+58 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.10
Nodes (15): BooleanVar, Canvas, Frame, Label, LabelFrame, Misc, main(), _open_in_file_browser() (+7 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (49): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+41 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (38): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+30 more)

### Community 8 - "jobs.py"
Cohesion: 0.09
Nodes (64): _api_recovery_error_rows(), _api_recovery_error_summary(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows() (+56 more)

### Community 9 - "status.py"
Cohesion: 0.08
Nodes (54): _batch_model_progress(), _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job(), _count_gemini_prediction_rows(), _count_jsonl_blob_lines() (+46 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.07
Nodes (36): Application services and desktop UI for PatientJournals., batch_run_provider(), find_dataset_near(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns… (+28 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "datasets.py"
Cohesion: 0.17
Nodes (26): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+18 more)

### Community 13 - "submit.py"
Cohesion: 0.16
Nodes (30): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), _batch_state_and_success(), _build_chunk_entry(), _build_rerun_entries(), _chunk_label(), _chunk_requests_file_name(), _discover_request_files_in_run_dir() (+22 more)

### Community 14 - "retrieve.py"
Cohesion: 0.11
Nodes (31): add_response_metadata_columns(), RetrieveBatchResult, _await_completion(), _batch_job_state(), _batch_job_successful(), _download_from_anthropic_output(), _effective_duplicate_strategy(), _expected_success_keys() (+23 more)

### Community 15 - "ui.py"
Cohesion: 0.10
Nodes (33): DuplicateStrategy, list_local_dataset_library(), build_retrieve_command(), build_submit_command(), build_validation_command(), app_settings_path(), AppSettings, CommandSpec (+25 more)

### Community 16 - "validation/cli.py"
Cohesion: 0.19
Nodes (19): Random, choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type(), _is_metadata_field() (+11 more)

### Community 17 - "inspection.py"
Cohesion: 0.20
Nodes (28): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, collect_files(), configured_image_extensions() (+20 more)

### Community 18 - "ValidatorApp"
Cohesion: 0.15
Nodes (7): Button, Entry, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 19 - "config/__init__.py"
Cohesion: 0.21
Nodes (14): add_reproducibility_columns(), _guess_blob_mime_type(), Blob, _recover_one_missing_page_via_api_key(), Configuration, schema, and model registry., generate_data(), process_file(), ProcessedFileResult (+6 more)

### Community 21 - "test_batch_retrieve_recovery.py"
Cohesion: 0.10
Nodes (10): Batch upload, submission, status, and retrieval commands., FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency() (+2 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.18
Nodes (27): _chosen_token_logprobs(), _collect_leaf_value_spans(), _collect_logprobs_by_pointer(), confidence_from_avg_logprobs(), _escape_pointer_segment(), extract_field_confidence_by_pointer(), extract_response_avg_logprobs(), extract_response_metadata() (+19 more)

### Community 23 - "batch/service.py"
Cohesion: 0.12
Nodes (18): CollectOutputsResult, BatchChunkPlan, BatchCollectOutputsRequest, BatchResultService, BatchSubmitPlan, BatchSubmitRequest, BatchSubmitService, Bucket (+10 more)

### Community 24 - "patientjournals/tasks.py"
Cohesion: 0.27
Nodes (21): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+13 more)

### Community 25 - "BrowserValidationSession"
Cohesion: 0.21
Nodes (6): BrowserValidationSession, Any, Server-side validation state for the browser validator., ValidationImageRef, _score_for_label(), _stringify_value()

### Community 26 - "submit_inputs.py"
Cohesion: 0.19
Nodes (26): _allowed_extensions(), _apply_fp_mode_to_blobs(), _apply_fp_mode_to_pdf_paths(), _apply_image_name_restriction(), _apply_year_filter_to_blobs(), _assert_gcs_input_source(), _configured_year_filter_tokens(), _dedupe_blob_image_names() (+18 more)

### Community 27 - "bucket.py"
Cohesion: 0.28
Nodes (16): _blob_extension(), _blob_size(), _bucket_depth(), _bucket_parent(), _bucket_relative_name(), _content_type_format_issue(), _extension_format_issue(), _folder_names_from_blob() (+8 more)

### Community 28 - "ocr.py"
Cohesion: 0.15
Nodes (14): Protocol, _break_name(), _configured_backend(), detect_configured_ocr(), detect_ocr(), extract_google_vision_lines(), GoogleVisionOcrBackend, image_identity() (+6 more)

### Community 29 - "submit_requests.py"
Cohesion: 0.18
Nodes (22): _normalize_prefix(), _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_request_config(), _build_request_line() (+14 more)

### Community 30 - "access.py"
Cohesion: 0.16
Nodes (18): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+10 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - "image_name_from_reference"
Cohesion: 0.14
Nodes (25): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_library(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), ImageAccessService, Any (+17 more)

### Community 33 - "ocr_context.py"
Cohesion: 0.21
Nodes (13): _cache_key(), CloudBlobIdentity, CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), prepare_ocr_metadata_for_blob(), Load a generation-matched sidecar without downloading the image. (+5 more)

### Community 34 - "test_ocr.py"
Cohesion: 0.20
Nodes (14): OcrAttempt, OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., FakeBucket, FakeOcrBackend, _png_bytes(), _symbol(), test_batch_ocr_preparation_creates_generation_bound_reusable_sidecar() (+6 more)

### Community 35 - "resolve_batch_run_readiness"
Cohesion: 0.18
Nodes (15): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks(), list_batch_chunks_with_state(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini… (+7 more)

### Community 36 - "preprocess.py"
Cohesion: 0.23
Nodes (14): Image, crop_margins(), enhance_contrast(), image_to_bytes(), load_image(), prepare_page(), preprocess_image(), preprocess_image_with_metadata() (+6 more)

### Community 37 - ".__init__"
Cohesion: 0.18
Nodes (12): ImageSource, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Path, SamplingMode, _safe_user() (+4 more)

### Community 38 - "OcrDocument"
Cohesion: 0.21
Nodes (7): OcrDocument, Render every line with minimal syntax and no repeated field names., OCR derived from, and cryptographically bound to, one image payload., render_ocr_context(), PreparedPage, The exact image payload and OCR context supplied to a model request., test_ocr_prompt_format_contains_all_text_without_json_field_overhead()

### Community 39 - "test_data_inspection.py"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 40 - "JobRegistry"
Cohesion: 0.29
Nodes (7): JobRegistry, list_app_registry_jobs(), RegisteredJob, start_command(), _summary_from_store_record(), JobSummary, test_job_registry_roundtrip()

### Community 41 - "Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/, Source Nodes

### Community 43 - "analysis.py"
Cohesion: 0.60
Nodes (10): _add_accuracy_scores(), load_validations(), main(), plot_label_distribution(), plot_nested_accuracy(), plot_overall_accuracy(), plot_top_level_accuracy(), DataFrame (+2 more)

### Community 44 - "Path"
Cohesion: 0.24
Nodes (18): _anthropic_custom_id_for_key(), _arg_batch_names(), _download_from_mldev_output(), _extract_batch_names_from_payload(), _find_submit_run_dir(), _latest_batch_job_file(), _normalize_key(), _output_destinations_from_submit_run() (+10 more)

### Community 45 - "get_batch_client"
Cohesion: 0.24
Nodes (15): _candidate_model_ids(), _check_model_ids(), _ConfigSnapshot, _iter_models(), main(), _norm(), _parse_args(), _print_models() (+7 more)

### Community 46 - "processing_metrics.py"
Cohesion: 0.38
Nodes (11): append_processing_record(), base_image_record(), _counter(), _number(), _numeric_summary(), Any, Path, read_processing_records() (+3 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 48 - "build_validation_datapoints"
Cohesion: 0.13
Nodes (8): build_validation_datapoints(), validation_sampling_group_key(), Validation UI and reporting commands., test_versioned_validation_uses_each_rows_schema_and_model(), test_balanced_ucb_prioritizes_under_sampled_schema_field(), test_balanced_ucb_separates_missing_and_present_values(), test_random_sampling_uses_unvalidated_datapoints(), test_validation_candidates_are_schema_fields_only()

### Community 49 - "prepare_ocr.py"
Cohesion: 0.29
Nodes (9): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+1 more)

### Community 58 - "_recover_missing_pages_via_api_key"
Cohesion: 0.18
Nodes (12): _build_api_key_generation_config(), _dataset_content_type(), _download_from_vertex_gcs_output(), _normalize_prefix(), _parse_gcs_uri(), _recover_missing_pages_via_api_key(), _recover_missing_pages_via_api_key_async(), _RecoveryResult (+4 more)

### Community 59 - "_api_key_recovery_failure_reason"
Cohesion: 0.67
Nodes (4): _api_key_recovery_failure_reason(), _compact_exception_text(), BaseException, _redact_error_text()

### Community 60 - "_list_input_blobs"
Cohesion: 0.40
Nodes (6): _list_input_blobs(), FakeBlob, FakeBucket, test_list_input_blobs_raises_when_restriction_matches_nothing(), test_list_input_blobs_scopes_to_restricted_image_names(), test_list_input_blobs_skips_duplicate_image_names_with_audit()

### Community 61 - "test_run_layout.py"
Cohesion: 0.31
Nodes (7): Shared dataset, parsing, and output helpers., _mk(), test_document_existing_runs_backfills_kind(), test_iter_all_run_dirs(), test_iter_run_dirs_reads_both_layouts(), test_reorganize_runs_dry_run_does_not_move(), test_reorganize_runs_moves_and_fixes_references()

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

## Knowledge Gaps
- **40 isolated node(s):** `patientjournals`, `UploadProfile`, `Batch-first architecture`, `graphify`, `Answer` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `LocalModelClient` (2× useful, score=1.998601947)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JobStore` connect `JobStore` to `WorkflowService`, `jobs.py`, `test_app_architecture.py`, `datasets.py`, `ui.py`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `JobRegistry`, `ui.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `ui.py` to `image_name_from_reference`, `JobStore`, `WorkflowService`, `PatientJournalsApp`, `jobs.py`, `test_app_architecture.py`, `access.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `WorkflowService` (e.g. with `AppHandler` and `ImageAccessService`) actually correct?**
  _`WorkflowService` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `PatientJournalsApp` (e.g. with `JobRegistry` and `AppSettings`) actually correct?**
  _`PatientJournalsApp` has 4 INFERRED edges - model-reasoned connections that need verification._