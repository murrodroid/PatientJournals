# Graph Report - PatientJournals  (2026-08-27)

## Corpus Check
- 90 files · ~108,201 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1547 nodes · 4468 edges · 60 communities (49 shown, 11 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 182 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `88801ce8`
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
- local/service.py
- bucket.py
- OcrDocument
- run_layout.py
- access.py
- Journal
- browser.py
- collect_outputs.py
- output_records.py
- resolve_batch_run_readiness
- image_name_from_reference
- .__init__
- ImageAccessService
- test_data_inspection.py
- JobRegistry
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- Path
- get_batch_client
- processing_metrics.py
- Journal
- test_validation_sampling.py
- Path
- patientjournals/__init__.py
- local/__init__.py
- patientjournals
- patientjournals.app.access
- patientjournals.app.catalog
- patientjournals.app.datasets
- patientjournals.app.image_access
- patientjournals.app.settings_store
- _await_completion
- _api_key_recovery_failure_reason

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 64 edges
2. `AppSettings` - 57 edges
3. `WorkflowService` - 54 edges
4. `PatientJournalsApp` - 44 edges
5. `retrieve_batch()` - 41 edges
6. `submit_batch()` - 39 edges
7. `serializable()` - 33 edges
8. `image_name_from_reference()` - 32 edges
9. `normalize_prefix()` - 29 edges
10. `finalize_dataset_with_failed_rows()` - 28 edges

## Surprising Connections (you probably didn't know these)
- `test_job_store_persists_background_tasks()` --uses--> `JobStore`  [INFERRED]
  tests/test_app_workflows.py → src/patientjournals/app/job_store.py
- `test_recover_dataset_gaps_only_targets_missing_pages()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_recover_dataset_gaps_reports_zero_row_api_completion()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_failed_page_retry_can_split_into_multiple_chunks()` --calls--> `_submit_failed_pages_as_batch()`  [INFERRED]
  tests/test_batch_retrieve_recovery.py → src/patientjournals/batch/retry.py
- `test_gemini_model_progress_counts_prediction_rows()` --indirect_call--> `status()`  [INFERRED]
  tests/test_batch_status.py → src/patientjournals/tasks.py

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

## Communities (60 total, 11 thin omitted)

### Community 0 - "JobStore"
Cohesion: 0.06
Nodes (37): Connection, Row, _copy_dataset_into_job(), _dataset_files(), JobStore, _json_dumps(), _json_loads(), Path (+29 more)

### Community 1 - "tools.py"
Cohesion: 0.18
Nodes (21): build_path_id_set(), _candidate_path_ids(), _cfg_get(), copy_dataset(), filter_dataset_by_image_names(), filter_dataset_by_input_ids(), flush_rows(), _gather_files() (+13 more)

### Community 2 - "WorkflowService"
Cohesion: 0.15
Nodes (10): _apply_runtime_overrides(), poll_local_batch_states(), One-shot API poll mapping each unfinished local batch run_dir to a live status.…, _restore_runtime_overrides(), run_local_draft_direct(), command_override_payload(), Any, App-facing workflow API. Tk, web handlers, and tests should call this layer… (+2 more)

### Community 3 - "retry.py"
Cohesion: 0.06
Nodes (62): ocr_context_for_blob(), _generate_recovery_response(), _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _append_retry_to_source_metadata(), _build_anthropic_batch_requests_for_retry(), _build_retry_anthropic_manifest_line() (+54 more)

### Community 4 - "config/schemas.py"
Cohesion: 0.06
Nodes (60): FieldConfidenceByPointer, model_validator, list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), resolve_schema_class(), ModelOption (+52 more)

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
Nodes (60): _api_recovery_error_rows(), _api_recovery_error_summary(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows() (+52 more)

### Community 9 - "status.py"
Cohesion: 0.10
Nodes (42): _batch_model_progress(), cancel_batch_run(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job() (+34 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.07
Nodes (37): batch_run_provider(), find_dataset_near(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns…, One row per batch submission from the authoritative app store. (+29 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "datasets.py"
Cohesion: 0.19
Nodes (24): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), inspect_cloud_dataset() (+16 more)

### Community 13 - "submit.py"
Cohesion: 0.05
Nodes (90): BatchChunkPlan, BatchSubmitPlan, _batch_state_and_success(), _build_chunk_entry(), _build_rerun_entries(), _chunk_label(), _chunk_requests_file_name(), _discover_request_files_in_run_dir() (+82 more)

### Community 14 - "retrieve.py"
Cohesion: 0.11
Nodes (37): add_reproducibility_columns(), add_response_metadata_columns(), _build_api_key_generation_config(), _dataset_content_type(), _download_from_anthropic_output(), _download_from_vertex_gcs_output(), _effective_duplicate_strategy(), _expected_success_keys() (+29 more)

### Community 15 - "ui.py"
Cohesion: 0.12
Nodes (30): DuplicateStrategy, list_local_dataset_library(), build_retrieve_command(), build_submit_command(), build_validation_command(), Retrieve a submitted batch in-process and record the result on the submit run.…, Resubmit the requests that did not succeed as a fresh batch. Clears the…, resubmit_failed_requests() (+22 more)

### Community 16 - "validation/cli.py"
Cohesion: 0.13
Nodes (28): Random, build_validation_datapoints(), choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type() (+20 more)

### Community 17 - "inspection.py"
Cohesion: 0.20
Nodes (28): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, collect_files(), configured_image_extensions() (+20 more)

### Community 18 - "ValidatorApp"
Cohesion: 0.15
Nodes (7): Button, Entry, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 19 - "config/__init__.py"
Cohesion: 0.17
Nodes (18): _guess_blob_mime_type(), Blob, _recover_one_missing_page_via_api_key(), Configuration, schema, and model registry., generate_data(), process_file(), ProcessedFileResult, Any (+10 more)

### Community 21 - "test_batch_retrieve_recovery.py"
Cohesion: 0.10
Nodes (10): Batch upload, submission, status, and retrieval commands., FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency() (+2 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.18
Nodes (27): _chosen_token_logprobs(), _collect_leaf_value_spans(), _collect_logprobs_by_pointer(), confidence_from_avg_logprobs(), _escape_pointer_segment(), extract_field_confidence_by_pointer(), extract_response_avg_logprobs(), extract_response_metadata() (+19 more)

### Community 23 - "batch/service.py"
Cohesion: 0.15
Nodes (12): CollectOutputsResult, RetrieveBatchResult, BatchCollectOutputsRequest, BatchResultService, BatchRetrieveRequest, BatchSubmitRequest, BatchSubmitService, Bucket (+4 more)

### Community 24 - "patientjournals/tasks.py"
Cohesion: 0.29
Nodes (20): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+12 more)

### Community 25 - "BrowserValidationSession"
Cohesion: 0.24
Nodes (4): BrowserValidationSession, Server-side validation state for the browser validator., _score_for_label(), _stringify_value()

### Community 26 - "local/service.py"
Cohesion: 0.21
Nodes (18): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, _emit(), _input_without_existing(), LocalRunProgress (+10 more)

### Community 27 - "bucket.py"
Cohesion: 0.28
Nodes (16): _blob_extension(), _blob_size(), _bucket_depth(), _bucket_parent(), _bucket_relative_name(), _content_type_format_issue(), _extension_format_issue(), _folder_names_from_blob() (+8 more)

### Community 28 - "OcrDocument"
Cohesion: 0.05
Nodes (54): Image, Protocol, _cached_document(), _download(), ocr_document_for_blob(), OCR the exact current GCS object, reusing only digest-matched sidecars., _sidecar_name(), _store_document() (+46 more)

### Community 29 - "run_layout.py"
Cohesion: 0.17
Nodes (19): category_root(), classify_legacy_dir(), _created_at_from_name(), document_existing_runs(), iter_all_run_dirs(), Path, Central conventions for the runs/ output folder. All job output lives under a…, Write a README documenting the runs/ layout. Returns its path. (+11 more)

### Community 30 - "access.py"
Cohesion: 0.10
Nodes (26): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+18 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - "browser.py"
Cohesion: 0.24
Nodes (18): cloud_object_by_image_name(), _format_blob_updated(), list_cloud_dataset_choices(), list_cloud_dataset_library(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), CloudResolution, _list_page_keys() (+10 more)

### Community 33 - "collect_outputs.py"
Cohesion: 0.25
Nodes (17): Counter, collect_outputs(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines(), _list_page_image_names() (+9 more)

### Community 34 - "output_records.py"
Cohesion: 0.24
Nodes (14): collect_valid_outputs_from_jsonl_sources(), GeminiOutputParseResult, iter_gemini_jsonl_results(), normalize_output_key(), parse_gemini_output_record(), response_has_value(), gemini_response(), output_line() (+6 more)

### Community 35 - "resolve_batch_run_readiness"
Cohesion: 0.19
Nodes (15): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks(), list_batch_chunks_with_state(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini… (+7 more)

### Community 36 - "image_name_from_reference"
Cohesion: 0.17
Nodes (20): copy_dataset_rows_for_image_names(), copy_dataset_rows_for_keys(), load_dataset_image_coverage(), load_dataset_key_coverage(), normalize_dataset_image_name(), _normalize_output_format(), Path, resolve_continue_dataset_path() (+12 more)

### Community 37 - ".__init__"
Cohesion: 0.15
Nodes (13): ImageSource, BrowserValidationManager, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Any, Path (+5 more)

### Community 38 - "ImageAccessService"
Cohesion: 0.29
Nodes (4): ImageAccessService, Any, Path, Short-lived image links for dataset inspection and submission previews.

### Community 39 - "test_data_inspection.py"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 40 - "JobRegistry"
Cohesion: 0.31
Nodes (6): JobRegistry, list_app_registry_jobs(), list_cloud_batch_jobs(), RegisteredJob, _summary_from_store_record(), JobSummary

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
Cohesion: 0.42
Nodes (9): append_processing_record(), _counter(), _number(), _numeric_summary(), Any, Path, read_processing_records(), summarize_processing_records() (+1 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 49 - "Path"
Cohesion: 0.53
Nodes (5): _count_images(), _image_extensions(), _mtime(), Path, _updated_label()

### Community 58 - "_await_completion"
Cohesion: 0.33
Nodes (6): _await_completion(), _batch_job_state(), _batch_job_successful(), _get_batch_job(), _normalize_job_state(), _terminal_states()

### Community 59 - "_api_key_recovery_failure_reason"
Cohesion: 0.67
Nodes (4): _api_key_recovery_failure_reason(), _compact_exception_text(), BaseException, _redact_error_text()

## Knowledge Gaps
- **36 isolated node(s):** `patientjournals`, `UploadProfile`, `graphify`, `Answer`, `Outcome` (+31 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JobStore` connect `JobStore` to `WorkflowService`, `jobs.py`, `test_app_architecture.py`, `datasets.py`, `ui.py`, `access.py`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `JobRegistry`, `ui.py`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `ui.py` to `browser.py`, `JobStore`, `WorkflowService`, `PatientJournalsApp`, `ImageAccessService`, `jobs.py`, `status.py`, `test_app_architecture.py`, `access.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `WorkflowService` (e.g. with `AppHandler` and `ImageAccessService`) actually correct?**
  _`WorkflowService` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `PatientJournalsApp` (e.g. with `JobRegistry` and `AppSettings`) actually correct?**
  _`PatientJournalsApp` has 4 INFERRED edges - model-reasoned connections that need verification._