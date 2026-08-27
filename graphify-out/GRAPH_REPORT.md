# Graph Report - PatientJournals  (2026-08-27)

## Corpus Check
- 87 files · ~105,177 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1470 nodes · 4292 edges · 58 communities (49 shown, 9 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 165 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Schema and Job Store
- Local Output Collection
- Workflow Execution Controls
- Model Retry Pipeline
- Output Schema Configuration
- Web App Interface
- Dataset Dashboard Analytics
- Media Upload Preprocessing
- Batch Job Finalization
- Batch Status Management
- Run Inspection Services
- Project Documentation Concepts
- Dataset Source Workflows
- Batch Submission Pipeline
- Batch Retrieval Pipeline
- App Settings and UI
- Validation Sampling CLI
- Batch Data Inspection
- Desktop Validator UI
- Submission Input Resolution
- Model and Schema Catalog
- Batch Recovery Tests
- Response Parsing and Confidence
- Batch Service Orchestration
- Task Runner Commands
- Browser Validation Session
- Batch Request Construction
- Bucket Data Validation
- Local Generation Core
- Run Layout Migration
- Inspection Test Fixtures
- Synthetic Dataset Architecture
- Cloud Storage Synchronization
- Batch Target Resolution
- Cloud Client Model Checks
- Batch Readiness State
- Dataset Coverage Utilities
- Validation Session Setup
- Image Access Service
- Missing Page Recovery
- Job Registry Commands
- API Recovery Internals
- Validation Sampling Tests
- Validation Accuracy Analysis
- Processing Metrics
- Gemini Output Parsing
- Submission Input Tests
- Journal Schema Diagram
- Browser Validation Manager
- Error Redaction Utilities
- Transcription Package Docs
- Local Transcription Commands
- PatientJournals Package
- Access Control Module
- Catalog Module
- Datasets Module
- Image Access Module
- Settings Store Module

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
- `test_batch_retrieve_request_namespace()` --uses--> `BatchRetrieveRequest`  [INFERRED]
  tests/test_batch_service.py → src/patientjournals/batch/service.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Web App Service Architecture** — readme_patientjournals_app_access, readme_patientjournals_app_catalog, readme_patientjournals_app_dashboard, readme_patientjournals_app_datasets, readme_patientjournals_app_image_access, readme_patientjournals_app_job_store, readme_patientjournals_app_jobs, readme_patientjournals_app_schemas, readme_patientjournals_app_settings_store, readme_patientjournals_app_task_runner, readme_patientjournals_app_web, readme_patientjournals_app_workflows_workflowservice [EXTRACTED 1.00]
- **Local Model Provider Support** — readme_local_generation, readme_gemini, readme_openai, readme_anthropic [EXTRACTED 1.00]
- **Reproducible Dataset Lineage** — readme_immutable_schema_versioning, readme_image_name_dataset_identity, readme_row_level_provenance, readme_image_processing_measurements [INFERRED 0.85]
- **Journal Record Model** — visualizations_patientjournals_journal, visualizations_patientjournals_hospital_stay, visualizations_patientjournals_patient, visualizations_patientjournals_diagnoses, visualizations_patientjournals_serum [EXTRACTED 1.00]
- **Synthetic Dataset Pipeline** — visualizations_patientjournals_front_page_images, visualizations_patientjournals_journal_schema, visualizations_patientjournals_orchestrator, visualizations_patientjournals_preprocessing, visualizations_patientjournals_parallel_api_requests, visualizations_patientjournals_llm, visualizations_patientjournals_dataset [EXTRACTED 1.00]
- **Quality Assurance Flow** — visualizations_patientjournals_dataset, visualizations_patientjournals_validators, visualizations_patientjournals_validations, visualizations_patientjournals_accuracy [EXTRACTED 1.00]
- **Journal Composition** — visualizations_patientjournals_schema_journal, visualizations_patientjournals_schema_hospital_stay, visualizations_patientjournals_schema_patient, visualizations_patientjournals_schema_diagnoses, visualizations_patientjournals_schema_serum [EXTRACTED 1.00]
- **Patient Details** — visualizations_patientjournals_schema_patient, visualizations_patientjournals_schema_age, visualizations_patientjournals_schema_address [EXTRACTED 1.00]
- **Diagnosis Structure** — visualizations_patientjournals_schema_diagnoses, visualizations_patientjournals_schema_bottom, visualizations_patientjournals_schema_section [EXTRACTED 1.00]

## Communities (58 total, 9 thin omitted)

### Community 0 - "Schema and Job Store"
Cohesion: 0.06
Nodes (39): Connection, Row, Application services and desktop UI for PatientJournals., _copy_dataset_into_job(), _dataset_files(), JobStore, _json_dumps(), _json_loads() (+31 more)

### Community 1 - "Local Output Collection"
Cohesion: 0.06
Nodes (78): ProgressCallback, collect_outputs(), collect_valid_outputs_from_jsonl_sources(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines() (+70 more)

### Community 2 - "Workflow Execution Controls"
Cohesion: 0.07
Nodes (39): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+31 more)

### Community 3 - "Model Retry Pipeline"
Cohesion: 0.07
Nodes (53): _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _append_retry_to_source_metadata(), _build_anthropic_batch_requests_for_retry(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _build_retry_gemini_request_line() (+45 more)

### Community 4 - "Output Schema Configuration"
Cohesion: 0.07
Nodes (51): FieldConfidenceByPointer, model_validator, resolve_schema_class(), Address, Age, Bottom, Diagnoses, FrontPage (+43 more)

### Community 5 - "Web App Interface"
Cohesion: 0.10
Nodes (14): BooleanVar, Canvas, Frame, LabelFrame, Misc, main(), _open_in_file_browser(), PatientJournalsApp (+6 more)

### Community 6 - "Dataset Dashboard Analytics"
Cohesion: 0.09
Nodes (51): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+43 more)

### Community 7 - "Media Upload Preprocessing"
Cohesion: 0.10
Nodes (47): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+39 more)

### Community 8 - "Batch Job Finalization"
Cohesion: 0.11
Nodes (51): _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows(), _dataset_files_in_run_dir(), _dataset_rows() (+43 more)

### Community 9 - "Batch Status Management"
Cohesion: 0.10
Nodes (42): _batch_model_progress(), cancel_batch_run(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job() (+34 more)

### Community 10 - "Run Inspection Services"
Cohesion: 0.07
Nodes (37): batch_run_provider(), find_dataset_near(), list_batch_chunks(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return the recorded retrieval results for a run, if it has been retrieved., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.… (+29 more)

### Community 11 - "Project Documentation Concepts"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "Dataset Source Workflows"
Cohesion: 0.16
Nodes (28): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), inspect_cloud_dataset() (+20 more)

### Community 13 - "Batch Submission Pipeline"
Cohesion: 0.16
Nodes (32): _batch_state_and_success(), _build_chunk_entry(), _build_rerun_entries(), _chunk_label(), _chunk_requests_file_name(), _discover_request_files_in_run_dir(), _ensure_requests_files_for_rerun(), _entries_with_replacement() (+24 more)

### Community 14 - "Batch Retrieval Pipeline"
Cohesion: 0.12
Nodes (30): add_response_metadata_columns(), _await_completion(), _batch_job_state(), _batch_job_successful(), _download_from_anthropic_output(), _effective_duplicate_strategy(), _expected_success_keys(), _extract_anthropic_response_metadata() (+22 more)

### Community 15 - "App Settings and UI"
Cohesion: 0.14
Nodes (22): DuplicateStrategy, build_retrieve_command(), build_submit_command(), build_validation_command(), app_settings_path(), AppSettings, CommandSpec, Path (+14 more)

### Community 16 - "Validation Sampling CLI"
Cohesion: 0.14
Nodes (26): Random, build_validation_datapoints(), choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type() (+18 more)

### Community 17 - "Batch Data Inspection"
Cohesion: 0.20
Nodes (28): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, collect_files(), default_batch_root() (+20 more)

### Community 18 - "Desktop Validator UI"
Cohesion: 0.14
Nodes (8): Button, Entry, Label, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 19 - "Submission Input Resolution"
Cohesion: 0.18
Nodes (28): _allowed_extensions(), _apply_fp_mode_to_blobs(), _apply_fp_mode_to_pdf_paths(), _apply_image_name_restriction(), _apply_year_filter_to_blobs(), _assert_gcs_input_source(), _configured_year_filter_tokens(), _dedupe_blob_image_names() (+20 more)

### Community 20 - "Model and Schema Catalog"
Cohesion: 0.12
Nodes (16): BaseHTTPRequestHandler, list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), ModelOption, SchemaOption, AppHandler (+8 more)

### Community 21 - "Batch Recovery Tests"
Cohesion: 0.10
Nodes (10): Batch upload, submission, status, and retrieval commands., FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency() (+2 more)

### Community 22 - "Response Parsing and Confidence"
Cohesion: 0.19
Nodes (25): _chosen_token_logprobs(), _collect_leaf_value_spans(), _collect_logprobs_by_pointer(), confidence_from_avg_logprobs(), _escape_pointer_segment(), extract_field_confidence_by_pointer(), extract_response_avg_logprobs(), extract_response_text() (+17 more)

### Community 23 - "Batch Service Orchestration"
Cohesion: 0.12
Nodes (15): RetrieveBatchResult, BatchChunkPlan, BatchCollectOutputsRequest, BatchResultService, BatchSubmitPlan, BatchSubmitRequest, BatchSubmitService, Bucket (+7 more)

### Community 24 - "Task Runner Commands"
Cohesion: 0.29
Nodes (20): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+12 more)

### Community 25 - "Browser Validation Session"
Cohesion: 0.21
Nodes (6): BrowserValidationSession, Any, Server-side validation state for the browser validator., ValidationImageRef, _score_for_label(), _stringify_value()

### Community 26 - "Batch Request Construction"
Cohesion: 0.20
Nodes (20): _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_request_config(), _build_request_line(), _guess_mime_type() (+12 more)

### Community 27 - "Bucket Data Validation"
Cohesion: 0.26
Nodes (18): Counter, _blob_extension(), _blob_size(), _bucket_depth(), _bucket_parent(), _bucket_relative_name(), _content_type_format_issue(), _extension_format_issue() (+10 more)

### Community 28 - "Local Generation Core"
Cohesion: 0.18
Nodes (17): add_reproducibility_columns(), _generate_recovery_response(), _guess_blob_mime_type(), Blob, _recover_one_missing_page_via_api_key(), Configuration, schema, and model registry., generate_data(), process_file() (+9 more)

### Community 29 - "Run Layout Migration"
Cohesion: 0.17
Nodes (19): category_root(), classify_legacy_dir(), _created_at_from_name(), document_existing_runs(), iter_all_run_dirs(), Path, Central conventions for the runs/ output folder. All job output lives under a…, Write a README documenting the runs/ layout. Returns its path. (+11 more)

### Community 30 - "Inspection Test Fixtures"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 31 - "Synthetic Dataset Architecture"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - "Cloud Storage Synchronization"
Cohesion: 0.26
Nodes (16): _iter_cloud_validation_rows(), cloud_object_by_image_name(), _format_blob_updated(), list_cloud_dataset_library(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), _list_page_keys(), build_storage_bucket() (+8 more)

### Community 33 - "Batch Target Resolution"
Cohesion: 0.24
Nodes (18): _anthropic_custom_id_for_key(), _arg_batch_names(), _download_from_mldev_output(), _extract_batch_names_from_payload(), _find_submit_run_dir(), _latest_batch_job_file(), _normalize_key(), _output_destinations_from_submit_run() (+10 more)

### Community 34 - "Cloud Client Model Checks"
Cohesion: 0.24
Nodes (15): _candidate_model_ids(), _check_model_ids(), _ConfigSnapshot, _iter_models(), main(), _norm(), _parse_args(), _print_models() (+7 more)

### Community 35 - "Batch Readiness State"
Cohesion: 0.18
Nodes (15): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks_with_state(), poll_local_batch_states(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini… (+7 more)

### Community 36 - "Dataset Coverage Utilities"
Cohesion: 0.26
Nodes (11): copy_dataset_rows_for_image_names(), copy_dataset_rows_for_keys(), load_dataset_image_coverage(), load_dataset_key_coverage(), normalize_dataset_image_name(), _normalize_output_format(), Path, resolve_continue_dataset_path() (+3 more)

### Community 37 - "Validation Session Setup"
Cohesion: 0.22
Nodes (10): ImageSource, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Path, SamplingMode, _safe_user() (+2 more)

### Community 38 - "Image Access Service"
Cohesion: 0.29
Nodes (4): ImageAccessService, Any, Path, Short-lived image links for dataset inspection and submission previews.

### Community 39 - "Missing Page Recovery"
Cohesion: 0.18
Nodes (12): _api_recovery_error_rows(), _api_recovery_error_summary(), Retrieve a submitted batch in-process and record the result on the submit run.…, Resubmit the requests that did not succeed as a fresh batch. Clears the…, Fill in failed/missing pages with synchronous API calls and record results.…, Recover only the pages genuinely missing from the existing dataset via API.…, recover_dataset_gaps(), recover_failed_via_api() (+4 more)

### Community 40 - "Job Registry Commands"
Cohesion: 0.27
Nodes (8): JobRegistry, list_app_registry_jobs(), list_cloud_batch_jobs(), RegisteredJob, start_command(), _summary_from_store_record(), JobSummary, test_job_registry_roundtrip()

### Community 41 - "API Recovery Internals"
Cohesion: 0.18
Nodes (12): _build_api_key_generation_config(), _dataset_content_type(), _download_from_vertex_gcs_output(), _normalize_prefix(), _parse_gcs_uri(), _recover_missing_pages_via_api_key(), _recover_missing_pages_via_api_key_async(), _RecoveryResult (+4 more)

### Community 42 - "Validation Sampling Tests"
Cohesion: 0.17
Nodes (3): Validation UI and reporting commands., test_balanced_ucb_prioritizes_under_sampled_schema_field(), test_random_sampling_uses_unvalidated_datapoints()

### Community 43 - "Validation Accuracy Analysis"
Cohesion: 0.60
Nodes (10): _add_accuracy_scores(), load_validations(), main(), plot_label_distribution(), plot_nested_accuracy(), plot_overall_accuracy(), plot_top_level_accuracy(), DataFrame (+2 more)

### Community 44 - "Processing Metrics"
Cohesion: 0.42
Nodes (9): append_processing_record(), _counter(), _number(), _numeric_summary(), Any, Path, read_processing_records(), summarize_processing_records() (+1 more)

### Community 45 - "Gemini Output Parsing"
Cohesion: 0.39
Nodes (7): GeminiOutputParseResult, iter_gemini_jsonl_results(), normalize_output_key(), parse_gemini_output_record(), response_has_value(), extract_response_metadata(), Any

### Community 46 - "Submission Input Tests"
Cohesion: 0.42
Nodes (5): FakeBlob, FakeBucket, test_list_input_blobs_raises_when_restriction_matches_nothing(), test_list_input_blobs_scopes_to_restricted_image_names(), test_list_input_blobs_skips_duplicate_image_names_with_audit()

### Community 47 - "Journal Schema Diagram"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 49 - "Error Redaction Utilities"
Cohesion: 0.67
Nodes (4): _api_key_recovery_failure_reason(), _compact_exception_text(), BaseException, _redact_error_text()

## Knowledge Gaps
- **32 isolated node(s):** `patientjournals`, `UploadProfile`, `Structured Patient Journal Transcription`, `uv Project Workflow`, `Invoke Operational Tasks` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JobStore` connect `Schema and Job Store` to `Workflow Execution Controls`, `Missing Page Recovery`, `Batch Job Finalization`, `Run Inspection Services`, `Dataset Source Workflows`, `App Settings and UI`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `PatientJournalsApp` connect `Web App Interface` to `Job Registry Commands`, `Dataset Source Workflows`, `App Settings and UI`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `App Settings and UI` to `Schema and Job Store`, `Workflow Execution Controls`, `Batch Readiness State`, `Web App Interface`, `Image Access Service`, `Missing Page Recovery`, `Batch Job Finalization`, `Batch Status Management`, `Run Inspection Services`, `Dataset Source Workflows`, `Bucket Data Validation`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `WorkflowService` (e.g. with `AppHandler` and `ImageAccessService`) actually correct?**
  _`WorkflowService` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `PatientJournalsApp` (e.g. with `JobRegistry` and `AppSettings`) actually correct?**
  _`PatientJournalsApp` has 4 INFERRED edges - model-reasoned connections that need verification._