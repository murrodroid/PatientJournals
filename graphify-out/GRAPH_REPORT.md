# Graph Report - PatientJournals  (2026-08-27)

## Corpus Check
- 99 files · ~115,512 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1666 nodes · 4856 edges · 66 communities (56 shown, 10 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 243 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f3025149`
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
- schema_specialists
- config/__init__.py
- access.py
- test_batch_retrieve_recovery.py
- response_parsing.py
- submit_inputs.py
- model_client.py
- submit_requests.py
- patientjournals/tasks.py
- bucket.py
- ocr.py
- test_run_layout.py
- .name
- Journal
- test_data_inspection.py
- ocr_context.py
- OcrDocument
- resolve_batch_run_readiness
- preprocess.py
- test_subagents.py
- prompts.py
- Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?
- JobRegistry
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- Path
- get_batch_client
- processing_metrics.py
- Journal
- AppHandler
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
- collect_outputs.py
- _list_input_blobs
- _api_key_recovery_failure_reason
- FakeBlob
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.
- batch/service.py
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

## Communities (66 total, 10 thin omitted)

### Community 0 - "JobStore"
Cohesion: 0.06
Nodes (40): Connection, Row, _copy_dataset_into_job(), _dataset_files(), JobStore, _json_dumps(), _json_loads(), Path (+32 more)

### Community 1 - "tools.py"
Cohesion: 0.07
Nodes (59): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, _emit(), _input_without_existing(), LocalRunProgress (+51 more)

### Community 2 - "WorkflowService"
Cohesion: 0.14
Nodes (12): _apply_runtime_overrides(), poll_local_batch_states(), One-shot API poll mapping each unfinished local batch run_dir to a live status.…, Submit a cloud batch in-process so the run directory is captured immediately., _restore_runtime_overrides(), run_batch_draft_direct(), run_local_draft_direct(), command_override_payload() (+4 more)

### Community 3 - "retry.py"
Cohesion: 0.13
Nodes (37): _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _append_retry_to_source_metadata(), _build_anthropic_batch_requests_for_retry(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _build_retry_gemini_request_line() (+29 more)

### Community 4 - "config/schemas.py"
Cohesion: 0.05
Nodes (67): FieldConfidenceByPointer, fixture, model_validator, list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), resolve_schema_class() (+59 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.06
Nodes (20): BooleanVar, Button, Canvas, Entry, Frame, Label, LabelFrame, Misc (+12 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (50): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+42 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (38): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+30 more)

### Community 8 - "jobs.py"
Cohesion: 0.08
Nodes (64): _api_recovery_error_rows(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows(), _dataset_files_in_run_dir() (+56 more)

### Community 9 - "status.py"
Cohesion: 0.07
Nodes (56): _batch_model_progress(), cancel_batch_run(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job() (+48 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.07
Nodes (36): Application services and desktop UI for PatientJournals., batch_run_provider(), find_dataset_near(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns… (+28 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "datasets.py"
Cohesion: 0.16
Nodes (29): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+21 more)

### Community 13 - "submit.py"
Cohesion: 0.13
Nodes (36): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), _batch_state_and_success(), _build_chunk_entry(), _build_rerun_entries(), _chunk_label(), _chunk_requests_file_name(), _discover_request_files_in_run_dir() (+28 more)

### Community 14 - "retrieve.py"
Cohesion: 0.12
Nodes (30): add_response_metadata_columns(), _await_completion(), _batch_job_state(), _batch_job_successful(), _download_from_anthropic_output(), _effective_duplicate_strategy(), _expected_success_keys(), _extract_anthropic_response_metadata() (+22 more)

### Community 15 - "ui.py"
Cohesion: 0.09
Nodes (35): DuplicateStrategy, build_retrieve_command(), build_submit_command(), build_validation_command(), start_command(), app_settings_path(), AppSettings, CommandSpec (+27 more)

### Community 16 - "validation/cli.py"
Cohesion: 0.05
Nodes (51): ImageSource, Random, BrowserValidationManager, BrowserValidationSession, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index() (+43 more)

### Community 17 - "inspection.py"
Cohesion: 0.21
Nodes (27): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, collect_files(), default_batch_root() (+19 more)

### Community 18 - "schema_specialists"
Cohesion: 0.15
Nodes (26): _anthropic_metadata(), combine_subagent_jsonl_sources(), CombinedSubagentOutputs, Path, Validate specialist results and join them into ordinary page records., _request_key_and_metadata(), write_combined_subagent_outputs(), generate_data() (+18 more)

### Community 19 - "config/__init__.py"
Cohesion: 0.24
Nodes (12): add_reproducibility_columns(), _generate_recovery_response(), _guess_blob_mime_type(), Blob, _recover_one_missing_page_via_api_key(), Configuration, schema, and model registry., process_file(), ProcessedFileResult (+4 more)

### Community 20 - "access.py"
Cohesion: 0.16
Nodes (18): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+10 more)

### Community 21 - "test_batch_retrieve_recovery.py"
Cohesion: 0.10
Nodes (10): Batch upload, submission, status, and retrieval commands., FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency() (+2 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.14
Nodes (32): GeminiOutputParseResult, iter_gemini_jsonl_results(), normalize_output_key(), parse_gemini_output_record(), response_has_value(), _chosen_token_logprobs(), _collect_leaf_value_spans(), _collect_logprobs_by_pointer() (+24 more)

### Community 23 - "submit_inputs.py"
Cohesion: 0.18
Nodes (27): _allowed_extensions(), _apply_fp_mode_to_blobs(), _apply_fp_mode_to_pdf_paths(), _apply_image_name_restriction(), _apply_year_filter_to_blobs(), _assert_gcs_input_source(), _configured_year_filter_tokens(), _dedupe_blob_image_names() (+19 more)

### Community 24 - "model_client.py"
Cohesion: 0.15
Nodes (20): _build_provider_client(), create_local_model_client(), _extract_anthropic_response_text(), _extract_openai_response_text(), _import_anthropic_async_client(), _import_openai_async_client(), LocalGenerationResult, LocalModelClient (+12 more)

### Community 25 - "submit_requests.py"
Cohesion: 0.22
Nodes (19): ocr_context_for_blob(), _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_anthropic_manifest_lines(), _build_request_config() (+11 more)

### Community 26 - "patientjournals/tasks.py"
Cohesion: 0.27
Nodes (21): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+13 more)

### Community 27 - "bucket.py"
Cohesion: 0.27
Nodes (18): _blob_extension(), _blob_size(), _bucket_depth(), _bucket_parent(), _bucket_relative_name(), _content_type_format_issue(), _extension_format_issue(), _folder_names_from_blob() (+10 more)

### Community 28 - "ocr.py"
Cohesion: 0.17
Nodes (13): _break_name(), _configured_backend(), detect_configured_ocr_batch(), extract_google_vision_lines(), GoogleVisionOcrBackend, OcrAttempt, OcrImageInput, Collapse Vision's symbol hierarchy into token-efficient visual lines. (+5 more)

### Community 29 - "test_run_layout.py"
Cohesion: 0.31
Nodes (7): Shared dataset, parsing, and output helpers., _mk(), test_document_existing_runs_backfills_kind(), test_iter_all_run_dirs(), test_iter_run_dirs_reads_both_layouts(), test_reorganize_runs_dry_run_does_not_move(), test_reorganize_runs_moves_and_fixes_references()

### Community 30 - ".name"
Cohesion: 0.25
Nodes (15): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_library(), list_cloud_dataset_prefixes(), ImageAccessService, Any, Path (+7 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - "test_data_inspection.py"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 33 - "ocr_context.py"
Cohesion: 0.19
Nodes (16): _cache_key(), CloudBlobIdentity, CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), _PendingOcr, prepare_ocr_metadata_for_blob() (+8 more)

### Community 34 - "OcrDocument"
Cohesion: 0.14
Nodes (17): OcrDocument, OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., OCR derived from, and cryptographically bound to, one image payload., FakeBucket, FakeOcrBackend, _png_bytes(), _symbol() (+9 more)

### Community 35 - "resolve_batch_run_readiness"
Cohesion: 0.19
Nodes (15): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks(), list_batch_chunks_with_state(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini… (+7 more)

### Community 36 - "preprocess.py"
Cohesion: 0.13
Nodes (23): Image, Protocol, detect_configured_ocr(), detect_ocr(), image_identity(), OcrBackend, Read canonical dimensions and digest from the exact serialized bytes., Run configured OCR, failing open unless ``ocr_required`` is set. (+15 more)

### Community 37 - "test_subagents.py"
Cohesion: 0.27
Nodes (11): decode_specialist_request_key(), encode_specialist_request_key(), page_key_from_request_key(), _FakeBlob, _gemini_line(), test_batch_request_fanout_and_disabled_compatibility(), test_combiner_joins_out_of_order_specialist_results(), test_combiner_withholds_page_when_specialist_is_missing() (+3 more)

### Community 38 - "prompts.py"
Cohesion: 0.20
Nodes (8): build_subagent_prompt(), ocr_context_header(), _prompt(), Single source of truth for non-schema model prompt text. Edit page, sub-agent,…, Render the compact role brief shared by batch, retry, and local paths., Normalize source indentation without altering intentional line breaks., test_ocr_context_uses_central_header_template(), test_subagent_prompt_is_rendered_from_central_prompt_definitions()

### Community 39 - "Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?, Source Nodes

### Community 40 - "JobRegistry"
Cohesion: 0.43
Nodes (3): JobRegistry, RegisteredJob, test_job_registry_roundtrip()

### Community 41 - "Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/, Source Nodes

### Community 42 - "AGENTS.md"
Cohesion: 0.50
Nodes (3): Batch-first architecture, graphify, Prompt ownership

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

### Community 49 - "prepare_ocr.py"
Cohesion: 0.24
Nodes (11): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+3 more)

### Community 58 - "_recover_missing_pages_via_api_key"
Cohesion: 0.18
Nodes (12): _build_api_key_generation_config(), _dataset_content_type(), _download_from_vertex_gcs_output(), _normalize_prefix(), _parse_gcs_uri(), _recover_missing_pages_via_api_key(), _recover_missing_pages_via_api_key_async(), _RecoveryResult (+4 more)

### Community 59 - "collect_outputs.py"
Cohesion: 0.16
Nodes (27): Counter, _api_recovery_error_summary(), collect_outputs(), collect_valid_outputs_from_jsonl_sources(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows() (+19 more)

### Community 60 - "_list_input_blobs"
Cohesion: 0.40
Nodes (6): _list_input_blobs(), FakeBlob, FakeBucket, test_list_input_blobs_raises_when_restriction_matches_nothing(), test_list_input_blobs_scopes_to_restricted_image_names(), test_list_input_blobs_skips_duplicate_image_names_with_audit()

### Community 61 - "_api_key_recovery_failure_reason"
Cohesion: 0.67
Nodes (4): _api_key_recovery_failure_reason(), _compact_exception_text(), BaseException, _redact_error_text()

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

### Community 66 - "batch/service.py"
Cohesion: 0.11
Nodes (21): CollectOutputsResult, RetrieveBatchResult, BatchChunkPlan, BatchCollectOutputsRequest, BatchResultService, BatchRetrieveRequest, BatchSubmitPlan, BatchSubmitRequest (+13 more)

### Community 67 - "Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage., Source Nodes

## Knowledge Gaps
- **47 isolated node(s):** `patientjournals`, `UploadProfile`, `Batch-first architecture`, `Prompt ownership`, `graphify` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `CloudBlobIdentity` (2× useful, score=1.999765272)
- `LocalModelClient` (2× useful, score=1.997808494) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `JobStore` connect `JobStore` to `WorkflowService`, `jobs.py`, `test_app_architecture.py`, `datasets.py`, `ui.py`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `JobRegistry`, `ui.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `ValidatorApp` connect `PatientJournalsApp` to `validation/cli.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `WorkflowService` (e.g. with `AppHandler` and `ImageAccessService`) actually correct?**
  _`WorkflowService` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `PatientJournalsApp` (e.g. with `JobRegistry` and `AppSettings`) actually correct?**
  _`PatientJournalsApp` has 4 INFERRED edges - model-reasoned connections that need verification._