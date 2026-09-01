# Graph Report - PatientJournals  (2026-08-31)

## Corpus Check
- 111 files · ~163,011 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2160 nodes · 6442 edges · 84 communities (74 shown, 10 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 385 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `342d9791`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Path
- tools.py
- WorkflowService
- retry.py
- app/schemas.py
- PatientJournalsApp
- dashboard.py
- upload.py
- test_data_inspection.py
- status.py
- test_app_architecture.py
- PatientJournals Conda Environment
- inspection.py
- publication.py
- retrieve.py
- AppSettings
- test_validation_sampling.py
- ValidatorApp
- schema_specialists
- validation/cli.py
- access.py
- JobStore
- response_parsing.py
- image_name_from_reference
- test_subagents.py
- submit_requests.py
- routing.py
- RetrieveBatchResult
- ocr.py
- BrowserValidationSession
- PageCandidateRecord
- Journal
- browser.py
- ocr_context.py
- OcrDocument
- submit.py
- PatientJournals research pipeline
- retrieve_model_validation
- .name
- Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?
- generate.py
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- config/schemas.py
- model_client.py
- FakeBlob
- Journal
- resolve_batch_run_readiness
- test_schema_management.py
- patientjournals/__init__.py
- local/__init__.py
- patientjournals
- patientjournals.app.access
- patientjournals.app.catalog
- patientjournals.app.datasets
- patientjournals.app.image_access
- patientjournals.app.settings_store
- datasets.py
- patientjournals/tasks.py
- job_store.py
- jobs.py
- PageCandidateWriter
- bucket.py
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.
- test_batch_verify.py
- collect_outputs.py
- Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage.
- _extraction_model_for_snapshot
- config/__init__.py
- BrowserValidationManager
- _resolve_final_validation_policy
- _recover_one_missing_page_via_api_key
- preprocess.py
- run_layout.py
- prepare_ocr.py
- Path
- CloudBlobIdentity
- _ImmutableArtifactBlob
- verify.py
- get_batch_client
- JobRegistry
- _build_anthropic_batch_requests_for_retry
- prompts.py

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 78 edges
2. `AppSettings` - 68 edges
3. `WorkflowService` - 68 edges
4. `retrieve_batch()` - 58 edges
5. `PageCandidateRecord` - 52 edges
6. `retrieve_model_validation()` - 51 edges
7. `submit_batch()` - 48 edges
8. `PatientJournalsApp` - 47 edges
9. `submit_model_validation()` - 46 edges
10. `_Bucket` - 39 edges

## Surprising Connections (you probably didn't know these)
- `test_recover_dataset_gaps_only_targets_missing_pages()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_recover_dataset_gaps_reports_zero_row_api_completion()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_direct_retrieve_records_candidate_location_on_submit_run()` --uses--> `RetrieveBatchResult`  [INFERRED]
  tests/test_batch_verify.py → src/patientjournals/batch/results.py
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

## Communities (84 total, 10 thin omitted)

### Community 0 - "Path"
Cohesion: 0.16
Nodes (14): _copy_dataset_into_job(), _dataset_files(), _file_sha256(), _portable_run_id(), Path, Link a verifier batch to its extraction job without versioning data., Return the run ID used by shared publication, independent of its root., Record verifier results and optionally publish a dataset version once. (+6 more)

### Community 1 - "tools.py"
Cohesion: 0.11
Nodes (40): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, _emit(), _input_without_existing(), LocalRunProgress (+32 more)

### Community 2 - "WorkflowService"
Cohesion: 0.14
Nodes (12): _apply_runtime_overrides(), _restore_runtime_overrides(), run_local_draft_direct(), command_override_payload(), Any, Submit the candidate-aware verifier batch for a retrieved extraction., Retrieve and record the verifier batch linked to an extraction job., Resolve live chunk states and output readiness under one run snapshot. (+4 more)

### Community 3 - "retry.py"
Cohesion: 0.13
Nodes (37): _anthropic_custom_id_for_key(), _append_retry_to_source_metadata(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _build_retry_gemini_request_line(), _chunk_label(), _count_requests_file(), _extract_location_from_batch_name() (+29 more)

### Community 4 - "app/schemas.py"
Cohesion: 0.17
Nodes (17): apply_schema_fields(), _canonical_json(), dataset_schema_field_paths(), _definitions(), _field_type(), flatten_schema_fields(), _locate_property(), _object_node() (+9 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.08
Nodes (24): BooleanVar, Canvas, Frame, IntVar, Label, LabelFrame, Misc, main() (+16 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (49): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+41 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (37): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+29 more)

### Community 8 - "test_data_inspection.py"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 9 - "status.py"
Cohesion: 0.10
Nodes (40): _batch_model_progress(), _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job(), _count_gemini_prediction_rows(), _count_jsonl_blob_lines() (+32 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.06
Nodes (42): Application services and desktop UI for PatientJournals., batch_run_provider(), find_dataset_near(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns… (+34 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "inspection.py"
Cohesion: 0.20
Nodes (28): Counter, main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, collect_files() (+20 more)

### Community 13 - "publication.py"
Cohesion: 0.07
Nodes (56): _atomic_write_json(), _cloud_version_number(), _CloudVersionObject, DatasetVersionPublication, _empty_ledger(), file_sha256(), _ledger_with_record(), _list_cloud_versions() (+48 more)

### Community 14 - "retrieve.py"
Cohesion: 0.09
Nodes (49): add_reproducibility_columns(), add_response_metadata_columns(), _anthropic_stop_reason(), _build_api_key_generation_config(), _dataset_content_type(), _download_from_vertex_gcs_output(), _effective_duplicate_strategy(), _expected_success_keys() (+41 more)

### Community 15 - "AppSettings"
Cohesion: 0.09
Nodes (42): DuplicateStrategy, list_local_dataset_library(), build_retrieve_command(), build_submit_command(), build_validation_command(), app_settings_path(), AppSettings, CommandSpec (+34 more)

### Community 16 - "test_validation_sampling.py"
Cohesion: 0.17
Nodes (3): Validation UI and reporting commands., test_balanced_ucb_prioritizes_under_sampled_schema_field(), test_random_sampling_uses_unvalidated_datapoints()

### Community 17 - "ValidatorApp"
Cohesion: 0.15
Nodes (7): Button, Entry, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 18 - "schema_specialists"
Cohesion: 0.18
Nodes (20): _build_anthropic_manifest_lines(), generate_data(), Any, BaseModel, merge_specialist_metadata(), merge_specialist_payloads(), Any, BaseModel (+12 more)

### Community 19 - "validation/cli.py"
Cohesion: 0.14
Nodes (25): Random, build_validation_datapoints(), choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type() (+17 more)

### Community 20 - "access.py"
Cohesion: 0.15
Nodes (19): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+11 more)

### Community 21 - "JobStore"
Cohesion: 0.13
Nodes (9): Connection, Row, JobStore, _json_dumps(), _json_loads(), SQLite-backed app state for jobs. Operational run folders are artifacts only.…, Small in-process background task runner with SQLite task state., TaskRunner (+1 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.10
Nodes (44): collect_valid_outputs_from_jsonl_sources(), GeminiOutputParseResult, iter_gemini_jsonl_results(), normalize_output_key(), parse_gemini_output_record(), response_has_value(), _chosen_token_logprobs(), _collect_leaf_value_spans() (+36 more)

### Community 23 - "image_name_from_reference"
Cohesion: 0.19
Nodes (17): copy_dataset_rows_for_image_names(), copy_dataset_rows_for_keys(), load_dataset_image_coverage(), load_dataset_key_coverage(), normalize_dataset_image_name(), _normalize_output_format(), Path, duplicate_image_names() (+9 more)

### Community 24 - "test_subagents.py"
Cohesion: 0.17
Nodes (21): gemini_finish_reason(), Return the normalized first-candidate finish reason, if present., _anthropic_metadata(), combine_subagent_jsonl_sources(), CombinedSubagentOutputs, Path, Validate specialist results and join them into ordinary page records., _request_key_and_metadata() (+13 more)

### Community 25 - "submit_requests.py"
Cohesion: 0.21
Nodes (16): ocr_context_for_blob(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_request_line(), _build_request_lines(), _guess_mime_type() (+8 more)

### Community 26 - "routing.py"
Cohesion: 0.06
Nodes (65): candidate_sha256(), Any, Canonical unflattened extraction candidates for second-pass validation., Keep reproducibility/provenance only; never carry model thoughts forward., sanitize_extraction_metadata(), _build_thresholds(), _control_sample_sha256(), decide_candidate_route() (+57 more)

### Community 27 - "RetrieveBatchResult"
Cohesion: 0.50
Nodes (4): RetrieveBatchResult, Bridge batch.retrieve -> batch.verify for the direct CLI workflow., _record_candidate_retrieval_for_cli(), BatchResultService

### Community 28 - "ocr.py"
Cohesion: 0.14
Nodes (20): Protocol, _break_name(), _configured_backend(), detect_configured_ocr(), detect_configured_ocr_batch(), detect_ocr(), extract_google_vision_lines(), GoogleVisionOcrBackend (+12 more)

### Community 29 - "BrowserValidationSession"
Cohesion: 0.21
Nodes (6): BrowserValidationSession, Any, Server-side validation state for the browser validator., ValidationImageRef, _score_for_label(), _stringify_value()

### Community 30 - "PageCandidateRecord"
Cohesion: 0.19
Nodes (17): _candidate_schema_identity(), _copy_or_download_candidate_artifact(), _download_gcs_file(), ModelValidationSubmitRequest, _parse_gcs_uri(), Client, Recompute every route and bind the exact decision set for validation., _resolve_extraction_schema() (+9 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - "browser.py"
Cohesion: 0.18
Nodes (15): ImageSource, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Path, SamplingMode, _safe_user() (+7 more)

### Community 33 - "ocr_context.py"
Cohesion: 0.26
Nodes (15): _cache_key(), CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), _PendingOcr, prepare_ocr_metadata_for_blob(), prepare_ocr_metadata_for_blobs() (+7 more)

### Community 34 - "OcrDocument"
Cohesion: 0.14
Nodes (16): OcrDocument, OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., OCR derived from, and cryptographically bound to, one image payload., FakeBucket, FakeOcrBackend, _png_bytes(), _symbol() (+8 more)

### Community 35 - "submit.py"
Cohesion: 0.05
Nodes (80): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), BatchChunkPlan, BatchCollectOutputsRequest, BatchSubmitPlan, BatchSubmitRequest, BatchSubmitService, Namespace (+72 more)

### Community 36 - "PatientJournals research pipeline"
Cohesion: 0.11
Nodes (17): 10. Operational settings, 1. Input selection and image preparation, 2. Positional OCR, 3. First-pass extraction batch, 4. Schema-specialist decomposition, 5. Retrieval and deterministic sweep, 6. Exact evidence bindings for final validation, 7. Candidate-aware final verifier (+9 more)

### Community 37 - "retrieve_model_validation"
Cohesion: 0.10
Nodes (33): _await_completion(), _batch_job_state(), _batch_job_successful(), _download_from_mldev_output(), _gemini_output_reference(), _get_batch_job(), Client, _apply_one_patch() (+25 more)

### Community 38 - ".name"
Cohesion: 0.20
Nodes (16): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_library(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), ImageAccessService, Any (+8 more)

### Community 39 - "Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?, Source Nodes

### Community 40 - "generate.py"
Cohesion: 0.20
Nodes (20): process_file(), ProcessedFileResult, is_fatal_api_error(), is_retryable_api_error(), BaseException, retry_delay_seconds(), append_processing_record(), base_image_record() (+12 more)

### Community 41 - "Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/, Source Nodes

### Community 42 - "AGENTS.md"
Cohesion: 0.40
Nodes (4): Batch-first architecture, graphify, Pipeline documentation, Prompt ownership

### Community 43 - "analysis.py"
Cohesion: 0.60
Nodes (10): _add_accuracy_scores(), load_validations(), main(), plot_label_distribution(), plot_nested_accuracy(), plot_overall_accuracy(), plot_top_level_accuracy(), DataFrame (+2 more)

### Community 44 - "config/schemas.py"
Cohesion: 0.07
Nodes (51): BaseHTTPRequestHandler, FieldConfidenceByPointer, list_batch_model_options(), list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), Models suitable for provider batch jobs, including model validation. (+43 more)

### Community 45 - "model_client.py"
Cohesion: 0.15
Nodes (20): _build_request_config(), _build_provider_client(), create_local_model_client(), _extract_anthropic_response_text(), _extract_openai_response_text(), _import_anthropic_async_client(), _import_openai_async_client(), LocalGenerationResult (+12 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 48 - "resolve_batch_run_readiness"
Cohesion: 0.17
Nodes (13): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), poll_local_batch_states(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini…, Query the batch API once and aggregate chunk states into a job-level status.… (+5 more)

### Community 49 - "test_schema_management.py"
Cohesion: 0.12
Nodes (15): fixture, model_from_json_schema(), Any, Build a Pydantic model for an immutable app-managed JSON Schema version. The…, test_absolute_local_image_hint_does_not_become_a_cloud_object(), test_dashboard_completeness_reports_leafs_not_parent_objects(), test_dashboard_infers_legacy_schema_and_includes_fully_missing_leafs(), test_managed_json_schema_builds_a_strict_runtime_model() (+7 more)

### Community 58 - "datasets.py"
Cohesion: 0.16
Nodes (27): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+19 more)

### Community 59 - "patientjournals/tasks.py"
Cohesion: 0.25
Nodes (22): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+14 more)

### Community 60 - "job_store.py"
Cohesion: 0.29
Nodes (5): _dataset_publication_idempotency_key(), _model_validation_idempotency_key(), Use the shared publisher's portable, content-bound identity., _submit_root_for_run_dir(), write_json_file()

### Community 61 - "jobs.py"
Cohesion: 0.07
Nodes (78): _api_recovery_error_rows(), _api_recovery_error_summary(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, cancel_batch_run(), command_overrides_for_run() (+70 more)

### Community 62 - "PageCandidateWriter"
Cohesion: 0.33
Nodes (3): PageCandidateWriter, Buffered JSONL writer that enforces one canonical record per page key., write_page_candidates()

### Community 63 - "bucket.py"
Cohesion: 0.27
Nodes (18): _blob_extension(), _blob_size(), _bucket_depth(), _bucket_parent(), _bucket_relative_name(), _content_type_format_issue(), _extension_format_issue(), _folder_names_from_blob() (+10 more)

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

### Community 65 - "test_batch_verify.py"
Cohesion: 0.05
Nodes (49): PageModelValidation, model_validator, ValidationIssuePatch, read_page_candidates(), _identity(), _ImmutableArtifactBucket, _input_record(), _input_record_without_ocr() (+41 more)

### Community 66 - "collect_outputs.py"
Cohesion: 0.23
Nodes (17): collect_outputs(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines(), _list_page_image_names(), list_prediction_output_blobs() (+9 more)

### Community 67 - "Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage., Source Nodes

### Community 68 - "_extraction_model_for_snapshot"
Cohesion: 0.17
Nodes (13): _assert_dynamic_schema_supported(), _canonical_json_sha256(), _extraction_model_for_snapshot(), _ExtractionSchemaSnapshot, Reuse built-in Pydantic validators only when their schema is exact., Fail closed instead of silently ignoring JSON Schema validation rules., verification_prompt_hash(), _write_submit_metadata() (+5 more)

### Community 69 - "config/__init__.py"
Cohesion: 0.06
Nodes (26): Batch upload, submission, status, and retrieval commands., Configuration, schema, and model registry., _apply_external_json_config(), Config, _default_api_key(), _load_provider_api_keys(), load_local_api_keys(), local_secrets_path() (+18 more)

### Community 72 - "_resolve_final_validation_policy"
Cohesion: 0.14
Nodes (15): _candidate_source_run_id(), _FinalValidationPolicySnapshot, _is_not_found(), _metadata_source_run_id(), Exception, Load the cloud-anchored policy; fail closed for legacy unanchored runs., Resolve immutable run policy without accepting retrieval-time promotion., Bind all candidates to the supplied extraction run before submission. (+7 more)

### Community 73 - "_recover_one_missing_page_via_api_key"
Cohesion: 0.24
Nodes (10): _api_key_recovery_failure_reason(), _bound_ocr_context_for_recovery(), _compact_exception_text(), _generate_recovery_response(), _generation_bound_blob(), _guess_blob_mime_type(), BaseException, Render the exact OCR sidecar recorded for the extraction request. (+2 more)

### Community 75 - "preprocess.py"
Cohesion: 0.16
Nodes (19): Image, Render every line with minimal syntax and no repeated field names., render_ocr_context(), crop_margins(), enhance_contrast(), image_to_bytes(), load_image(), prepare_page() (+11 more)

### Community 77 - "run_layout.py"
Cohesion: 0.17
Nodes (19): category_root(), classify_legacy_dir(), _created_at_from_name(), document_existing_runs(), iter_all_run_dirs(), Path, Central conventions for the runs/ output folder. All job output lives under a…, Write a README documenting the runs/ layout. Returns its path. (+11 more)

### Community 78 - "prepare_ocr.py"
Cohesion: 0.24
Nodes (11): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+3 more)

### Community 79 - "Path"
Cohesion: 0.21
Nodes (20): _anthropic_custom_id_for_key(), _arg_batch_names(), _download_from_anthropic_output(), _extract_batch_names_from_payload(), _find_submit_run_dir(), _latest_batch_job_file(), _normalize_key(), _output_destinations_from_submit_run() (+12 more)

### Community 80 - "CloudBlobIdentity"
Cohesion: 0.11
Nodes (26): CloudBlobIdentity, _load_bound_ocr_evidence(), Reload and validate the exact image/OCR evidence in an extraction manifest., _binding_for_record(), _identity_matches(), input_manifest_record_for_blob(), InputImageManifestRecord, _mime_type() (+18 more)

### Community 81 - "_ImmutableArtifactBlob"
Cohesion: 0.22
Nodes (3): _FakePreconditionFailure, _ImmutableArtifactBlob, Exception

### Community 82 - "verify.py"
Cohesion: 0.09
Nodes (53): _get_anthropic_client(), _anthropic_custom_id_for_key(), _vertex_compatible_schema(), _anthropic_request(), _anthropic_thinking_config(), _chunk_file_name(), _create_verification_run_dir(), _final_validation_policy_anchor_object_name() (+45 more)

### Community 83 - "get_batch_client"
Cohesion: 0.24
Nodes (15): _candidate_model_ids(), _check_model_ids(), _ConfigSnapshot, _iter_models(), main(), _norm(), _parse_args(), _print_models() (+7 more)

### Community 84 - "JobRegistry"
Cohesion: 0.39
Nodes (4): JobRegistry, RegisteredJob, start_command(), test_job_registry_roundtrip()

### Community 85 - "_build_anthropic_batch_requests_for_retry"
Cohesion: 0.50
Nodes (5): _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests_for_retry(), Any, timedelta

### Community 86 - "prompts.py"
Cohesion: 0.20
Nodes (8): build_subagent_prompt(), ocr_context_header(), _prompt(), Single source of truth for non-schema model prompt text. Edit page, sub-agent,…, Render the compact role brief shared by batch, retry, and local paths., Normalize source indentation without altering intentional line breaks., test_ocr_context_uses_central_header_template(), test_subagent_prompt_is_rendered_from_central_prompt_definitions()

## Knowledge Gaps
- **62 isolated node(s):** `patientjournals`, `UploadProfile`, `Batch-first architecture`, `Pipeline documentation`, `Prompt ownership` (+57 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `CloudBlobIdentity` (2× useful, score=1.999765272)
- `LocalModelClient` (2× useful, score=1.997808494) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `WorkflowService`, `JobRegistry`, `jobs.py`, `AppSettings`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `JobStore` connect `JobStore` to `Path`, `WorkflowService`, `app/schemas.py`, `test_app_architecture.py`, `publication.py`, `AppSettings`, `test_schema_management.py`, `datasets.py`, `job_store.py`, `jobs.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `WorkflowService` connect `WorkflowService` to `app/schemas.py`, `PatientJournalsApp`, `.name`, `BrowserValidationManager`, `test_app_architecture.py`, `config/schemas.py`, `AppSettings`, `verify.py`, `JobStore`, `jobs.py`, `PageCandidateRecord`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 55 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 55 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `WorkflowService` (e.g. with `PatientJournalsApp` and `AppHandler`) actually correct?**
  _`WorkflowService` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `retrieve_batch()` (e.g. with `.recover_missing_with_api()` and `output_schema_name()`) actually correct?**
  _`retrieve_batch()` has 3 INFERRED edges - model-reasoned connections that need verification._