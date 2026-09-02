# Graph Report - PatientJournals  (2026-09-01)

## Corpus Check
- 111 files · ~168,569 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2204 nodes · 6570 edges · 90 communities (77 shown, 13 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 407 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ac70ccd6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- JobStore
- tools.py
- WorkflowService
- retry.py
- catalog.py
- PatientJournalsApp
- dashboard.py
- upload.py
- prompts.py
- status.py
- test_app_architecture.py
- PatientJournals Conda Environment
- inspection.py
- publication.py
- retrieve.py
- AppSettings
- _recover_missing_pages_via_api_key
- settings.py
- schema_specialists
- validation/cli.py
- access.py
- submit_inputs.py
- response_parsing.py
- collect_outputs.py
- ValidatorApp
- submit_requests.py
- routing.py
- test_batch_service.py
- ocr.py
- model_client.py
- resolve_batch_run_readiness
- Journal
- BrowserValidationSession
- ocr_context.py
- test_ocr.py
- submit.py
- PatientJournals research pipeline
- Path
- ImageAccessService
- Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?
- FakeBlob
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- PageCandidateRecord
- get_batch_client
- test_schema_management.py
- Journal
- config/schemas.py
- bucket.py
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
- submit_model_validation
- jobs.py
- test_batch_submit_inputs.py
- .name
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.
- test_batch_verify.py
- batch/__init__.py
- Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage.
- config/__init__.py
- test_batch_retrieve_recovery.py
- test_data_inspection.py
- _prepare_validation_page
- PageCandidateWriter
- choose_balanced_ucb_datapoint
- batch/service.py
- preprocess.py
- workflows.py
- output_handler.py
- prepare_ocr.py
- test_validation_sampling.py
- input_manifest.py
- _ImmutableArtifactBlob
- verify.py
- AppHandler
- _recover_one_missing_page_via_api_key
- CloudBlobIdentity
- OcrDocument
- _build_anthropic_batch_requests_for_retry
- test_extraction_image_bindings.py
- .__init__

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 78 edges
2. `AppSettings` - 72 edges
3. `WorkflowService` - 72 edges
4. `retrieve_batch()` - 58 edges
5. `submit_batch()` - 52 edges
6. `PageCandidateRecord` - 52 edges
7. `retrieve_model_validation()` - 51 edges
8. `PatientJournalsApp` - 50 edges
9. `submit_model_validation()` - 46 edges
10. `_Bucket` - 42 edges

## Surprising Connections (you probably didn't know these)
- `test_job_store_persists_background_tasks()` --uses--> `JobStore`  [INFERRED]
  tests/test_app_workflows.py → src/patientjournals/app/job_store.py
- `test_recover_dataset_gaps_only_targets_missing_pages()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_recover_dataset_gaps_reports_zero_row_api_completion()` --uses--> `AppSettings`  [INFERRED]
  tests/test_app_architecture.py → src/patientjournals/app/models.py
- `test_direct_retrieve_records_candidate_location_on_submit_run()` --uses--> `RetrieveBatchResult`  [INFERRED]
  tests/test_batch_verify.py → src/patientjournals/batch/results.py
- `test_failed_page_retry_can_split_into_multiple_chunks()` --calls--> `_submit_failed_pages_as_batch()`  [INFERRED]
  tests/test_batch_retrieve_recovery.py → src/patientjournals/batch/retry.py

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

## Communities (90 total, 13 thin omitted)

### Community 0 - "JobStore"
Cohesion: 0.06
Nodes (45): Connection, Row, _copy_dataset_into_job(), _dataset_files(), _dataset_publication_idempotency_key(), _file_sha256(), JobStore, _json_dumps() (+37 more)

### Community 1 - "tools.py"
Cohesion: 0.08
Nodes (53): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, _emit(), _input_without_existing(), LocalRunProgress (+45 more)

### Community 2 - "WorkflowService"
Cohesion: 0.12
Nodes (15): _apply_runtime_overrides(), poll_local_batch_states(), One-shot API poll mapping each unfinished local batch run_dir to a live status.…, _restore_runtime_overrides(), run_local_draft_direct(), command_override_payload(), Any, Submit the candidate-aware verifier batch for a retrieved extraction. (+7 more)

### Community 3 - "retry.py"
Cohesion: 0.13
Nodes (37): _anthropic_custom_id_for_key(), _append_retry_to_source_metadata(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _chunk_label(), _count_requests_file(), _extract_location_from_batch_name(), _guess_key_mime_type() (+29 more)

### Community 4 - "catalog.py"
Cohesion: 0.15
Nodes (19): list_batch_model_options(), list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), Models suitable for provider batch jobs, including model validation., resolve_schema_class(), ModelOption (+11 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.07
Nodes (30): BooleanVar, Button, Canvas, Frame, IntVar, Label, LabelFrame, Misc (+22 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (49): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+41 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (37): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+29 more)

### Community 8 - "prompts.py"
Cohesion: 0.20
Nodes (8): build_subagent_prompt(), ocr_context_header(), _prompt(), Single source of truth for non-schema model prompt text. Edit page, sub-agent,…, Render the compact role brief shared by batch, retry, and local paths., Normalize source indentation without altering intentional line breaks., test_ocr_context_uses_central_header_template(), test_subagent_prompt_is_rendered_from_central_prompt_definitions()

### Community 9 - "status.py"
Cohesion: 0.06
Nodes (58): _aggregate_state_lines(), _anthropic_model_progress(), _batch_summary(), _cancel_batch_job(), _count_gemini_prediction_rows(), _count_jsonl_blob_lines(), _extract_batch_names_from_payload(), _extract_location_from_batch_name() (+50 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.06
Nodes (42): batch_run_provider(), list_submit_jobs(), Return the provider ("gemini"/"anthropic") recorded for a submit run., Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns…, One row per batch submission from the authoritative app store., Retrieve a submitted batch in-process and record the result on the submit run.… (+34 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "inspection.py"
Cohesion: 0.20
Nodes (28): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, collect_files(), configured_image_extensions() (+20 more)

### Community 13 - "publication.py"
Cohesion: 0.07
Nodes (56): _atomic_write_json(), _cloud_version_number(), _CloudVersionObject, DatasetVersionPublication, _empty_ledger(), file_sha256(), _ledger_with_record(), _list_cloud_versions() (+48 more)

### Community 14 - "retrieve.py"
Cohesion: 0.11
Nodes (36): add_response_metadata_columns(), RetrieveBatchResult, _anthropic_stop_reason(), _arg_batch_names(), _download_from_anthropic_output(), _effective_duplicate_strategy(), _expected_success_keys(), _extract_anthropic_response_metadata() (+28 more)

### Community 15 - "AppSettings"
Cohesion: 0.09
Nodes (38): DuplicateStrategy, build_retrieve_command(), build_submit_command(), app_settings_path(), AppSettings, Path, SubmitJobDraft, _coerce_settings() (+30 more)

### Community 16 - "_recover_missing_pages_via_api_key"
Cohesion: 0.13
Nodes (22): _build_api_key_generation_config(), _dataset_content_type(), _download_bound_request_artifact(), _download_from_vertex_gcs_output(), _FirstPassRecoveryEvidence, _normalize_prefix(), _parse_gcs_uri(), Any (+14 more)

### Community 17 - "settings.py"
Cohesion: 0.19
Nodes (13): _apply_external_json_config(), Config, _default_api_key(), _load_provider_api_keys(), load_local_api_keys(), local_secrets_path(), Path, save_local_api_key() (+5 more)

### Community 18 - "schema_specialists"
Cohesion: 0.12
Nodes (35): _anthropic_metadata(), combine_subagent_jsonl_sources(), CombinedSubagentOutputs, Path, Validate specialist results and join them into ordinary page records., _request_key_and_metadata(), write_combined_subagent_outputs(), generate_data() (+27 more)

### Community 19 - "validation/cli.py"
Cohesion: 0.21
Nodes (17): build_validation_datapoints(), eligible_flat_fields(), flatten_row(), _get_field_type(), _is_metadata_field(), _is_missing_value(), _is_validation_schema_field(), _parse_corrected_value() (+9 more)

### Community 20 - "access.py"
Cohesion: 0.16
Nodes (18): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+10 more)

### Community 21 - "submit_inputs.py"
Cohesion: 0.15
Nodes (26): _allowed_extensions(), _apply_fp_mode_to_blobs(), _apply_fp_mode_to_pdf_paths(), _apply_image_name_restriction(), _apply_year_filter_to_blobs(), _assert_gcs_input_source(), _configured_year_filter_tokens(), _dedupe_blob_image_names() (+18 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.09
Nodes (46): collect_valid_outputs_from_jsonl_sources(), gemini_finish_reason(), GeminiOutputParseResult, iter_gemini_jsonl_results(), normalize_output_key(), parse_gemini_output_record(), Return the normalized first-candidate finish reason, if present., response_has_value() (+38 more)

### Community 23 - "collect_outputs.py"
Cohesion: 0.18
Nodes (23): Counter, collect_outputs(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines(), _list_page_image_names() (+15 more)

### Community 24 - "ValidatorApp"
Cohesion: 0.15
Nodes (6): Entry, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 25 - "submit_requests.py"
Cohesion: 0.22
Nodes (19): ocr_context_for_blob(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_anthropic_manifest_lines(), _build_request_line(), _build_request_lines(), _guess_mime_type() (+11 more)

### Community 26 - "routing.py"
Cohesion: 0.07
Nodes (59): _build_thresholds(), _control_sample_sha256(), decide_candidate_route(), _decide_candidate_route_with_thresholds(), DeterministicRoutingDecision, _force_control_sample(), _frontpage_rule_ids(), _is_frontpage_model() (+51 more)

### Community 27 - "test_batch_service.py"
Cohesion: 0.13
Nodes (9): BatchCollectOutputsRequest, BatchSubmitRequest, Namespace, test_batch_collect_outputs_request_namespace(), test_batch_retrieve_request_namespace(), test_batch_submit_request_namespace(), test_complete_submission_ignores_stale_sample_values(), test_rerun_restores_recorded_transport_semantics() (+1 more)

### Community 28 - "ocr.py"
Cohesion: 0.14
Nodes (19): Protocol, _break_name(), _configured_backend(), detect_configured_ocr(), detect_configured_ocr_batch(), detect_ocr(), extract_google_vision_lines(), GoogleVisionOcrBackend (+11 more)

### Community 29 - "model_client.py"
Cohesion: 0.15
Nodes (21): _build_retry_gemini_request_line(), _build_request_config(), _build_provider_client(), create_local_model_client(), _extract_anthropic_response_text(), _extract_openai_response_text(), _import_anthropic_async_client(), _import_openai_async_client() (+13 more)

### Community 30 - "resolve_batch_run_readiness"
Cohesion: 0.20
Nodes (11): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini…, Query the batch API once and aggregate chunk states into a job-level status.…, resolve_batch_run_readiness() (+3 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 33 - "ocr_context.py"
Cohesion: 0.26
Nodes (15): _cache_key(), CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), _PendingOcr, prepare_ocr_metadata_for_blob(), prepare_ocr_metadata_for_blobs() (+7 more)

### Community 34 - "test_ocr.py"
Cohesion: 0.19
Nodes (17): OcrAttempt, OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., FakeBucket, FakeOcrBackend, _png_bytes(), _symbol(), test_batch_ocr_preparation_creates_generation_bound_reusable_sidecar() (+9 more)

### Community 35 - "submit.py"
Cohesion: 0.13
Nodes (38): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), _batch_state_and_success(), _blob_names_sha256(), _build_chunk_entry(), _build_rerun_entries(), _chunk_label(), _chunk_requests_file_name() (+30 more)

### Community 36 - "PatientJournals research pipeline"
Cohesion: 0.11
Nodes (18): 10. Operational settings, 1. Input selection and image preparation, 2. Positional OCR, 3. First-pass extraction batch, 4. Schema-specialist decomposition, 5. Retrieval and deterministic sweep, 6. Exact evidence bindings for final validation, 7. Candidate-aware final verifier (+10 more)

### Community 37 - "Path"
Cohesion: 0.22
Nodes (21): _anthropic_custom_id_for_key(), _extract_batch_names_from_payload(), _find_submit_run_dir(), _load_request_artifacts(), _local_request_file_path(), _normalize_key(), _output_destinations_from_submit_run(), _provider_from_batch_names() (+13 more)

### Community 38 - "ImageAccessService"
Cohesion: 0.38
Nodes (4): ImageAccessService, Any, Path, Short-lived image links for dataset inspection and submission previews.

### Community 39 - "Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?, Source Nodes

### Community 41 - "Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/, Source Nodes

### Community 42 - "AGENTS.md"
Cohesion: 0.40
Nodes (4): Batch-first architecture, graphify, Pipeline documentation, Prompt ownership

### Community 43 - "analysis.py"
Cohesion: 0.60
Nodes (10): _add_accuracy_scores(), load_validations(), main(), plot_label_distribution(), plot_nested_accuracy(), plot_overall_accuracy(), plot_top_level_accuracy(), DataFrame (+2 more)

### Community 44 - "PageCandidateRecord"
Cohesion: 0.19
Nodes (17): _candidate_schema_identity(), _copy_or_download_candidate_artifact(), _download_gcs_file(), ModelValidationSubmitRequest, _parse_gcs_uri(), Client, Recompute every route and bind the exact decision set for validation., _resolve_extraction_schema() (+9 more)

### Community 45 - "get_batch_client"
Cohesion: 0.24
Nodes (15): _candidate_model_ids(), _check_model_ids(), _ConfigSnapshot, _iter_models(), main(), _norm(), _parse_args(), _print_models() (+7 more)

### Community 46 - "test_schema_management.py"
Cohesion: 0.11
Nodes (17): fixture, Application services and desktop UI for PatientJournals., model_from_json_schema(), Any, Build a Pydantic model for an immutable app-managed JSON Schema version. The…, test_absolute_local_image_hint_does_not_become_a_cloud_object(), test_dashboard_completeness_reports_leafs_not_parent_objects(), test_dashboard_infers_legacy_schema_and_includes_fully_missing_leafs() (+9 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 48 - "config/schemas.py"
Cohesion: 0.20
Nodes (17): Address, Age, Bottom, Diagnoses, HospitalStay, list_output_schemas(), PageLine, Patient (+9 more)

### Community 49 - "bucket.py"
Cohesion: 0.31
Nodes (15): _blob_extension(), _blob_size(), _bucket_depth(), _bucket_parent(), _bucket_relative_name(), _content_type_format_issue(), _extension_format_issue(), _folder_names_from_blob() (+7 more)

### Community 58 - "datasets.py"
Cohesion: 0.15
Nodes (29): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+21 more)

### Community 59 - "patientjournals/tasks.py"
Cohesion: 0.25
Nodes (22): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+14 more)

### Community 60 - "submit_model_validation"
Cohesion: 0.07
Nodes (35): _get_anthropic_client(), _anthropic_custom_id_for_key(), _vertex_compatible_schema(), _anthropic_request(), _anthropic_thinking_config(), _assert_dynamic_schema_supported(), _canonical_json_sha256(), _chunk_file_name() (+27 more)

### Community 61 - "jobs.py"
Cohesion: 0.07
Nodes (77): _api_recovery_error_rows(), _api_recovery_error_summary(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_model_progress(), _batch_submit_namespace(), BatchSubmitOutcome, build_validation_command() (+69 more)

### Community 62 - "test_batch_submit_inputs.py"
Cohesion: 0.42
Nodes (5): FakeBlob, FakeBucket, test_list_input_blobs_raises_when_restriction_matches_nothing(), test_list_input_blobs_scopes_to_restricted_image_names(), test_list_input_blobs_skips_duplicate_image_names_with_audit()

### Community 63 - ".name"
Cohesion: 0.25
Nodes (18): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), Count the deduplicated cloud population without signing preview URLs., _list_page_keys(), build_storage_bucket() (+10 more)

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

### Community 65 - "test_batch_verify.py"
Cohesion: 0.06
Nodes (43): PageModelValidation, model_validator, ValidationIssuePatch, _identity(), _ImmutableArtifactBucket, _input_record(), _input_record_without_ocr(), _MissingPolicyBlob (+35 more)

### Community 67 - "Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage., Source Nodes

### Community 68 - "config/__init__.py"
Cohesion: 0.17
Nodes (22): add_reproducibility_columns(), Configuration, schema, and model registry., process_file(), ProcessedFileResult, is_fatal_api_error(), is_retryable_api_error(), BaseException, retry_delay_seconds() (+14 more)

### Community 69 - "test_batch_retrieve_recovery.py"
Cohesion: 0.11
Nodes (13): FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency(), test_failed_page_retry_can_split_into_multiple_chunks() (+5 more)

### Community 70 - "test_data_inspection.py"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 71 - "_prepare_validation_page"
Cohesion: 0.25
Nodes (9): _anthropic_signed_url_expiration(), timedelta, _mime_type_for_name(), _prepare_validation_page(), build_model_validation_prompt(), _compact_json(), Any, Build the compact, provider-independent verifier request text. (+1 more)

### Community 72 - "PageCandidateWriter"
Cohesion: 0.11
Nodes (21): _candidate_source_run_id(), Bind all candidates to the supplied extraction run before submission., _validate_candidate_source_run(), candidate_sha256(), PageCandidateWriter, Any, Path, Canonical unflattened extraction candidates for second-pass validation. (+13 more)

### Community 73 - "choose_balanced_ucb_datapoint"
Cohesion: 0.23
Nodes (10): Random, choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), _is_validated(), _score_for_label(), _sum_for_sampling_group(), validation_sampling_group_key() (+2 more)

### Community 74 - "batch/service.py"
Cohesion: 0.28
Nodes (10): BatchChunkPlan, BatchSubmitPlan, BatchSubmitService, _filter_blobs_missing_from_dataset(), Namespace, _resolve_downscale(), _resolve_num_batches(), _resolve_sample_seed() (+2 more)

### Community 75 - "preprocess.py"
Cohesion: 0.23
Nodes (14): Image, crop_margins(), enhance_contrast(), image_to_bytes(), load_image(), prepare_page(), preprocess_image(), preprocess_image_with_metadata() (+6 more)

### Community 76 - "workflows.py"
Cohesion: 0.18
Nodes (13): cancel_batch_run(), list_batch_chunks(), list_batch_chunks_with_state(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, Resubmit the requests that did not succeed as a fresh batch. Clears the…, resubmit_failed_requests(), _count_images(), _image_extensions() (+5 more)

### Community 77 - "output_handler.py"
Cohesion: 0.40
Nodes (13): FieldConfidenceByPointer, FrontPage, identity_columns(), _build_confidence_tree(), data_to_rows(), default_rows(), _escape_pointer_segment(), _has_field_confidence() (+5 more)

### Community 78 - "prepare_ocr.py"
Cohesion: 0.24
Nodes (11): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+3 more)

### Community 79 - "test_validation_sampling.py"
Cohesion: 0.17
Nodes (3): Validation UI and reporting commands., test_balanced_ucb_prioritizes_under_sampled_schema_field(), test_random_sampling_uses_unvalidated_datapoints()

### Community 80 - "input_manifest.py"
Cohesion: 0.14
Nodes (20): _load_bound_ocr_evidence(), Reload and validate the exact image/OCR evidence in an extraction manifest., _binding_for_record(), _identity_matches(), input_manifest_record_for_blob(), InputImageManifestRecord, _mime_type(), _normalized_prefix() (+12 more)

### Community 81 - "_ImmutableArtifactBlob"
Cohesion: 0.22
Nodes (3): _FakePreconditionFailure, _ImmutableArtifactBlob, Exception

### Community 82 - "verify.py"
Cohesion: 0.08
Nodes (65): _await_completion(), _batch_job_state(), _batch_job_successful(), _download_from_mldev_output(), _gemini_output_reference(), _get_batch_job(), Client, _apply_one_patch() (+57 more)

### Community 84 - "_recover_one_missing_page_via_api_key"
Cohesion: 0.24
Nodes (10): _api_key_recovery_failure_reason(), _bound_ocr_context_for_recovery(), _compact_exception_text(), _generate_recovery_response(), _generation_bound_blob(), _guess_blob_mime_type(), BaseException, Render the exact OCR sidecar recorded for the extraction request. (+2 more)

### Community 86 - "OcrDocument"
Cohesion: 0.19
Nodes (7): OcrDocument, OCR derived from, and cryptographically bound to, one image payload., Render every line with minimal syntax and no repeated field names., render_ocr_context(), PreparedPage, The exact image payload and OCR context supplied to a model request., test_ocr_prompt_format_contains_all_text_without_json_field_overhead()

### Community 87 - "_build_anthropic_batch_requests_for_retry"
Cohesion: 0.32
Nodes (8): _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests_for_retry(), Any, timedelta, decode_specialist_request_key(), page_key_from_request_key(), test_specialist_request_key_round_trip()

### Community 88 - "test_extraction_image_bindings.py"
Cohesion: 0.53
Nodes (5): _identity(), _input_record(), test_anthropic_first_pass_signed_url_is_generation_qualified(), test_gemini_first_pass_uses_write_once_staged_generation(), test_image_manifest_can_pin_exact_bytes_without_ocr()

### Community 89 - ".__init__"
Cohesion: 0.16
Nodes (13): ImageSource, BrowserValidationManager, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Any, Path (+5 more)

## Knowledge Gaps
- **63 isolated node(s):** `patientjournals`, `UploadProfile`, `Batch-first architecture`, `Pipeline documentation`, `Prompt ownership` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `CloudBlobIdentity` (2× useful, score=1.999765272)
- `LocalModelClient` (2× useful, score=1.997808494) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `datasets.py`, `WorkflowService`, `jobs.py`, `AppSettings`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `JobStore` connect `JobStore` to `WorkflowService`, `test_app_architecture.py`, `workflows.py`, `publication.py`, `test_schema_management.py`, `AppSettings`, `datasets.py`, `jobs.py`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `WorkflowService` connect `WorkflowService` to `JobStore`, `PatientJournalsApp`, `ImageAccessService`, `test_app_architecture.py`, `workflows.py`, `PageCandidateRecord`, `AppSettings`, `verify.py`, `AppHandler`, `.__init__`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `WorkflowService` (e.g. with `PatientJournalsApp` and `AppHandler`) actually correct?**
  _`WorkflowService` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `retrieve_batch()` (e.g. with `.recover_missing_with_api()` and `output_schema_name()`) actually correct?**
  _`retrieve_batch()` has 3 INFERRED edges - model-reasoned connections that need verification._