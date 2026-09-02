# Graph Report - PatientJournals  (2026-09-01)

## Corpus Check
- 111 files · ~167,749 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2200 nodes · 6549 edges · 82 communities (71 shown, 11 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 403 edges (avg confidence: 0.92)
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
- image_name_from_reference
- PatientJournalsApp
- dashboard.py
- upload.py
- build_subagent_prompt
- status.py
- test_app_architecture.py
- PatientJournals Conda Environment
- inspection.py
- publication.py
- retrieve.py
- AppSettings
- run_layout.py
- settings.py
- schema_specialists
- validation/cli.py
- access.py
- submit_inputs.py
- response_parsing.py
- config/__init__.py
- ValidatorApp
- submit_requests.py
- routing.py
- batch/service.py
- ocr.py
- model_client.py
- resolve_batch_run_readiness
- Journal
- BrowserValidationSession
- ocr_context.py
- test_ocr.py
- submit.py
- PatientJournals research pipeline
- PageModelValidation
- ImageAccessService
- Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?
- FakeBlob
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- app/schemas.py
- test_schema_management.py
- Journal
- config/schemas.py
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
- _list_input_blobs
- .name
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.
- test_batch_verify.py
- batch/__init__.py
- Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage.
- generate.py
- test_batch_retrieve_recovery.py
- test_data_inspection.py
- _prepare_validation_page
- source_run_id_from_path
- preprocess.py
- workflows.py
- output_handler.py
- prepare_ocr.py
- CloudBlobIdentity
- _ImmutableArtifactBlob
- verify.py
- OcrDocument
- _build_anthropic_batch_requests_for_retry
- BrowserValidationManager
- browser.py

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 78 edges
2. `AppSettings` - 72 edges
3. `WorkflowService` - 72 edges
4. `retrieve_batch()` - 58 edges
5. `PageCandidateRecord` - 52 edges
6. `submit_batch()` - 51 edges
7. `retrieve_model_validation()` - 51 edges
8. `PatientJournalsApp` - 50 edges
9. `submit_model_validation()` - 46 edges
10. `_Bucket` - 41 edges

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

## Communities (82 total, 11 thin omitted)

### Community 0 - "JobStore"
Cohesion: 0.09
Nodes (27): Connection, Row, _copy_dataset_into_job(), _dataset_files(), _dataset_publication_idempotency_key(), _file_sha256(), JobStore, _json_dumps() (+19 more)

### Community 1 - "tools.py"
Cohesion: 0.08
Nodes (52): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, create_local_model_client(), _emit(), _input_without_existing() (+44 more)

### Community 2 - "WorkflowService"
Cohesion: 0.12
Nodes (15): _apply_runtime_overrides(), poll_local_batch_states(), One-shot API poll mapping each unfinished local batch run_dir to a live status.…, _restore_runtime_overrides(), run_local_draft_direct(), command_override_payload(), Any, Submit the candidate-aware verifier batch for a retrieved extraction. (+7 more)

### Community 3 - "retry.py"
Cohesion: 0.14
Nodes (34): _anthropic_custom_id_for_key(), _append_retry_to_source_metadata(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _chunk_label(), _count_requests_file(), _extract_location_from_batch_name(), _guess_key_mime_type() (+26 more)

### Community 4 - "image_name_from_reference"
Cohesion: 0.20
Nodes (17): copy_dataset_rows_for_image_names(), copy_dataset_rows_for_keys(), load_dataset_image_coverage(), load_dataset_key_coverage(), normalize_dataset_image_name(), _normalize_output_format(), Path, duplicate_image_names() (+9 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.07
Nodes (30): BooleanVar, Button, Canvas, Frame, IntVar, Label, LabelFrame, Misc (+22 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (49): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+41 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (37): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+29 more)

### Community 8 - "build_subagent_prompt"
Cohesion: 0.67
Nodes (3): build_subagent_prompt(), Render the compact role brief shared by batch, retry, and local paths., test_subagent_prompt_is_rendered_from_central_prompt_definitions()

### Community 9 - "status.py"
Cohesion: 0.05
Nodes (68): BaseHTTPRequestHandler, list_batch_model_options(), list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), Models suitable for provider batch jobs, including model validation., _batch_model_progress() (+60 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.07
Nodes (38): batch_run_provider(), find_dataset_near(), list_submit_jobs(), Return the provider ("gemini"/"anthropic") recorded for a submit run., Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns… (+30 more)

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
Cohesion: 0.05
Nodes (97): resolve_service_account_path(), add_reproducibility_columns(), add_response_metadata_columns(), RetrieveBatchResult, _anthropic_custom_id_for_key(), _anthropic_stop_reason(), _api_key_recovery_failure_reason(), _arg_batch_names() (+89 more)

### Community 15 - "AppSettings"
Cohesion: 0.08
Nodes (35): DuplicateStrategy, build_retrieve_command(), build_submit_command(), build_validation_command(), Retrieve a submitted batch in-process and record the result on the submit run.…, Resubmit the requests that did not succeed as a fresh batch. Clears the…, resubmit_failed_requests(), run_retrieve_direct() (+27 more)

### Community 16 - "run_layout.py"
Cohesion: 0.17
Nodes (19): category_root(), classify_legacy_dir(), _created_at_from_name(), document_existing_runs(), iter_all_run_dirs(), Path, Central conventions for the runs/ output folder. All job output lives under a…, Write a README documenting the runs/ layout. Returns its path. (+11 more)

### Community 17 - "settings.py"
Cohesion: 0.16
Nodes (16): resolve_schema_class(), resolve_output_schema(), _apply_external_json_config(), Config, _default_api_key(), _load_provider_api_keys(), load_local_api_keys(), local_secrets_path() (+8 more)

### Community 18 - "schema_specialists"
Cohesion: 0.12
Nodes (38): ocr_context_for_blob(), _build_retry_gemini_request_line(), combine_subagent_jsonl_sources(), CombinedSubagentOutputs, Path, Validate specialist results and join them into ordinary page records., write_combined_subagent_outputs(), _build_request_line() (+30 more)

### Community 19 - "validation/cli.py"
Cohesion: 0.09
Nodes (29): Random, build_validation_datapoints(), choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type() (+21 more)

### Community 20 - "access.py"
Cohesion: 0.16
Nodes (18): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+10 more)

### Community 21 - "submit_inputs.py"
Cohesion: 0.14
Nodes (25): _allowed_extensions(), _apply_fp_mode_to_blobs(), _apply_fp_mode_to_pdf_paths(), _apply_image_name_restriction(), _apply_year_filter_to_blobs(), _assert_gcs_input_source(), _configured_year_filter_tokens(), _dedupe_blob_image_names() (+17 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.16
Nodes (29): _anthropic_metadata(), _request_key_and_metadata(), _chosen_token_logprobs(), _collect_leaf_value_spans(), _collect_logprobs_by_pointer(), confidence_from_avg_logprobs(), _escape_pointer_segment(), extract_field_confidence_by_pointer() (+21 more)

### Community 23 - "config/__init__.py"
Cohesion: 0.11
Nodes (38): Counter, collect_outputs(), collect_valid_outputs_from_jsonl_sources(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines() (+30 more)

### Community 24 - "ValidatorApp"
Cohesion: 0.14
Nodes (8): Entry, build_image_index(), display_image_name(), load_dataset(), main(), Path, SamplingMode, ValidatorApp

### Community 25 - "submit_requests.py"
Cohesion: 0.25
Nodes (15): _anthropic_custom_id_for_key(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_anthropic_manifest_lines(), _build_request_config(), _guess_mime_type(), _inline_json_refs() (+7 more)

### Community 26 - "routing.py"
Cohesion: 0.07
Nodes (59): _build_thresholds(), _control_sample_sha256(), decide_candidate_route(), _decide_candidate_route_with_thresholds(), DeterministicRoutingDecision, _force_control_sample(), _frontpage_rule_ids(), _is_frontpage_model() (+51 more)

### Community 27 - "batch/service.py"
Cohesion: 0.10
Nodes (20): BatchChunkPlan, BatchCollectOutputsRequest, BatchSubmitPlan, BatchSubmitRequest, BatchSubmitService, Namespace, _filter_blobs_missing_from_dataset(), Namespace (+12 more)

### Community 28 - "ocr.py"
Cohesion: 0.13
Nodes (20): Protocol, _break_name(), _configured_backend(), detect_configured_ocr(), detect_configured_ocr_batch(), detect_ocr(), extract_google_vision_lines(), GoogleVisionOcrBackend (+12 more)

### Community 29 - "model_client.py"
Cohesion: 0.17
Nodes (18): _build_provider_client(), _extract_anthropic_response_text(), _extract_openai_response_text(), _import_anthropic_async_client(), _import_openai_async_client(), LocalGenerationResult, LocalModelClient, _pick_value() (+10 more)

### Community 30 - "resolve_batch_run_readiness"
Cohesion: 0.23
Nodes (13): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks(), list_batch_chunks_with_state(), Reduce per-chunk live states into a single job-level status. Returns…, Return the app-facing batch state, including output-file readiness. Some Gemini… (+5 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - "BrowserValidationSession"
Cohesion: 0.23
Nodes (5): BrowserValidationSession, Any, Server-side validation state for the browser validator., _score_for_label(), _stringify_value()

### Community 33 - "ocr_context.py"
Cohesion: 0.25
Nodes (15): _cache_key(), CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), _PendingOcr, prepare_ocr_metadata_for_blob(), prepare_ocr_metadata_for_blobs() (+7 more)

### Community 34 - "test_ocr.py"
Cohesion: 0.18
Nodes (15): OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., test_verification_refuses_changed_ocr_text_with_same_image_and_line_count(), FakeBucket, FakeOcrBackend, _png_bytes(), _symbol(), test_batch_ocr_preparation_creates_generation_bound_reusable_sidecar() (+7 more)

### Community 35 - "submit.py"
Cohesion: 0.15
Nodes (35): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), _batch_state_and_success(), _blob_names_sha256(), _build_chunk_entry(), _build_rerun_entries(), _chunk_label(), _chunk_requests_file_name() (+27 more)

### Community 36 - "PatientJournals research pipeline"
Cohesion: 0.11
Nodes (18): 10. Operational settings, 1. Input selection and image preparation, 2. Positional OCR, 3. First-pass extraction batch, 4. Schema-specialist decomposition, 5. Retrieval and deterministic sweep, 6. Exact evidence bindings for final validation, 7. Candidate-aware final verifier (+10 more)

### Community 37 - "PageModelValidation"
Cohesion: 0.22
Nodes (11): PageModelValidation, model_validator, ValidationIssuePatch, test_apply_validation_patches_is_rfc6901_aware(), test_apply_validation_patches_rejects_negative_array_indices(), test_field_correction_metadata_distinguishes_accepted_from_complete_dataset(), test_field_correction_metadata_records_original_and_applied_values(), test_publishable_dataset_uses_existing_model_to_rows_path() (+3 more)

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

### Community 44 - "app/schemas.py"
Cohesion: 0.17
Nodes (17): apply_schema_fields(), _canonical_json(), dataset_schema_field_paths(), _definitions(), _field_type(), flatten_schema_fields(), _locate_property(), _object_node() (+9 more)

### Community 46 - "test_schema_management.py"
Cohesion: 0.11
Nodes (17): fixture, Application services and desktop UI for PatientJournals., model_from_json_schema(), Any, Build a Pydantic model for an immutable app-managed JSON Schema version. The…, test_absolute_local_image_hint_does_not_become_a_cloud_object(), test_dashboard_completeness_reports_leafs_not_parent_objects(), test_dashboard_infers_legacy_schema_and_includes_fully_missing_leafs() (+9 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 48 - "config/schemas.py"
Cohesion: 0.20
Nodes (17): Address, Age, Bottom, Diagnoses, HospitalStay, list_output_schemas(), PageLine, Patient (+9 more)

### Community 58 - "datasets.py"
Cohesion: 0.15
Nodes (29): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+21 more)

### Community 59 - "patientjournals/tasks.py"
Cohesion: 0.25
Nodes (22): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+14 more)

### Community 60 - "submit_model_validation"
Cohesion: 0.08
Nodes (40): _get_anthropic_client(), _anthropic_request(), _anthropic_thinking_config(), _assert_dynamic_schema_supported(), _candidate_schema_identity(), _canonical_json_sha256(), _chunk_file_name(), _copy_or_download_candidate_artifact() (+32 more)

### Community 61 - "jobs.py"
Cohesion: 0.08
Nodes (68): _api_recovery_error_rows(), _api_recovery_error_summary(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows() (+60 more)

### Community 62 - "_list_input_blobs"
Cohesion: 0.40
Nodes (6): _list_input_blobs(), FakeBlob, FakeBucket, test_list_input_blobs_raises_when_restriction_matches_nothing(), test_list_input_blobs_scopes_to_restricted_image_names(), test_list_input_blobs_skips_duplicate_image_names_with_audit()

### Community 63 - ".name"
Cohesion: 0.16
Nodes (33): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), Count the deduplicated cloud population without signing preview URLs., _list_page_keys(), _normalize_job_state() (+25 more)

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

### Community 65 - "test_batch_verify.py"
Cohesion: 0.06
Nodes (44): candidate_sha256(), PageCandidateRecord, PageCandidateWriter, Any, BaseModel, field_validator, Path, Canonical unflattened extraction candidates for second-pass validation. (+36 more)

### Community 67 - "Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage., Source Nodes

### Community 68 - "generate.py"
Cohesion: 0.33
Nodes (9): generate_data(), process_file(), ProcessedFileResult, Any, BaseModel, is_fatal_api_error(), is_retryable_api_error(), BaseException (+1 more)

### Community 69 - "test_batch_retrieve_recovery.py"
Cohesion: 0.11
Nodes (13): FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency(), test_failed_page_retry_can_split_into_multiple_chunks() (+5 more)

### Community 70 - "test_data_inspection.py"
Cohesion: 0.14
Nodes (10): Local data inspection and health checks., FakeBlob, FakeBucket, png_bytes(), test_summarize_batch_data_can_skip_nested_files(), test_summarize_batch_data_counts_files_and_folders(), test_summarize_bucket_data_counts_prefix_blobs(), test_validate_batch_data_can_use_multiple_cores() (+2 more)

### Community 71 - "_prepare_validation_page"
Cohesion: 0.17
Nodes (13): _anthropic_signed_url_expiration(), timedelta, _load_bound_ocr_evidence(), _mime_type_for_name(), _prepare_validation_page(), Reload and validate the exact image/OCR evidence in an extraction manifest., verification_prompt_hash(), build_model_validation_prompt() (+5 more)

### Community 72 - "source_run_id_from_path"
Cohesion: 0.28
Nodes (9): _candidate_source_run_id(), _final_validation_policy_payload(), Bind all candidates to the supplied extraction run before submission., _scope_candidates(), _validate_candidate_source_run(), _write_final_validation_policy(), Return a portable run identifier from a local run-directory path., source_run_id_from_path() (+1 more)

### Community 75 - "preprocess.py"
Cohesion: 0.18
Nodes (16): Image, crop_margins(), enhance_contrast(), image_to_bytes(), load_image(), prepare_page(), PreparedPage, preprocess_image() (+8 more)

### Community 76 - "workflows.py"
Cohesion: 0.12
Nodes (24): JobRegistry, list_app_registry_jobs(), RegisteredJob, start_command(), app_settings_path(), JobSummary, Path, _coerce_settings() (+16 more)

### Community 77 - "output_handler.py"
Cohesion: 0.40
Nodes (13): FieldConfidenceByPointer, FrontPage, identity_columns(), _build_confidence_tree(), data_to_rows(), default_rows(), _escape_pointer_segment(), _has_field_confidence() (+5 more)

### Community 78 - "prepare_ocr.py"
Cohesion: 0.24
Nodes (11): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+3 more)

### Community 80 - "CloudBlobIdentity"
Cohesion: 0.10
Nodes (32): CloudBlobIdentity, _FirstPassRecoveryEvidence, Any, Exact image and optional OCR evidence used by one first-pass request., _verify_first_pass_image_bindings(), _binding_for_record(), ExtractionImageBinding, _identity_matches() (+24 more)

### Community 81 - "_ImmutableArtifactBlob"
Cohesion: 0.11
Nodes (10): _FakePreconditionFailure, _ImmutableArtifactBlob, _ImmutableArtifactBucket, _MissingPolicyBlob, _MissingPolicyBucket, Exception, test_final_validation_policy_is_create_only_and_anchors_prefixes(), test_immutable_validation_artifact_rejects_mismatched_existing_bytes() (+2 more)

### Community 82 - "verify.py"
Cohesion: 0.08
Nodes (66): _await_completion(), _batch_job_state(), _batch_job_successful(), _download_from_mldev_output(), _gemini_output_reference(), _get_batch_job(), Client, _apply_one_patch() (+58 more)

### Community 86 - "OcrDocument"
Cohesion: 0.14
Nodes (10): ocr_context_header(), _prompt(), Single source of truth for non-schema model prompt text. Edit page, sub-agent,…, Normalize source indentation without altering intentional line breaks., OcrDocument, OCR derived from, and cryptographically bound to, one image payload., Render every line with minimal syntax and no repeated field names., render_ocr_context() (+2 more)

### Community 87 - "_build_anthropic_batch_requests_for_retry"
Cohesion: 0.50
Nodes (5): _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests_for_retry(), Any, timedelta

### Community 89 - "browser.py"
Cohesion: 0.24
Nodes (10): ImageSource, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Path, SamplingMode, _safe_user() (+2 more)

## Knowledge Gaps
- **63 isolated node(s):** `patientjournals`, `UploadProfile`, `Batch-first architecture`, `Pipeline documentation`, `Prompt ownership` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `CloudBlobIdentity` (2× useful, score=1.999765272)
- `LocalModelClient` (2× useful, score=1.997808494) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `WorkflowService`, `workflows.py`, `AppSettings`, `datasets.py`, `resolve_batch_run_readiness`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `JobStore` connect `JobStore` to `WorkflowService`, `test_app_architecture.py`, `app/schemas.py`, `workflows.py`, `publication.py`, `AppSettings`, `test_schema_management.py`, `datasets.py`, `jobs.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `WorkflowService` connect `WorkflowService` to `JobStore`, `PatientJournalsApp`, `ImageAccessService`, `status.py`, `test_app_architecture.py`, `workflows.py`, `app/schemas.py`, `AppSettings`, `verify.py`, `BrowserValidationManager`, `submit_model_validation`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `WorkflowService` (e.g. with `PatientJournalsApp` and `AppHandler`) actually correct?**
  _`WorkflowService` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `retrieve_batch()` (e.g. with `.recover_missing_with_api()` and `output_schema_name()`) actually correct?**
  _`retrieve_batch()` has 3 INFERRED edges - model-reasoned connections that need verification._