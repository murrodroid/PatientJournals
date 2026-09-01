# Graph Report - PatientJournals  (2026-08-27)

## Corpus Check
- 107 files · ~137,834 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1928 nodes · 5775 edges · 79 communities (69 shown, 10 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 331 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `342d9791`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app/schemas.py
- tools.py
- WorkflowService
- retry.py
- config/schemas.py
- PatientJournalsApp
- dashboard.py
- upload.py
- resolve_batch_run_readiness
- status.py
- test_app_architecture.py
- PatientJournals Conda Environment
- inspection.py
- publication.py
- retrieve.py
- workflows.py
- validation/cli.py
- ValidatorApp
- schema_specialists
- config/__init__.py
- access.py
- collect_outputs.py
- response_parsing.py
- ImageAccessService
- model_client.py
- submit_requests.py
- JobStore
- PageCandidateWriter
- ocr.py
- BrowserValidationSession
- submit_model_validation
- Journal
- .name
- ocr_context.py
- test_ocr.py
- submit.py
- preprocess.py
- verify.py
- build_validation_datapoints
- Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?
- _recover_missing_pages_via_api_key
- Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/
- AGENTS.md
- analysis.py
- test_batch_retrieve_recovery.py
- utc_now_iso
- Path
- Journal
- PageModelValidation
- prepare_ocr.py
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
- image_name_from_reference
- jobs.py
- input_manifest.py
- get_batch_client
- Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.
- test_batch_verify.py
- _registered_store
- Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage.
- _prepare_validation_page
- FakeBlob
- OcrDocument
- .__init__
- test_schema_management.py
- _resolve_extraction_schema
- TaskRunner
- _build_anthropic_batch_requests_for_retry
- JobRegistry
- subagent_outputs.py
- _api_key_recovery_failure_reason

## God Nodes (most connected - your core abstractions)
1. `JobStore` - 75 edges
2. `AppSettings` - 65 edges
3. `WorkflowService` - 64 edges
4. `retrieve_batch()` - 52 edges
5. `submit_batch()` - 48 edges
6. `PatientJournalsApp` - 45 edges
7. `retrieve_model_validation()` - 44 edges
8. `submit_model_validation()` - 37 edges
9. `serializable()` - 35 edges
10. `_Bucket` - 35 edges

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

## Communities (79 total, 10 thin omitted)

### Community 0 - "app/schemas.py"
Cohesion: 0.17
Nodes (17): apply_schema_fields(), _canonical_json(), dataset_schema_field_paths(), _definitions(), _field_type(), flatten_schema_fields(), _locate_property(), _object_node() (+9 more)

### Community 1 - "tools.py"
Cohesion: 0.11
Nodes (41): ProgressCallback, main(), parse_args(), _progress_printer(), Namespace, create_local_model_client(), _emit(), _input_without_existing() (+33 more)

### Community 2 - "WorkflowService"
Cohesion: 0.13
Nodes (12): _apply_runtime_overrides(), cancel_batch_run(), Cancel every non-terminal batch job belonging to a submit run. Returns the…, _restore_runtime_overrides(), run_local_draft_direct(), command_override_payload(), Any, Submit the candidate-aware verifier batch for a retrieved extraction. (+4 more)

### Community 3 - "retry.py"
Cohesion: 0.16
Nodes (31): _anthropic_custom_id_for_key(), _append_retry_to_source_metadata(), _build_retry_anthropic_manifest_line(), _build_retry_batch_generation_config(), _build_retry_gemini_request_line(), _chunk_label(), _count_requests_file(), _extract_location_from_batch_name() (+23 more)

### Community 4 - "config/schemas.py"
Cohesion: 0.07
Nodes (52): FieldConfidenceByPointer, fixture, resolve_schema_class(), Address, Age, Bottom, Diagnoses, FrontPage (+44 more)

### Community 5 - "PatientJournalsApp"
Cohesion: 0.10
Nodes (15): BooleanVar, Canvas, Frame, Label, LabelFrame, Misc, main(), _open_in_file_browser() (+7 more)

### Community 6 - "dashboard.py"
Cohesion: 0.09
Nodes (49): analyze_dataset_file(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _counter(), _counter_key(), dashboard_summary_json(), DashboardSummary (+41 more)

### Community 7 - "upload.py"
Cohesion: 0.13
Nodes (37): _allowed_page_extensions(), _apply_fp_mode_filter(), _apply_image_settings(), _build_bucket(), _effective_batch_limit(), _effective_workers(), _ensure_unique_pdf_names(), _extension_for_format() (+29 more)

### Community 8 - "resolve_batch_run_readiness"
Cohesion: 0.16
Nodes (17): aggregate_batch_state(), BatchRunReadiness, _is_failure_state(), _is_success_state(), list_batch_chunks(), list_batch_chunks_with_state(), poll_local_batch_states(), Reduce per-chunk live states into a single job-level status. Returns… (+9 more)

### Community 9 - "status.py"
Cohesion: 0.06
Nodes (61): _batch_model_progress(), _aggregate_state_lines(), _anthropic_model_progress(), _batch_state(), _batch_summary(), _cancel_batch_job(), _count_gemini_prediction_rows(), _count_jsonl_blob_lines() (+53 more)

### Community 10 - "test_app_architecture.py"
Cohesion: 0.06
Nodes (44): Application services and desktop UI for PatientJournals., batch_run_provider(), find_dataset_near(), list_submit_jobs(), Return the text of any locally written error file for a run, if present., Return saved results when they satisfy the requested retrieval options. This is…, Locate a dataset file at ``reference`` or, failing that, in its directory.…, Read up to ``limit`` rows from a dataset for a quick on-screen preview. Returns… (+36 more)

### Community 11 - "PatientJournals Conda Environment"
Cohesion: 0.06
Nodes (38): Document and Spreadsheet I/O Dependencies, Google AI and Cloud Dependency Stack, Image and Data Processing Dependency Stack, PatientJournals Conda Environment, Python 3.11, Anthropic, Anthropic Message Batches, Balanced UCB Validation Sampling (+30 more)

### Community 12 - "inspection.py"
Cohesion: 0.10
Nodes (38): main(), _nonnegative_int(), _parse_args(), _print_summary(), _print_validation(), Namespace, Local data inspection and health checks., collect_files() (+30 more)

### Community 13 - "publication.py"
Cohesion: 0.11
Nodes (26): Exception, _atomic_write_json(), DatasetVersionPublication, _empty_ledger(), file_sha256(), _normalize_prefix(), _not_found(), _parse_ledger() (+18 more)

### Community 14 - "retrieve.py"
Cohesion: 0.10
Nodes (49): add_response_metadata_columns(), RetrieveBatchResult, _anthropic_custom_id_for_key(), _anthropic_stop_reason(), _arg_batch_names(), _download_from_anthropic_output(), _download_from_mldev_output(), _effective_duplicate_strategy() (+41 more)

### Community 15 - "workflows.py"
Cohesion: 0.09
Nodes (39): DuplicateStrategy, build_retrieve_command(), build_submit_command(), build_validation_command(), Resubmit the requests that did not succeed as a fresh batch. Clears the…, resubmit_failed_requests(), app_settings_path(), AppSettings (+31 more)

### Community 16 - "validation/cli.py"
Cohesion: 0.19
Nodes (19): Random, choose_balanced_ucb_datapoint(), choose_random_datapoint(), _count_for_sampling_group(), eligible_flat_fields(), flatten_row(), _get_field_type(), _is_metadata_field() (+11 more)

### Community 17 - "ValidatorApp"
Cohesion: 0.15
Nodes (7): Button, Entry, display_image_name(), main(), Path, SamplingMode, ValidatorApp

### Community 18 - "schema_specialists"
Cohesion: 0.19
Nodes (19): generate_data(), Any, BaseModel, merge_specialist_metadata(), merge_specialist_payloads(), Any, BaseModel, Build a compact search brief; the response schema carries field detail. (+11 more)

### Community 19 - "config/__init__.py"
Cohesion: 0.15
Nodes (24): _generate_recovery_response(), _guess_blob_mime_type(), _recover_one_missing_page_via_api_key(), Configuration, schema, and model registry., process_file(), ProcessedFileResult, is_fatal_api_error(), is_retryable_api_error() (+16 more)

### Community 20 - "access.py"
Cohesion: 0.16
Nodes (18): CommandRunner, CompletedProcess, AccessCheckReport, AccessCheckResult, active_gcloud_account(), _bucket_fix(), _configured_prefixes(), _default_runner() (+10 more)

### Community 21 - "collect_outputs.py"
Cohesion: 0.12
Nodes (35): Counter, collect_outputs(), collect_valid_outputs_from_jsonl_sources(), CollectedGeminiOutputs, _counter_to_dict(), _expand_local_output_paths(), _flush_collected_rows(), _iter_blob_lines() (+27 more)

### Community 22 - "response_parsing.py"
Cohesion: 0.07
Nodes (46): BaseHTTPRequestHandler, list_batch_model_options(), list_google_model_options(), list_live_google_model_options(), list_schema_options(), _model_option_from_name(), Models suitable for provider batch jobs, including model validation., ModelOption (+38 more)

### Community 23 - "ImageAccessService"
Cohesion: 0.29
Nodes (4): ImageAccessService, Any, Path, Short-lived image links for dataset inspection and submission previews.

### Community 24 - "model_client.py"
Cohesion: 0.17
Nodes (18): _build_provider_client(), _extract_anthropic_response_text(), _extract_openai_response_text(), _import_anthropic_async_client(), _import_openai_async_client(), LocalGenerationResult, LocalModelClient, _pick_value() (+10 more)

### Community 25 - "submit_requests.py"
Cohesion: 0.20
Nodes (20): ocr_context_for_blob(), _anthropic_custom_id_for_key(), _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests(), _build_anthropic_manifest_line(), _build_anthropic_manifest_lines(), _build_request_config() (+12 more)

### Community 26 - "JobStore"
Cohesion: 0.15
Nodes (7): Connection, Row, JobStore, _json_dumps(), _json_loads(), SQLite-backed app state for jobs. Operational run folders are artifacts only.…, test_job_store_persists_background_tasks()

### Community 27 - "PageCandidateWriter"
Cohesion: 0.15
Nodes (14): candidate_sha256(), PageCandidateWriter, Any, Path, Canonical unflattened extraction candidates for second-pass validation., Keep reproducibility/provenance only; never carry model thoughts forward., Buffered JSONL writer that enforces one canonical record per page key., read_page_candidates() (+6 more)

### Community 28 - "ocr.py"
Cohesion: 0.18
Nodes (11): _break_name(), _configured_backend(), detect_configured_ocr_batch(), extract_google_vision_lines(), GoogleVisionOcrBackend, OcrImageInput, Collapse Vision's symbol hierarchy into token-efficient visual lines., Send up to 16 images through one Vision images:annotate RPC. (+3 more)

### Community 29 - "BrowserValidationSession"
Cohesion: 0.24
Nodes (4): BrowserValidationSession, Server-side validation state for the browser validator., _score_for_label(), _stringify_value()

### Community 30 - "submit_model_validation"
Cohesion: 0.13
Nodes (20): _anthropic_request(), _anthropic_thinking_config(), _chunk_file_name(), _gemini_generation_config(), _gemini_request_line(), ModelValidationSubmitResult, _PreparedValidationPage, Keep each durable/submitted Message Batch safely below 256 MB. (+12 more)

### Community 31 - "Journal"
Cohesion: 0.11
Nodes (19): Accuracy, Address, Age, Bottom, Dataset, Diagnoses, Front Page Images, Hospital Stay (+11 more)

### Community 32 - ".name"
Cohesion: 0.18
Nodes (33): _iter_cloud_validation_rows(), cloud_object_by_image_name(), list_cloud_dataset_choices(), list_cloud_dataset_library(), list_cloud_dataset_prefixes(), resolve_local_images_on_cloud(), _list_page_keys(), _blob_extension() (+25 more)

### Community 33 - "ocr_context.py"
Cohesion: 0.22
Nodes (17): _cache_key(), CloudOcrMetadata, _download(), load_ocr_metadata_for_blob(), ocr_document_for_blob(), _PendingOcr, prepare_ocr_metadata_for_blob(), prepare_ocr_metadata_for_blobs() (+9 more)

### Community 34 - "test_ocr.py"
Cohesion: 0.18
Nodes (15): OcrAttempt, OcrLine, One OCR line with a compact, normalized axis-aligned bounding box., FakeBucket, FakeOcrBackend, _png_bytes(), _symbol(), test_batch_ocr_preparation_creates_generation_bound_reusable_sidecar() (+7 more)

### Community 35 - "submit.py"
Cohesion: 0.05
Nodes (80): Fail before request generation when required cloud sidecars are unavailable., validate_ocr_metadata_for_blobs(), BatchChunkPlan, BatchCollectOutputsRequest, BatchSubmitPlan, BatchSubmitRequest, BatchSubmitService, Namespace (+72 more)

### Community 36 - "preprocess.py"
Cohesion: 0.14
Nodes (22): Image, Protocol, detect_configured_ocr(), detect_ocr(), image_identity(), OcrBackend, Read canonical dimensions and digest from the exact serialized bytes., Run configured OCR, failing open unless ``ocr_required`` is set. (+14 more)

### Community 37 - "verify.py"
Cohesion: 0.10
Nodes (47): _await_completion(), _batch_job_state(), _batch_job_successful(), _gemini_output_reference(), _get_batch_job(), _get_anthropic_client(), _apply_one_patch(), apply_validation_patches() (+39 more)

### Community 38 - "build_validation_datapoints"
Cohesion: 0.12
Nodes (10): build_validation_datapoints(), Any, resolve_image_path(), validation_sampling_group_key(), Validation UI and reporting commands., test_versioned_validation_uses_each_rows_schema_and_model(), test_balanced_ucb_prioritizes_under_sampled_schema_field(), test_balanced_ucb_separates_missing_and_present_values() (+2 more)

### Community 39 - "Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?, Source Nodes

### Community 40 - "_recover_missing_pages_via_api_key"
Cohesion: 0.16
Nodes (18): add_reproducibility_columns(), _build_api_key_generation_config(), _dataset_content_type(), _download_from_vertex_gcs_output(), _normalize_prefix(), _parse_gcs_uri(), Any, Upload a pre-v1 candidate outside the canonical datasets namespace. (+10 more)

### Community 41 - "Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/, Source Nodes

### Community 42 - "AGENTS.md"
Cohesion: 0.50
Nodes (3): Batch-first architecture, graphify, Prompt ownership

### Community 43 - "analysis.py"
Cohesion: 0.60
Nodes (10): _add_accuracy_scores(), load_validations(), main(), plot_label_distribution(), plot_nested_accuracy(), plot_overall_accuracy(), plot_top_level_accuracy(), DataFrame (+2 more)

### Community 44 - "test_batch_retrieve_recovery.py"
Cohesion: 0.10
Nodes (10): Batch upload, submission, status, and retrieval commands., FakeBlob, FakeBucket, BaseModel, SimpleOutput, test_api_key_recovery_failure_reason_includes_exception_detail(), test_api_key_recovery_retries_transient_errors(), test_api_key_recovery_uses_configured_concurrency() (+2 more)

### Community 45 - "utc_now_iso"
Cohesion: 0.33
Nodes (5): Record extraction retrieval without publishing a dataset version. Model-…, Link a verifier batch to its extraction job without versioning data., Record verifier results and optionally publish a dataset version once., _safe_job_id(), utc_now_iso()

### Community 46 - "Path"
Cohesion: 0.24
Nodes (9): _copy_dataset_into_job(), _dataset_files(), _model_validation_idempotency_key(), _path_identity(), Path, Return a stable identity for a run directory without requiring it to exist., read_json_file(), _submit_root_for_run_dir() (+1 more)

### Community 47 - "Journal"
Cohesion: 0.22
Nodes (9): Address, Age, Bottom, Diagnoses, Hospital Stay, Journal, Patient, Section (+1 more)

### Community 48 - "PageModelValidation"
Cohesion: 0.27
Nodes (8): model_validator, PageModelValidation, ValidationIssuePatch, test_apply_validation_patches_is_rfc6901_aware(), test_apply_validation_patches_rejects_negative_array_indices(), test_publishable_dataset_uses_existing_model_to_rows_path(), test_sparse_validation_schema_enforces_status_and_patch_contract(), test_sparse_validation_schema_rejects_overlapping_patch_paths()

### Community 49 - "prepare_ocr.py"
Cohesion: 0.24
Nodes (11): OcrMetadataPreparation, CloudOcrPreparationSummary, main(), _manifest_object_name(), _parse_args(), prepare_cloud_ocr_metadata(), Namespace, Populate GCS OCR sidecars for the configured batch input selection. (+3 more)

### Community 58 - "datasets.py"
Cohesion: 0.15
Nodes (29): combine_dataset_files(), _count_csv_rows(), count_dataset_rows(), _count_jsonl_rows(), _dataset_content_type(), download_cloud_dataset(), _flatten_dataset_row(), _format_blob_updated() (+21 more)

### Community 59 - "patientjournals/tasks.py"
Cohesion: 0.25
Nodes (22): _add_flag(), _add_option(), app_run(), check_models(), collect_outputs(), config_path(), config_show(), data_batch() (+14 more)

### Community 60 - "image_name_from_reference"
Cohesion: 0.21
Nodes (16): copy_dataset_rows_for_image_names(), copy_dataset_rows_for_keys(), load_dataset_image_coverage(), load_dataset_key_coverage(), normalize_dataset_image_name(), _normalize_output_format(), Path, ensure_row_image_name() (+8 more)

### Community 61 - "jobs.py"
Cohesion: 0.09
Nodes (64): _api_recovery_error_rows(), _api_recovery_error_summary(), _append_retry_child_to_source_metadata(), _batch_chunk_summaries_from_payload(), _batch_submit_namespace(), BatchSubmitOutcome, command_overrides_for_run(), _count_output_rows() (+56 more)

### Community 62 - "input_manifest.py"
Cohesion: 0.12
Nodes (26): CloudBlobIdentity, _binding_for_record(), ExtractionImageBinding, _identity_matches(), input_manifest_record_for_blob(), InputImageManifestRecord, _mime_type(), _normalized_prefix() (+18 more)

### Community 63 - "get_batch_client"
Cohesion: 0.24
Nodes (15): _candidate_model_ids(), _check_model_ids(), _ConfigSnapshot, _iter_models(), main(), _norm(), _parse_args(), _print_models() (+7 more)

### Community 64 - "Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind., Source Nodes

### Community 65 - "test_batch_verify.py"
Cohesion: 0.13
Nodes (19): PageCandidateRecord, BaseModel, field_validator, One page-level candidate before one-to-many dataset flattening., _identity(), _input_record(), _ocr_metadata(), test_all_scope_requires_candidates_equal_full_input_manifest() (+11 more)

### Community 66 - "_registered_store"
Cohesion: 0.56
Nodes (8): Path, _registered_store(), test_candidate_retrieval_does_not_publish_v1(), test_completed_model_validation_publishes_immutable_v1(), test_job_store_defensively_rejects_unsafe_validation_publication(), test_repeated_validation_runs_create_traceable_v1_v2(), test_report_only_validation_never_publishes_dataset(), test_retrieving_same_published_verifier_run_is_idempotent()

### Community 67 - "Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage., Source Nodes

### Community 68 - "_prepare_validation_page"
Cohesion: 0.29
Nodes (8): _mime_type_for_name(), _prepare_validation_page(), verification_prompt_hash(), build_model_validation_prompt(), _compact_json(), Any, Build the compact, provider-independent verifier request text., test_validation_prompt_is_evidence_first_and_candidate_last()

### Community 70 - "OcrDocument"
Cohesion: 0.11
Nodes (15): build_subagent_prompt(), ocr_context_header(), _prompt(), Single source of truth for non-schema model prompt text. Edit page, sub-agent,…, Render the compact role brief shared by batch, retry, and local paths., Normalize source indentation without altering intentional line breaks., OcrDocument, OCR derived from, and cryptographically bound to, one image payload. (+7 more)

### Community 71 - ".__init__"
Cohesion: 0.15
Nodes (13): ImageSource, BrowserValidationManager, _create_validation_run_dir(), _local_image_index(), _ordered_dataset_image_names(), _placeholder_cloud_image_index(), Any, Path (+5 more)

### Community 72 - "test_schema_management.py"
Cohesion: 0.25
Nodes (6): test_absolute_local_image_hint_does_not_become_a_cloud_object(), test_dashboard_completeness_reports_leafs_not_parent_objects(), test_dashboard_infers_legacy_schema_and_includes_fully_missing_leafs(), test_dataset_inspection_handles_lists_and_prioritizes_provenance(), test_removing_all_nested_leafs_prunes_the_parent_object(), test_schema_versions_round_trip_through_cloud()

### Community 73 - "_resolve_extraction_schema"
Cohesion: 0.31
Nodes (10): _candidate_schema_identity(), _copy_or_download_candidate_artifact(), _download_gcs_file(), ModelValidationSubmitRequest, _parse_gcs_uri(), Client, _resolve_extraction_schema(), _resolve_input_manifest() (+2 more)

### Community 75 - "_build_anthropic_batch_requests_for_retry"
Cohesion: 0.50
Nodes (5): _anthropic_signed_url_expiration(), _anthropic_strict_json_schema(), _build_anthropic_batch_requests_for_retry(), Any, timedelta

### Community 76 - "JobRegistry"
Cohesion: 0.33
Nodes (6): JobRegistry, list_app_registry_jobs(), RegisteredJob, start_command(), JobSummary, test_job_registry_roundtrip()

### Community 78 - "subagent_outputs.py"
Cohesion: 0.15
Nodes (21): gemini_finish_reason(), Return the normalized first-candidate finish reason, if present., _anthropic_metadata(), combine_subagent_jsonl_sources(), CombinedSubagentOutputs, Path, Validate specialist results and join them into ordinary page records., _request_key_and_metadata() (+13 more)

### Community 81 - "_api_key_recovery_failure_reason"
Cohesion: 0.67
Nodes (4): _api_key_recovery_failure_reason(), _compact_exception_text(), BaseException, _redact_error_text()

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

- **Why does `JobStore` connect `JobStore` to `app/schemas.py`, `WorkflowService`, `_registered_store`, `test_schema_management.py`, `test_app_architecture.py`, `TaskRunner`, `utc_now_iso`, `Path`, `workflows.py`, `datasets.py`, `jobs.py`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `WorkflowService` connect `WorkflowService` to `app/schemas.py`, `PatientJournalsApp`, `verify.py`, `.__init__`, `_resolve_extraction_schema`, `test_app_architecture.py`, `workflows.py`, `response_parsing.py`, `ImageAccessService`, `JobStore`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `PatientJournalsApp` connect `PatientJournalsApp` to `datasets.py`, `WorkflowService`, `JobRegistry`, `workflows.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `JobStore` (e.g. with `finalize_dataset_with_failed_rows()` and `find_dataset_near()`) actually correct?**
  _`JobStore` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 52 inferred relationships involving `AppSettings` (e.g. with `_configured_prefixes()` and `resolve_validator_identity()`) actually correct?**
  _`AppSettings` has 52 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `WorkflowService` (e.g. with `PatientJournalsApp` and `AppHandler`) actually correct?**
  _`WorkflowService` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `retrieve_batch()` (e.g. with `.recover_missing_with_api()` and `output_schema_name()`) actually correct?**
  _`retrieve_batch()` has 3 INFERRED edges - model-reasoned connections that need verification._