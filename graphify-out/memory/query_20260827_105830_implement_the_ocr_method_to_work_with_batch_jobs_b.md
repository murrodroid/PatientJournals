---
type: "query"
date: "2026-08-27T10:58:30.938071+00:00"
question: "Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage."
contributor: "graphify"
outcome: "useful"
source_nodes: ["CloudOcrPreparationSummary", "prepare_cloud_ocr_metadata", "prepare_ocr_metadata_for_blob", "CloudOcrMetadata", "CloudBlobIdentity", "validate_ocr_metadata_for_blobs", "ocr_context_for_blob", "submit.py"]
---

# Q: Implement the OCR method to work with batch jobs by retrieving cloud images and creating durable OCR metadata for each; make batch jobs the main point of usage.

## Answer

Expanded from the original request via graph vocabulary: [ocr, batch, cloud, storage, metadata, blob, sidecar, request, image, upload, submit, retrieve]. prepare_cloud_ocr_metadata enumerates the same GCS selection as submit, and prepare_ocr_metadata_for_blob downloads an exact object generation, runs OCR once, then stores a compact versioned CloudOcrMetadata sidecar plus a cloud manifest. validate_ocr_metadata_for_blobs preflights generation-matched sidecars before request generation. ocr_context_for_blob is retrieval-only, so submit, retry, and recovery never invoke OCR during model request construction; missing or stale required metadata directs operators to batch.ocr.

## Outcome

- Signal: useful

## Source Nodes

- CloudOcrPreparationSummary
- prepare_cloud_ocr_metadata
- prepare_ocr_metadata_for_blob
- CloudOcrMetadata
- CloudBlobIdentity
- validate_ocr_metadata_for_blobs
- ocr_context_for_blob
- submit.py