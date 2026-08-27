---
type: "query"
date: "2026-08-27T11:09:37.216487+00:00"
question: "Is OCR done linearly, and why not parallelize all images or call OCR in batches?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["GoogleVisionOcrBackend", "detect_configured_ocr", "prepare_cloud_ocr_metadata", "OcrMetadataPreparation", "CloudBlobIdentity"]
---

# Q: Is OCR done linearly, and why not parallelize all images or call OCR in batches?

## Answer

Expanded via graph vocabulary: [OCR, GoogleVisionOcrBackend, detect_configured_ocr, prepare_cloud_ocr_metadata, ThreadPoolExecutor, sidecar, batch]. The original implementation used 8 concurrent workers but each worker made one document_text_detection RPC per image, so it was parallel but RPC-inefficient. It now groups up to 16 exact image payloads in each batch_annotate_images RPC, splits earlier at the configured byte cap, and runs multiple RPC batches concurrently. A single request for all images is invalid: synchronous Vision is limited to 16; asynchronous Vision supports 2000 but GCS ImageSource does not support object versioning, which conflicts with the exact-generation OCR contract.

## Outcome

- Signal: useful

## Source Nodes

- GoogleVisionOcrBackend
- detect_configured_ocr
- prepare_cloud_ocr_metadata
- OcrMetadataPreparation
- CloudBlobIdentity