---
type: "query"
date: "2026-08-27T09:25:02.899264+00:00"
question: "Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/"
contributor: "graphify"
outcome: "useful"
source_nodes: ["preprocess_image_with_metadata", "image_to_bytes", "generate_data", "LocalModelClient", "build_live_request_contents", "run_local_job", "WorkflowService"]
---

# Q: Provide a clear path for the implementation of the 2 following changes. Explain the exact implementation method. Work through them one at a time. 1. Create a preprocessing step before sending through model, which OCR scans the entire page and provides all detected text with image positional arguments. These positions need to be correct for the actual bytes which are given to the model at the final stage, meaning the processed image. 2. Create a method for which we can implement agents to subdivide the transcription task into subproblems that are solved individually. Consider deeply different implementations of doing such and how we could do this best. https://pydantic.dev/docs/ai/core-concepts/agent/

## Answer

Expanded from original query via graph vocabulary: image, preprocessing, processed, bytes, model, prompt, text, location, bounds, resolution, resize, crop, pydantic, task, split, parallel, workflow, orchestrator, worker, schema, validation, generation. Recommended design: serialize the processed page once, OCR those exact immutable bytes, record pixel and normalized polygons plus a SHA-256 identity, and pass the same bytes and OCR context through unified provider prompt builders. For agentic transcription, add a TranscriptionEngine abstraction and implement a deterministic programmatic fan-out and fan-in pipeline using reusable Pydantic AI workers with structured outputs, a deterministic merge, and a verifier. Keep dynamic manager delegation and Pydantic Graph as later extensions after the fixed hybrid decomposition is validated.

## Outcome

- Signal: useful

## Source Nodes

- preprocess_image_with_metadata
- image_to_bytes
- generate_data
- LocalModelClient
- build_live_request_contents
- run_local_job
- WorkflowService