---
type: "query"
date: "2026-08-27T10:41:02.149338+00:00"
question: "Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind."
contributor: "graphify"
outcome: "useful"
source_nodes: ["PreparedPage", "OcrDocument", "LocalModelClient"]
---

# Q: Implement OCR, with the goal of providing additional context for the model. Build while keeping minimization of input token usage in mind.

## Answer

Expanded from original query via graph vocab: [image, preprocess, bytes, model, context, token, text, metadata, page]. Implemented Google Vision document OCR after final image serialization. PreparedPage binds the exact model bytes, dimensions, SHA-256 digest, preprocessing metadata, and OCR lines. OCR is rendered as compact normalized line boxes using x1,y1,x2,y2|text, wired through Gemini, OpenAI, Anthropic, batch submission, retry, and recovery. Batch sidecars are reused only on digest match and page uploads are create-only. Added configuration, dependency, and six tests; full suite passes 134 tests.

## Outcome

- Signal: useful

## Source Nodes

- PreparedPage
- OcrDocument
- LocalModelClient