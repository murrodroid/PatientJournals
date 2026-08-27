## Batch-first architecture

Cloud batch jobs are the primary production path for this project. Design and implement features against the batch workflow first:

1. `uv run invoke batch.upload`
2. `uv run invoke batch.ocr`
3. `uv run invoke batch.submit`
4. `uv run invoke batch.status --watch`
5. `uv run invoke batch.retrieve --wait`

Rules:
- A feature is not complete if it only works through `patientjournals.local`. Local generation is a secondary development and recovery path.
- Durable preprocessing output needed by a batch request must live in cloud storage. Batch request construction should retrieve prepared artifacts; it must not perform expensive preprocessing, OCR, or full image downloads.
- OCR must be prepared before submission with `batch.ocr`. Sidecars must be bound to the exact GCS object generation, and missing or stale metadata must block batch submission by default.
- Cloud preprocessing should use bounded provider-native batches and parallelism rather than one remote call per input. Keep batch sizes and concurrency configurable so production runs can stay within provider quotas.
- New operational commands and tests should preserve the sequence upload -> cloud preprocessing -> submit -> retrieve.
- Prefer provider-independent request preparation before the provider-specific Gemini or Anthropic batch adapters.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
