# Agent log: review v5 vs v2 paper

- Timestamp: 2025-12-20T15:44:46-08:00
- Agent: Codex CLI
- Model: GPT-5.2

## Scope

- Read and compared:
  - `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v2_dec16.pdf`
  - `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v5.pdf`
- Method: extracted text with `pdftotext -layout` and diffed key sections (abstract, task definitions, shift motivation, limitations/future work, references).

## Key findings

- v5 is a clear improvement in framing and correctness (task definitions now match the implemented benchmark structure: log-coordinates + motifs; levels are explicitly a taxonomy label, not a strict complexity ordering).
- v5 is more practitioner-oriented (adds “When to care” and “What to do” boxes) and adds a minimal literature review with references.
- Remaining issues include a few math/implementation alignment details (coordinate definitions and Level 7 gating) and some presentation choices (internal “requested runs” list embedded in the paper).
