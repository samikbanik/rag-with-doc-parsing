# Enterprise Agentic RAG Chat — Consolidated Plan (post-review)

## Context

Goal: an enterprise chat where employees ask questions over parsed company documents/wikis and
trigger actions via an agentic tool-calling loop. Retrieval is RAG over a vector DB. OpenAI is the
only LLM/embedding provider. Deployment is out of scope; focus is parsing → indexing → retrieval →
agent → UI.

Decisions already made with the user:
- Python 3.12 (via `uv`) backend; React + TypeScript (Vite) frontend
- Qdrant (Docker) as vector DB; Postgres (Docker) for app state; OpenAI `text-embedding-3-large`
  truncated to **1024 dims**
- **No ACLs, no OCR** for now (schema keeps an unused `acl` field; image-only pages are logged and skipped)
- First ingestion flow = **local directory path**; then UI upload; **Notion** is the first real connector
- Action tools are **stubs** behind a confirmation step
- Ingestion is both scheduled and user-triggered, through **one job queue** (`procrastinate`)

Approving this plan means: create `PLAN.md` in the repo from this document and begin **Milestone 0 only**.

## Architecture

```
Sources ─► Connector ─► Parser ─► ParsedDocument IR ─► Chunker ─► Embedder ─► Qdrant (dense + sparse + payload)
(local dir │ upload │ Notion)          │ (data/parsed/*.json)                     ▲
                                       └─► Postgres: documents, chunks meta, jobs, conversations, traces
Chat UI ─► FastAPI (SSE) ─► Agent loop ─► tools: search_documents / get_document / stubbed actions
                                 └─► Retriever: query rewrite → hybrid (RRF) → [rerank] → context assembly → LLM
```

Two independently runnable halves (ingestion, agent) sharing Qdrant + Postgres schemas only.

## Stack

| Concern | Choice | Notes |
|---|---|---|
| Backend | Python 3.12, `uv`, `ruff`, `pytest`, `pydantic-settings` | pin 3.12 for torch/Docling compat |
| Parsing | Docling (PDF/DOCX/PPTX/XLSX, OCR disabled), `markdown-it-py` + `BeautifulSoup` (MD/HTML), `pymupdf4llm` behind a flag as fast PDF path | benchmark Docling on real docs in M1 |
| Token counting | `tiktoken` | chunk sizing + cost estimates |
| Vector DB | Qdrant via docker-compose; dense 1024-d cosine + sparse BM25 (`fastembed` `Qdrant/bm25`, server-side IDF modifier) | payload indexes on `doc_id`, `source`, `doc_type`, `last_modified` |
| App DB | Postgres via docker-compose; SQLAlchemy 2 + Alembic | documents, chunks, jobs, conversations, traces |
| Jobs + schedule | `procrastinate` | Postgres-native queue with retries and periodic tasks; replaces APScheduler |
| LLM | OpenAI Responses API via a thin `LLMClient` wrapper (structured outputs, tool calls, streaming) | model IDs in config only |
| Rerank | Pluggable: `none` (default) / cross-encoder / LLM listwise | enable only if eval shows gain |
| Eval | custom recall@k / MRR + `ragas` | golden set bootstrapped synthetically, then curated |
| Frontend | React 18 + TS + Vite, Tailwind, TanStack Query, `react-markdown`, `openapi-typescript` for API types, `fetch`+`ReadableStream` for SSE (POST) | |
| Tests | pytest, parser snapshot tests, `testcontainers` for Qdrant/Postgres, `vcrpy` cassettes for OpenAI | CI runs without spending tokens |

## Repo layout

```
doc-parser-with-rag/
├── PLAN.md  pyproject.toml  uv.lock  docker-compose.yml  .env.example  Makefile
├── config/settings.yaml                # models, dims, chunk sizes, collection name, pipeline_version
├── alembic/
├── src/ragchat/
│   ├── core/        settings.py  llm.py (OpenAI wrapper)  db.py  logging.py
│   ├── ingest/
│   │   ├── connectors/  base.py  local.py  upload.py  notion.py
│   │   ├── parsers/     base.py  router.py  docling_parser.py  markdown.py  html.py  pymupdf.py
│   │   ├── ir.py        # ParsedDocument, Block
│   │   ├── chunking.py  embedding.py  state.py  pipeline.py  jobs.py
│   ├── retrieval/   vectorstore.py  hybrid.py  rerank.py  query_rewrite.py  assemble.py
│   ├── agent/       tools.py  registry.py  loop.py  prompts.py  memory.py
│   ├── api/         app.py  chat.py  documents.py  jobs.py  auth.py (X-API-Key)
│   └── cli.py       # rag ingest | query | eval | worker | status | reindex
├── eval/            golden.jsonl  generate_golden.py  run_eval.py  baselines/
├── tests/           unit/  snapshots/  integration/  cassettes/
├── frontend/        # Vite app
└── data/{raw,parsed}/   # gitignored
```

## Key design rules (from review)

1. **IR is typed, not Markdown**: `Block(type, text, level, page, bbox, table_html)`; needed for citations.
2. **Chunking**: split on heading hierarchy, then token-bound 400–800 tokens with ~10% overlap; never split code blocks; tables ≤ limit → one chunk, else split by rows repeating the header row; every chunk gets `section_id`, `chunk_index`, `breadcrumb`. Embedded text = breadcrumb + text; displayed text = text.
3. **IDs**: `chunk_id = uuid5(NS, f"{doc_id}:{chunk_index}:{content_hash}")` (Qdrant needs UUID/int).
4. **Re-ingest without a gap**: upsert new points, then delete `doc_id = X AND content_hash != new`.
5. **Index config guard**: collection metadata stores `embedding_model`, `dims`, `pipeline_version`; upsert refuses on mismatch; `rag reindex` re-chunks from `data/parsed/` without re-parsing.
6. **Embedding cache** keyed by `sha256(embed_text + model + dims)` in Postgres; batch ≤ 2048 inputs; async with semaphore + retry; print token/cost estimate before embedding.
7. **Parsing runs in a process pool** (CPU-bound); embedding runs async.
8. **Retrieval order**: rewrite (structured output: standalone query, filters, sub-queries, `needs_retrieval`) → Qdrant Query API prefetch dense + sparse, RRF fusion, top-40 → optional rerank → top-8 → neighbor expansion by `doc_id`+`chunk_index` → max 3 chunks/doc.
9. **Answer contract**: `{answer, citations[{chunk_id, title, uri, page, snippet}], retrieved_chunk_ids, trace_id}`; refuse to answer without citations; retrieved text wrapped in `<document>` tags.
10. **Agent loop**: max 6 tool iterations, token budget, timeout; side-effecting tools have `requires_confirmation=True` and return an `action_proposal` event; execution happens on `POST /chat/{id}/confirm_action`.
11. **Traces table** in Postgres (query, rewritten query, chunk IDs + scores, tool calls, tokens, latency) feeds the UI trace panel and the eval set. No Langfuse for now.
12. **Measure before adding**: eval harness lands before hybrid/rerank/contextual prefixes; each is a config flag turned on only when recall@8 improves on the golden set.

## Milestones

| # | Milestone | Deliverable | Done when |
|---|---|---|---|
| 0 | Scaffold | uv project, docker-compose (Qdrant + Postgres), Alembic init, settings, `LLMClient`, CLI skeleton, Vite+React skeleton, Makefile, ruff/pytest config | `rag status` reports Qdrant + Postgres + OpenAI reachable |
| 1 | Local ingestion, end to end (sync) | `LocalFolderConnector`, parser router, Docling + MD/HTML parsers → IR JSON, chunker, embedding cache, Qdrant schema + upsert, ingest state, `rag ingest --path <dir>` with hash-based incremental re-ingest; Docling benchmark on sample corpus | Re-running on an unchanged dir does no work; changed file reindexes only itself |
| 2 | Baseline RAG | Dense-only retrieval, context assembly, cited answer via `rag query "..."`, traces persisted | Hand-checked correct citations on ~10 questions |
| 3 | Eval harness | `generate_golden.py` (synthetic Q/A from chunks) → curated `golden.jsonl`; `rag eval` prints recall@k/MRR + ragas, diffs vs `baselines/` | Baseline numbers recorded |
| 4 | Retrieval quality (flag by flag) | Sparse BM25 + RRF, query rewrite, rerank plug-ins, neighbor expansion, optional LLM contextual prefixes | Each flag kept only if eval improves; results noted in PLAN.md |
| 5 | Job system | `procrastinate` app, `rag worker`, per-document ingest jobs with retries, `jobs` table + progress; CLI ingest becomes enqueue + follow | Kill worker mid-run → restart resumes cleanly |
| 6 | Agent | Tool registry, `search_documents`, `get_document`, `list_documents`, stubbed `create_ticket`/`draft_email`/`schedule_meeting`/`lookup_employee`, confirmation protocol, rolling memory | Multi-step question answered with 2+ tool calls; action proposed then executed on confirm |
| 7 | API + frontend | FastAPI: `POST /chat` (SSE: token/tool_call/tool_result/citations/action_proposal/done), `confirm_action`, conversations, `POST /documents/upload`, documents, jobs; `X-API-Key`. React: chat with clickable citations, sources drawer, agent trace panel, action confirmation cards, ingestion status page (upload + Sync now) | Non-developer can upload a doc and get a cited answer in the UI |
| 8 | Notion connector + scheduling | Notion blocks → IR, recursive walk from configured root IDs, DB rows as docs, incremental by `last_edited_time`, 3 req/s throttle; `procrastinate` periodic sync per source | Editing a Notion page shows up in answers after next sync |
| 9 | Hardening | Structured JSON logs, error taxonomy, retry policy review, eval in CI, load-test ingest on a large dir | Clean CI; documented known limits |

## Verification (per milestone, summarized)

- **Unit**: chunker (deterministic IDs, table splitting, overlap), IR normalization, query-rewrite parsing — pure pytest.
- **Snapshot**: sample docs in `tests/fixtures/` → expected IR JSON; fails on parser regressions.
- **Integration**: testcontainers Qdrant + Postgres; ingest fixture dir → assert point counts, payload fields, incremental no-op on second run, no-gap re-ingest.
- **LLM**: `vcrpy` cassettes for OpenAI calls; a `--record` mode refreshes them.
- **Eval**: `rag eval` against `eval/golden.jsonl`; baselines committed; CI fails on recall@8 regression > threshold.
- **E2E**: `make up && rag ingest --path ./samples && rag query "..."`; later `npm run dev` → upload → ask → citations link back to source.

## Deferred / explicitly out of scope now

ACLs, OCR, deployment, auth beyond API key, multi-tenant collections, Langfuse/OTel, HyDE/multi-query beyond the rewrite step, webhooks from Notion, Confluence/SharePoint/Drive connectors.

## Status

- **2026-09-22 — Milestone 0 complete** (commit `fd8d1f8`). `rag status` green for Qdrant and Postgres; OpenAI needs a real key in `.env`. Frontend skeleton builds. Directory renamed to `rag-with-doc-parsing`; remote `origin` added, not yet pushed.
- **2026-09-22 — Milestone 1 complete** (uncommitted at time of writing). Local directory ingestion, synchronous:
  `ingest/ir.py` (typed IR, uuid5 ids), parser router + Markdown (markdown-it-py) / HTML (bs4) / Docling
  (PDF, DOCX, PPTX, XLSX; OCR off, heading hierarchy inferred from font style for PDFs), structure-aware
  chunker (breadcrumbs, sentence overlap, code never split, tables split by rows with header repeated),
  Postgres `documents` / `chunks` / `embedding_cache` (migration `0001_ingest_state`), Qdrant collection
  with named dense + sparse slots, payload indexes and the index-config guard stored as collection
  metadata, `rag ingest --path <dir> [--workers N --force --no-prune --dry-run]` and
  `rag reindex [--recreate]`. Parsing runs in a spawn process pool; embedding is async with cache.
  No-gap re-ingest deletes `doc_id = X AND id NOT IN new_ids` (a superset of `content_hash != new`
  that also covers re-chunking with fewer chunks). Tests: 45 (unit + Docling snapshots, skipped when
  models are not cached + integration via testcontainers, skipped without Docker).
  **Docling benchmark** (M2 MacBook, MPS): converter init 1.6 s; 2-page fixture 0.43 s warm;
  58-page synthetic manual with tables 9.0 s ≈ **6.4 pages/s** (`scripts/bench_docling.py`).
  Fast enough that the `pymupdf4llm` fast path was not added; revisit if real corpora are slower.
  Known limits: failed documents are retried on every run; nested PPTX bullets come out flat;
  a 401 from OpenAI fails each document rather than aborting the run.
- **2026-09-22 — Milestone 2 complete.** Baseline RAG, dense-only: `retrieval/retriever.py` (dense
  top-`prefetch_k` → `max_chunks_per_doc` cap → `top_k`), `retrieval/assemble.py` (numbered
  `<document id="n" title uri page>` blocks under `retrieval.max_context_tokens`), `agent/answer.py`
  (structured output `AnswerDraft{answer, citations[int], insufficient_context}` → validated
  `Answer{answer, citations[{chunk_id,title,uri,page,snippet}], retrieved_chunk_ids, trace_id}`;
  refuses without a valid citation), `agent/traces.py` + migration `0002_traces` (query, retrieved
  ids+scores, cited ids, tokens, retrieval/llm/total latency, error), `rag query "..." [--top-k
  --show-context --json]`. `LLMClient.generate_structured` now returns token usage.
  **Acceptance:** `scripts/smoke_query.py eval/smoke_questions.jsonl` — 12/12 questions over the
  fixture corpus answered with the correct source cited (11 factual + 1 refusal), ~1–2 s each.
  Tests: 61 (OpenAI via vcrpy cassettes in `tests/cassettes/`, re-record with `--record-mode=once`).
  Observations for M4: gpt-4.1 often reports citations only in the structured list, not as inline
  `[n]` markers (prompt tweak before the M7 UI); "revenue was 142" dropped the "USD millions" unit
  that sits in the neighbouring chunk — a case for neighbour expansion. Fixed in passing: pruning
  now also deletes the IR JSON so `rag reindex` cannot resurrect removed documents.
- **2026-09-22 — Milestone 3 complete.** Eval harness: `src/ragchat/eval/{golden,metrics,runner,
  generate}.py`; `eval/generate_golden.py` samples indexed chunks and asks the small model for
  questions whose supporting span is verbatim in the chunk → `eval/golden.candidates.jsonl`
  (gitignored) → hand-curated `eval/golden.jsonl` (38 items: 25 synthetic edited, 9 manual incl. a
  multi-chunk case, 4 refusals incl. hard negatives). Golden entries match retrieved chunks by
  file name + chunk_index or verbatim span, so they survive re-ingests and other checkouts.
  `rag eval [--answers] [--ragas] [--compare current] [--save-baseline name] [--out runs.jsonl]`:
  recall@{1,3,5,8,10}, hit@k, MRR; with answers: citation precision/hit rate, false-refusal rate,
  refusal accuracy; with ragas 0.4 (via our OpenAI client): faithfulness, answer relevancy,
  context precision/recall. Exit 1 when recall@k drops > `eval.regression_threshold` vs the
  baseline. **Baseline recorded** in `eval/baselines/current.json` (commit `41f1391` config,
  dense-only): MRR 0.926, recall@1 0.838, recall@3..10 1.000, citation precision 1.0, refusal
  accuracy 1.0, faithfulness 0.996, answer relevancy 0.940, context precision 0.886.
  The fixture corpus (24 chunks) saturates recall@≥3; recall@1 / MRR / context precision are the
  numbers M4 flags must move — a larger corpus should be ingested before M4 for meaningful deltas.
  Notes: ragas pulls in the langchain stack and pins `openai` to 3.3 (eval-only dependency; the
  judge uses the project's `AsyncOpenAI` client, not LLMClient). Judges are given breadcrumb +
  text, exactly what the model sees — scoring bare chunk text produced false "unfaithful" verdicts.
- **Next: Milestone 4** — retrieval quality flag by flag (sparse BM25 + RRF, query rewrite, rerank,
  neighbour expansion, contextual prefixes), each kept only if `rag eval` improves.
