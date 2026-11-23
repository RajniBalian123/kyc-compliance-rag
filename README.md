# KYC Compliance RAG — Demo

A retrieval system that answers regulatory questions from a rulebook corpus, grounded in
the retrieved passages, with the source text and its regulatory category visible.

This is a personal demo project. It rebuilds, from scratch and with synthetic data, the
retrieval pattern behind a client-onboarding compliance system: chunk a corpus of rules,
tag each chunk with its regulatory category, embed it, and answer questions from the
retrieved passages — never from the model's memory.

## How it works

1. **Ingest.** Rule documents are loaded from `data/sample_docs/`.
2. **Chunk.** Each document is split into chunks of at most 600 characters, with
   100-character overlap, breaking at paragraph boundaries where possible so no sentence
   is cut in half.
3. **Categorize.** Each chunk is tagged with a regulatory category derived from its
   source filename: AML, PEP, KYC, FATF, or GENERAL.
4. **Embed.** Each chunk becomes a 384-dimensional vector with `all-MiniLM-L6-v2`
   (SentenceTransformers, runs locally on CPU, no API call — the text never leaves the
   machine).
5. **Store.** Chunks, metadata, and vectors live in a ChromaDB collection indexed with
   HNSW and cosine similarity, persisted to disk.
6. **Query.** A question is embedded, the top 5 chunks are retrieved, and the answer is
   built from those chunks only, with text, category, and relevance score shown.

## Quick start

```bash
pip install -r requirements.txt
python -m src.query "What due diligence is required for a politically exposed person?"
python -m src.run_eval
```

The first run ingests the sample corpus automatically (the store persists to disk, so
re-ingestion only happens when it is empty).

## Evaluation

`evals/golden_questions.json` holds a small golden set: questions with the expected
regulatory category and the key phrases the retrieved passages should contain.
`src/run_eval.py` reports recall@5 per question — whether the expected category and
phrases show up in the top 5 retrieved chunks. Retrieval quality is measured, not assumed.

## What this proves

- Dense retrieval over a small, sensitive corpus with local embeddings.
- Chunking that preserves meaning: fixed size with overlap plus paragraph-aware breaks.
- Category metadata carried through retrieval so an agent can see *which* rule a passage
  came from.
- A grounded-answer contract: no retrieved passage, no answer.

## What production adds

The local build is dense-only retrieval. A production version replaces ChromaDB with a
managed search service that adds the four things legal work demands: BM25 keyword search
(for exact citations like "s.9.3(1)(a)"), a semantic reranker, metadata filtering (search
only PEP-category chunks), and hybrid fusion of dense and keyword rankings.

## Notes

- The documents in `data/sample_docs/` are short fictional passages written for this
  demo. They are not real regulatory text and this project is not legal advice.
- No company code, client data, or proprietary documents are in this repository.
