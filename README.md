# Extraction POC

Prototype that ingests a financial PDF, enriches structured chunks with an LLM, stores them in a local Qdrant vector store, and uses a LangGraph agent to retrieve and synthesize answers as JSON.

## Repository Layout
- main_agentnvda.py: CLI entrypoint; prompts for a PDF path and optional re-ingestion, runs the LangGraph workflow, writes results.
- setup_config.py: Shared constants, environment bootstrapping (API keys), and utility helpers.
- ingest_vectorize.py: Parses PDFs with Unstructured, chunks by title, enriches chunks via ChatOllama, embeds with fastembed + BM25, and populates Qdrant.
- rag_tools.py: RAG “librarian” tool (query optimization, hybrid search, reranking via CrossEncoder).
- agent_graph.py: LangGraph state machine (planner → tool executor → synthesizer) plus graph export helper.
- schema.json: Example JSON schema for company metadata.
- input/: Sample source PDF (NVIDIAAn.pdf).
- output/: Sample extraction output (extracted.json).
- enriched_chunks.json, qdrant_db/: Cached enriched chunks and persisted Qdrant collection.

## Prerequisites
- Python 3.10+ recommended.
- Local Ollama with models referenced in code: `llama3.1:8b`, `mistral-small3.2`, `gpt-oss:20b` (adjust in code if using alternatives).
- Graphviz/pygraphviz installed for graph PNG export (optional).
- Environment variables: `OPENAI_API_KEY` (prompted if missing), `LANGCHAIN_API_KEY` or `LANGSMITH_API_KEY`.

Install deps:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
```bash
python main_agentnvda.py
```
- When prompted, press Enter to use the default sample PDF (`input/NVIDIAAn.pdf`) or provide a path.
- Choose whether to re-run ingestion (`Y`) or reuse the existing Qdrant DB/cache (`N`).
- The agent builds a plan, retrieves with the librarian tool, and synthesizes a JSON answer.

Outputs:
- output/extracted.json: Final extracted JSON.
- extraction_agent_graph.png: Graphviz export of the LangGraph (if pygraphviz is available).
- qdrant_db/: Local Qdrant store populated during ingestion.

## How It Works (High Level)
1) Parse PDF with Unstructured → title-aware chunks.
2) Enrich each chunk with summary/keywords/questions (ChatOllama structured output).
3) Embed enriched text (dense + sparse) and upsert into Qdrant.
4) At query time, optimize the question, retrieve hybrid results, rerank with CrossEncoder.
5) LangGraph planner decides tool calls; synthesizer LLM produces JSON response.

## Notes
- Cached `enriched_chunks.json` and `qdrant_db/` allow fast reuse; delete or choose re-ingestion to rebuild.
- Adjust model names or temperatures in the Python modules if your local Ollama setup differs.
- The current sample output is in `output/extracted.json` for NVIDIA’s Q2 FY2026 results.
