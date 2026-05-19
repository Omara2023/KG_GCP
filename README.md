# Custom GCP/Vertex AI Document Intelligence & RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) infrastructure pipeline built on Google Cloud Platform (GCP). The system processes unstructured PDF documents via ingestion pipelines, applies advanced multi-strategy retrieval mechanics, and utilises an n-step LLM critic loop to enforce semantic data grounding.

> **🛠️ Architectural Status (May 2026):** **Work-In-Progress / Architectural Pivot.** The stable, functional core infrastructure was deployed to Cloud Run in August 2025. The repository is currently sitting mid-refactor as part of a planned migration from naive vector proximity search to a hybrid GraphRAG pipeline (Neo4j). As a result, recent commits on the main branch reflect active architectural changes and are non-functional.

## System Architecture & Mechanics

### 1. Ingestion & Document Processing Pipeline
- **Storage & Triggers:** Raw PDF payloads are stored via Google Cloud Storage (GCS) buckets.
- **Data Integrity:** Input validation and structural typing enforced across the extraction pipeline using **Pydantic** data models.

### 2. Advanced Retrieval Engine
To maximise semantic relevance, the engine bypasses naive vector lookup by orchestrating a sequential retrieval pipeline:
- **Query Transformation:** Implements automated Prompt Rewriting and Step-Back abstraction to optimise raw user inputs with options for:
- **Pre-retrieval Choice:** Selects from multiple pre-retrieval strategies to effectively aggregate context fragments includeing:
- **Decomposition:** Executes sub-query generation to break down complex, multi-part prompts into separate  search tasks.
- **Step back:** Rewriting overly specifc queries to optimise context matching.

### 3. Self-Correcting N-Step Critic Loop
To suppress hallucination patterns in retrieval workflows, the generation phase runs inside a deterministic evaluation state-machine:
1. **Generation:** The base Gemini client generates a context-bounded response candidate.
2. **Evaluation:** The LLM Critic parses the generated payload against validation criteria (alignment etc).
3. **Loop/Terminate:** If the candidate fail criteria, the engine mutates the next search query based on the failure delta and repeats the loop. The process automatically terminates prematurely upon a "satisfactory" grade or drops to a safe fallback after $N$ iterations.

### 4. Enterprise Observability & Logging
- **BigQuery Logging:** Every state transition, raw query text, vector distance metric, and critic evaluation payload is streamed to BigQuery for historical auditing, token costing, and evaluation metrics.
- **Deployment:** Microservices are containerised with Docker and fully hosted via **Google Cloud Run** for serverless scaling.

## Tech Stack

- **Cloud Infrastructure:** Google Cloud Platform (GCP - GCS, Cloud Run, BigQuery)
- **Core Models:** Vertex AI (Gemini Client & Text Embeddings)
- **Data Validation & Orchestration:** Python, Pydantic, Vector Search
- **Data Logging:** BigQuery, Cloud Logging

## Active Refactor: GraphRAG Migration

The current vector-only retrieval framework is undergoing an infrastructure overhaul to transition into a hybrid **GraphRAG** pipeline:
- **Target Database:** Neo4j (Graph Database)
- **Objective:** Extracting entities and explicit semantic relationships (parent child retrieval etc) from text chunks during the ingestion phase to build a global knowledge graph, enabling deterministic multi-hop reasoning that vector-only proximity searches struggle to resolve.