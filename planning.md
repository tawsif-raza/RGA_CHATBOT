# Project Plan: Jargon-Aware Enterprise RAG

## 1. Project Objective
Build a production-grade Retrieval-Augmented Generation (RAG) system that connects structured corporate data and messy internal documentation to a fine-tuned, domain-specific language model. The system will synthesize jargon-heavy internal records (from the `EnterpriseRAG-Bench` dataset) into clear, accurate answers.

## 2. System Architecture
*   **Database (Structured Metadata):** Microsoft SQL Server running in a Docker container on an AWS EC2 instance.
*   **Vector Database (Unstructured Data):** Pinecone index for storing semantic embeddings of chunked enterprise documents.
*   **Orchestration & Retrieval:** FastAPI backend utilizing LangChain to route queries, embed text, and fetch context.
*   **Generation Engine:** A Qwen2.5-0.5B-Instruct model, fine-tuned using QLoRA and TRL SFTTrainer, deployed locally to synthesize retrieved documents.
*   **User Interface:** Streamlit web application for querying and visualizing the retrieved context alongside the model's answer.

## 3. Directory Structure
```text
enterprise-rag-project/
├── backend/
│   ├── api.py               # FastAPI application and endpoints
│   ├── retriever.py         # LangChain and Pinecone connection logic
│   └── requirements.txt
├── data/
│   ├── prepare_data.py      # Script to download/merge EnterpriseRAG-Bench
│   └── enterprise_rag_finetune.jsonl # Formatted data for fine-tuning
├── finetuning/
│   ├── train_qlora.py       # TRL SFTTrainer and PEFT configuration
│   └── export_model.py      # Script to merge LoRA weights and export
├── frontend/
│   ├── app.py               # Streamlit chat interface
│   └── requirements.txt
├── infrastructure/
│   ├── docker-compose.yml   # MSSQL Server configuration
│   └── init.sql             # SQL schema definitions for metadata
└── ingestion/
    ├── extract_sql.py       # Pulls metadata from MSSQL
    └── embed_pinecone.py    # Chunks text and pushes vectors to Pinecone
    