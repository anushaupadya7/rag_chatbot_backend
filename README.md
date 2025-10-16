
# RAG Chatbot Backend

This project is a backend system for a Retrieval-Augmented Generation (RAG) chatbot. It ingests various document types (PDFs, text files, etc.), processes them into a searchable vector knowledge base, and uses a Large Language Model (LLM) to answer user queries based on the ingested content.

The system is designed to be modular, with separate components for data loading, vector storage, and search/synthesis.

## Features

- **Multi-Format Document Ingestion**: Supports loading documents from various formats including:
  - PDF (`.pdf`)
  - Text (`.txt`)
  - CSV (`.csv`)
  - Microsoft Word (`.docx`)
  - Microsoft Excel (`.xlsx`)
  - JSON (`.json`)
- **Efficient Vector Storage**: Uses FAISS (Facebook AI Similarity Search) to create and manage a local vector store for fast and efficient similarity searches.
- **State-of-the-Art Embeddings**: Leverages `sentence-transformers` (`all-MiniLM-L6-v2` model) to generate high-quality embeddings for document chunks.
- **Fast LLM Integration**: Integrates with the Groq API for extremely fast inference using models like `gemma2-9b-it` for the generation step.
- **Modular Architecture**: The code is organized into logical modules for data loading (`data_loader.py`), vector store management (`vectorstore.py`), and the core RAG logic (`search.py`).

## Project Structure

```
rag_chatbot_backend/
├── app.py                # Main application entry point and example usage
├── data/                 # Directory to store your source documents
├── faiss_store/          # Default directory for the persisted FAISS vector store
├── notebook/             # Jupyter notebooks for experimentation and development
│   ├── rag.py
│   └── document.py
├── src/                  # Source code modules
│   ├── data_loader.py    # Handles loading and processing of different file types
│   ├── search.py         # Core RAG search and summarization logic
│   └── vectorstore.py    # FAISS vector store implementation
├── requirements.txt      # Project dependencies
└── .env                  # For storing API keys (e.g., GROQ_API_KEY)
```
---
Data Ingestion Pipeline
![RAG Pipeline](./Data_Ingestion_Pipeline.png?WT.mc_id=a)

Data Retrieval Pipeline
![RAG Pipeline](./Data_Retrieval_Pipeline.png?WT.mc_id=a)

Multimodal RAG
![RAG Pipeline](./RAG.png?WT.mc_id=a)

Multimodal RAG (PDF data)
![RAG Pipeline](./Multimodal_RAG.png?WT.mc_id=a)

---

![RAG Pipeline](./multimodal_pdf_rag.png?WT.mc_id=a)

Agentic RAG

![RAG Pipeline](./agentic_rag.png?WT.mc_id=a)


---

## Getting Started

### Prerequisites

- Python 3.8+
- An API key from Groq

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd rag_chatbot_backend
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing all necessary packages like `langchain`, `faiss-cpu`, `sentence-transformers`, `langchain-groq`, `python-dotenv`, etc.)*

3.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

4.  **Add your documents:**
    Place all the documents you want to query into the `data/` directory. The `data_loader.py` script will automatically find and process them.

### Running the Application

The `app.py` file provides an example of how to use the RAG system.

1.  The first time you run the application, it needs to build the vector store from your documents. In `app.py`, uncomment the line `store.build_from_documents(docs)`.
2.  Run the application:
    ```bash
    python app.py
    ```

This will load the documents, build and save the FAISS index, perform a search for the query "What is supervised learning?", and print the generated summary. For subsequent runs, you can comment out the `build_from_documents` line to load the pre-built index, which is much faster.
