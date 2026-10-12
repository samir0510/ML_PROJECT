# ============================================================
# STEP 2: Chunks → Vector Store (FAISS)
# ============================================================
# This file takes the text chunks from Step 1 and converts
# them into "embeddings" — mathematical vectors of numbers.
#
# Chunks about similar topics will have vectors close together.
# When a user asks a question, we convert the question into a
# vector and find the top closest chunks in our vector store.
#
# We use FAISS (Facebook AI Similarity Search) to store and
# search vectors locally on your computer.
# ============================================================

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Project root path (parent of chatbot/ folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# Directory where FAISS vector index is saved
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def get_embedding_model(api_key: Optional[str] = None):
    """
    Initialize Google Gemini Embeddings.
    Uses 'models/gemini-embedding-001'.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() == "" or key == "your_gemini_api_key_here":
        raise ValueError(
            "[ERROR] GEMINI_API_KEY is not set!\n"
            "1. Copy .env.example to .env\n"
            "2. Open .env and add your key: GEMINI_API_KEY=...\n"
            "3. Get your key at: https://aistudio.google.com/app/apikey"
        )

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )


def build_vector_store(chunks: List[Document], api_key: Optional[str] = None) -> FAISS:
    """
    Convert chunks to embeddings and save them to local FAISS index.
    Processes in safe batches to stay within Gemini free tier rate limits (100 req/min).

    Args:
        chunks : List of Document chunks from step1_pdf_loader.py
        api_key: Optional Gemini API key (defaults to .env)

    Returns:
        FAISS vector store
    """
    embeddings = get_embedding_model(api_key)

    print(f"[EMBEDDINGS] Creating vector embeddings for {len(chunks)} chunks...")
    print("             Processing in batches to stay within free-tier quota...")

    batch_size = 40
    vector_store = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   Embedding batch {batch_num} of {total_batches} ({len(batch)} chunks)...")

        # Retry loop for rate limits
        for attempt in range(4):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embedding=embeddings)
                else:
                    temp_store = FAISS.from_documents(batch, embedding=embeddings)
                    vector_store.merge_from(temp_store)
                break  # Batch succeeded
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_time = 35
                    print(f"   [RATE LIMIT] Free quota paused. Waiting {wait_time}s before resuming...")
                    time.sleep(wait_time)
                else:
                    raise e

        # Small pause between batches if more remain
        if i + batch_size < len(chunks):
            time.sleep(2)

    # Save index to disk so we only have to build it once
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))

    print(f"[OK] Vector store successfully created and saved to: {VECTOR_STORE_DIR}")
    return vector_store


def load_vector_store(api_key: Optional[str] = None) -> FAISS:
    """
    Load existing FAISS index from disk.

    Args:
        api_key: Optional Gemini API key (defaults to .env)

    Returns:
        FAISS vector store
    """
    if not is_vector_store_ready():
        raise FileNotFoundError(
            f"[ERROR] Vector store not found at: {VECTOR_STORE_DIR}\n"
            "Please run build_vector_store() first."
        )

    embeddings = get_embedding_model(api_key)

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"[OK] Vector store loaded from disk: {VECTOR_STORE_DIR}")
    return vector_store


def is_vector_store_ready() -> bool:
    """
    Returns True if the FAISS index files (index.faiss, index.pkl) exist.
    """
    faiss_file = VECTOR_STORE_DIR / "index.faiss"
    pkl_file = VECTOR_STORE_DIR / "index.pkl"
    return faiss_file.exists() and pkl_file.exists()


# ── Quick Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    from .step1_pdf_loader import load_pdf_as_chunks

    pdf_files = list(PROJECT_ROOT.glob("*.pdf"))
    if not pdf_files:
        print("[ERROR] No PDF file found in project folder.")
    else:
        pdf_path = str(pdf_files[0])
        print(f"Using PDF: {pdf_path}")

        try:
            chunks = load_pdf_as_chunks(pdf_path)
            vs = build_vector_store(chunks)

            query = "What are the symptoms of Late Blight?"
            print(f"\n[TEST SEARCH] Query: '{query}'")
            results = vs.similarity_search(query, k=2)

            for i, res in enumerate(results, start=1):
                page = res.metadata.get("page", "?")
                print(f"\n--- Match {i} (Page {page}) ---")
                print(res.page_content[:250] + "...")

        except Exception as e:
            print(f"\n[NOTICE] {e}")
