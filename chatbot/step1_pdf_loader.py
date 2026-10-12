# ============================================================
# STEP 1: PDF → Text Chunks
# ============================================================
# This file reads your PDF and splits it into small pieces
# called "chunks". Each chunk is ~750 characters long.
# Smaller chunks help the AI find exact answers faster.
# ============================================================

import sys
from pathlib import Path
from typing import List

# Ensure safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pdfplumber                                    # Reads PDF files
from langchain_core.documents import Document        # A LangChain "document" is just text + metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into chunks


def load_pdf_as_chunks(pdf_path: str, chunk_size: int = 750, chunk_overlap: int = 75) -> List[Document]:
    """
    Read a PDF file and split it into small text chunks.

    Args:
        pdf_path     : Full path to your PDF file
        chunk_size   : How many characters per chunk (default 750)
        chunk_overlap: How many characters each chunk shares with the next
                       (overlap keeps sentences from being cut mid-way)

    Returns:
        A list of LangChain Document objects.
        Each Document has:
          - page_content : the actual text of the chunk
          - metadata     : {'source': filename, 'page': page number}
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"[PDF] Reading file: {pdf_file.name}")

    # ── Step 1: Extract raw text from every page ─────────────────────────
    all_documents = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Found {total_pages} pages")

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            # Skip empty pages (some PDFs have blank pages)
            if not text or text.strip() == "":
                continue

            # Create one Document per page (we'll chunk it next)
            doc = Document(
                page_content=text.strip(),
                metadata={
                    "source": pdf_file.name,   # Which file this came from
                    "page": page_num,           # Which page number
                }
            )
            all_documents.append(doc)

    print(f"   Extracted text from {len(all_documents)} pages")

    # ── Step 2: Split each page into smaller chunks ───────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_documents)

    print(f"   Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    print(f"[OK] PDF loading complete!\n")

    return chunks


# ── Quick test: run this file directly to see what the chunks look like ──
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_files = list(project_root.glob("*.pdf"))

    if not pdf_files:
        print("[ERROR] No PDF found in project directory.")
    else:
        pdf_path = str(pdf_files[0])
        chunks = load_pdf_as_chunks(pdf_path)

        print(f"Total chunks created: {len(chunks)}")
        print("\n--- Example: First 3 chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[Chunk {i+1}] (Page {chunk.metadata['page']})")
            print(f"Text: {chunk.page_content[:200]}...")
