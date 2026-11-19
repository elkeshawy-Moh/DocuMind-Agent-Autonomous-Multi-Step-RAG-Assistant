# ==============================================================================
# ingest.py: DOCUMENT PROCESSING & VECTOR DATABASE PIPELINE
# ==============================================================================

import os
import re
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIRECTORY = "./documind_chroma_db"

# Initialize the free HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def clean_extracted_text(text: str) -> str:
    """
    Step 2: Text Cleaning.
    Removes excessive whitespace, broken newlines, and non-printable characters.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_vector_store():
    """
    Loads the existing ChromaDB vector store from disk if available.
    """
    if os.path.exists(DB_DIRECTORY):
        return Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=embeddings
        )
    return None

def process_uploaded_pdfs(pdf_files) -> str:
    """
    End-to-End Ingestion Pipeline:
    1. Text Extraction -> 2. Cleaning -> 3. Chunking -> 4. Embedding -> 5. Vector Storage
    """
    if not pdf_files:
        return "⚠️ Please upload at least one PDF file first."

    try:
        # Clear previous database if rebuilding with new files
        if os.path.exists(DB_DIRECTORY):
            shutil.rmtree(DB_DIRECTORY)

        all_documents = []

        # Step 1: Text Extraction from multiple uploaded PDFs
        for file_obj in pdf_files:
            file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
            file_name = os.path.basename(file_path)

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Step 2: Text Cleaning & Metadata Enrichment
            for page in pages:
                page.page_content = clean_extracted_text(page.page_content)
                page.metadata["source_file"] = file_name
                # Convert 0-based page index to 1-based human-readable page number
                page.metadata["page_number"] = page.metadata.get("page", 0) + 1

            all_documents.extend(pages)

        if not all_documents:
            return "⚠️ Could not extract readable text from the uploaded PDFs."

        # Step 3: Chunking (Splitting text into overlapping semantic segments)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(all_documents)

        # Step 4 & 5: Generate Embeddings & Store in Chroma Vector Database
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIRECTORY
        )

        return (
            f"✅ **Knowledge Base Built Successfully!**\n"
            f"- **Files Processed:** {len(pdf_files)}\n"
            f"- **Total Pages Extracted:** {len(all_documents)}\n"
            f"- **Total Semantic Chunks Indexed:** {len(chunks)}"
        )

    except Exception as e:
        return f"❌ Error during document processing: {str(e)}"
