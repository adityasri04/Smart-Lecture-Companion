"""
RAG (Retrieval-Augmented Generation) pipeline over the lecture transcript.
Uses sentence-transformers for embedding and FAISS for in-memory vector search.
This is the open-source equivalent of how NotebookLM grounds answers in source material.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_rag_index(transcript: str) -> dict:
    """
    Build a FAISS vector index from a lecture transcript.
    Call this once after transcription; store the result in st.session_state.

    Steps:
      1. Split transcript into overlapping chunks (400 chars, 80 char overlap)
      2. Embed each chunk with all-MiniLM-L6-v2 (~90MB, downloads once)
      3. Store vectors in a FAISS flat L2 index (in memory)

    Returns a dict with:
      - 'index':   FAISS IndexFlatL2 object
      - 'chunks':  list of text strings (parallel to index vectors)
      - 'model':   SentenceTransformer instance (reused for query encoding)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(transcript)

    embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    embeddings  = embed_model.encode(chunks, show_progress_bar=False)
    embeddings  = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return {"index": index, "chunks": chunks, "model": embed_model}


def retrieve_chunks(question: str, rag_state: dict, top_k: int = 4) -> list[str]:
    """
    Retrieve the top_k most relevant transcript chunks for a question.

    Args:
        question:  The user's question string.
        rag_state: The dict returned by build_rag_index().
        top_k:     Number of chunks to return.

    Returns:
        List of text strings — the most relevant excerpts from the transcript.
    """
    query_vec = rag_state["model"].encode([question])
    query_vec = np.array(query_vec, dtype="float32")

    distances, indices = rag_state["index"].search(query_vec, top_k)
    return [rag_state["chunks"][i] for i in indices[0] if i < len(rag_state["chunks"])]
