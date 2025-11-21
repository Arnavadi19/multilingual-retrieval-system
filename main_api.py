# main_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os
import logging

# --- 1. NEW IMPORTS ---
# You need to import the classes your Retriever depends on, 
# and the Retriever itself. Adjust the 'core.' prefix based on your file structure.
from core.embedder import MultilingualEmbedder
from core.indexer import VectorIndex
from core.retriever import CrossLingualRetriever 
from data_loader import DataLoader # Assuming DataLoader is used for loading corpus texts

logger = logging.getLogger("fastapi")
logger.setLevel(logging.INFO)

# --- Define Constants ---
# Use a sensible default path for your model and index
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" # Placeholder for your actual model name
INDEX_DIR = "index" # Placeholder for your saved index
CORPUS_PATH = "data/corpus.jsonl" # Placeholder for your corpus file

# --- Initialization Function ---
def initialize_retrieval_system() -> CrossLingualRetriever:
    """Loads model, index, and corpus texts once."""
    try:
        logger.info("Starting System Initialization...")
        
        # 1. Initialize Embedder 
        embedder = MultilingualEmbedder(model_name=MODEL_NAME) 
        
        # 2. Load Index
        index = VectorIndex(index_dir=INDEX_DIR)
        if not index.load():
            raise FileNotFoundError(f"Failed to load index files from directory: {INDEX_DIR}")

        # 3. Load Corpus Texts (CORRECTION IS HERE)
        # The DataLoader uses language list from config.py by default.
        data_loader = DataLoader() 
        
        # NOTE: Your logic requires loading corpus documents into the DataLoader instance 
        # before calling get_corpus_texts().
        # You need an explicit call to load_corpus() here:
        data_loader.load_corpus() 
        
        corpus_texts = data_loader.get_corpus_texts() # get_corpus_texts uses loaded data
        
        # 4. Initialize Retriever and set the corpus texts
        retriever = CrossLingualRetriever(embedder=embedder, index=index)
        retriever.set_corpus_texts(corpus_texts)
        
        logger.info("System Initialization Complete. Index and Model Ready.")
        return retriever
        
    except Exception as e:
        logger.error(f"FATAL ERROR during system initialization: {e}")
        raise

# --- Initialize Global Retriever Instance ---
retriever: CrossLingualRetriever = initialize_retrieval_system()

# --- FastAPI App Setup ---
app = FastAPI(title="Multilingual Retrieval API")

# --- CORS Configuration (Same as before) ---
origins = ["http://localhost:8000", 
           "http://localhost:5173", 
           "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (Same as before, but mapping to your result keys) ---
class SearchResult(BaseModel):
    rank: int
    doc_id: str
    language: str
    score: float
    text: Optional[str] = None # Optional because return_full_text is default False

# --- The Search Endpoint ---
@app.get("/api/search", response_model=List[SearchResult])
async def search_endpoint(query: str, top_k: int = 10):
    """
    Performs a cross-lingual search using the global retriever instance.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    try:
        logger.info(f"API received query: '{query}'")
        
        # --- 5. Call the core retrieval logic ---
        # The return_full_text=True ensures 'text' is included for the frontend
        results = retriever.retrieve(
            query=query, 
            top_k=top_k, 
            return_full_text=True
        )
        
        # The list of dicts returned by retriever.retrieve() already matches 
        # the SearchResult Pydantic model keys (rank, doc_id, language, score, text).
        return results

    except Exception as e:
        logger.error(f"Error during search for query '{query}': {e}")
        # Return a standard 500 status on internal error
        raise HTTPException(status_code=500, detail="Internal server error during retrieval.")


# --- Server Run Command ---
# uvicorn main_api:app --reload --host 0.0.0.0 --port 8000