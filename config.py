"""
Configuration file for the Multilingual Information Retrieval System.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Supported languages and their MIRACL dataset identifiers
LANGUAGES = {
    'hindi': 'miracl/hi',
    'bengali': 'miracl/bn',
    'telugu': 'miracl/te'
}

# Language code mapping
LANG_CODE_MAP = {
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu'
}

# Model configuration
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
EMBEDDING_DIMENSION = 768  # Dimension of the multilingual model

# Retrieval parameters
DEFAULT_TOP_K = 10  # Number of documents to retrieve
BATCH_SIZE = 32  # Batch size for encoding documents

# Index configuration
INDEX_FILENAME = "multilingual_index.npz"
METADATA_FILENAME = "document_metadata.json"
FAISS_INDEX_FILENAME = "faiss_index.bin"

# Indexing backend ('numpy' or 'faiss')
DEFAULT_INDEX_BACKEND = 'numpy'  # Can be overridden via CLI
USE_GPU_FOR_FAISS = True  # Use GPU for FAISS if available

# Sample size for quick testing (set to None to use full corpus)
# For production, set to None. For testing, use a smaller number like 5000
CORPUS_SAMPLE_SIZE = 10000

# Device configuration (auto-detect GPU)
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging
LOG_LEVEL = "INFO"
