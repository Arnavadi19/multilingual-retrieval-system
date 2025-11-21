"""
Embedding generator module using sentence-transformers.
"""

import numpy as np
import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, DEVICE, BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """Generates dense embeddings for multilingual text using sentence-transformers."""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """
        Initialize the embedder with a sentence-transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = BATCH_SIZE,
               show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        logger.info(f"Encoding complete. Shape: {embeddings.shape}")
        return embeddings
    
    def encode_corpus(self, corpus_texts: List[str], 
                     batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for a corpus of documents.
        
        Args:
            corpus_texts: List of document texts
            batch_size: Batch size for encoding
        
        Returns:
            NumPy array of document embeddings
        """
        logger.info(f"Encoding corpus of {len(corpus_texts)} documents...")
        return self.encode(corpus_texts, batch_size=batch_size, show_progress=True)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
        
        Returns:
            NumPy array with shape (1, embedding_dim)
        """
        return self.encode(query, show_progress=False)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
