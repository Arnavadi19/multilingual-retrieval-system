"""
Indexer module for creating and managing the vector index.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from config import INDEX_DIR, INDEX_FILENAME, METADATA_FILENAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndex:
    """Manages the vector index for multilingual document retrieval."""
    
    def __init__(self, index_dir: Path = INDEX_DIR):
        """
        Initialize the VectorIndex.
        
        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.embeddings = None
        self.metadata = {
            'doc_ids': [],
            'languages': [],
            'doc_texts': [],
            'num_documents': 0,
            'embedding_dim': 0
        }
    
    def build(self, embeddings: np.ndarray, 
              doc_ids: List[str], 
              languages: List[str],
              doc_texts: Optional[List[str]] = None) -> None:
        """
        Build the index from embeddings and metadata.
        
        Args:
            embeddings: NumPy array of document embeddings (n_docs, embedding_dim)
            doc_ids: List of document IDs
            languages: List of document languages
            doc_texts: List of document texts (optional, for retrieval display)
        """
        assert len(doc_ids) == len(languages) == embeddings.shape[0], \
            "Mismatch between number of embeddings and metadata"
        
        if doc_texts is not None:
            assert len(doc_texts) == embeddings.shape[0], \
                "Mismatch between number of embeddings and document texts"
        
        self.embeddings = embeddings
        self.metadata = {
            'doc_ids': doc_ids,
            'languages': languages,
            'doc_texts': doc_texts if doc_texts is not None else [],
            'num_documents': len(doc_ids),
            'embedding_dim': embeddings.shape[1]
        }
        
        logger.info(f"Index built with {self.metadata['num_documents']} documents")
    
    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self.embeddings is None:
            raise ValueError("No index to save. Build the index first.")
        
        # Save embeddings as compressed numpy file
        embeddings_path = self.index_dir / INDEX_FILENAME
        np.savez_compressed(embeddings_path, embeddings=self.embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save metadata as JSON
        metadata_path = self.index_dir / METADATA_FILENAME
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self) -> bool:
        """
        Load the index and metadata from disk.
        
        Returns:
            True if loading was successful, False otherwise
        """
        embeddings_path = self.index_dir / INDEX_FILENAME
        metadata_path = self.index_dir / METADATA_FILENAME
        
        if not embeddings_path.exists() or not metadata_path.exists():
            logger.warning("Index files not found")
            return False
        
        try:
            # Load embeddings
            data = np.load(embeddings_path)
            self.embeddings = data['embeddings']
            logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {self.metadata['num_documents']} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> tuple:
        """
        Search the index for documents most similar to the query.
        
        Args:
            query_embedding: Query embedding vector (1, embedding_dim) or (embedding_dim,)
            top_k: Number of top results to return
        
        Returns:
            Tuple of (indices, scores) for top-k results
        """
        if self.embeddings is None:
            raise ValueError("No index loaded. Build or load an index first.")
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def get_document_info(self, index: int) -> Dict[str, str]:
        """
        Get metadata for a document at a specific index.
        
        Args:
            index: Index of the document
        
        Returns:
            Dictionary with document metadata
        """
        result = {
            'doc_id': self.metadata['doc_ids'][index],
            'language': self.metadata['languages'][index]
        }
        
        # Add text if available
        if self.metadata.get('doc_texts') and index < len(self.metadata['doc_texts']):
            result['text'] = self.metadata['doc_texts'][index]
        
        return result
    
    def index_exists(self) -> bool:
        """Check if index files exist on disk."""
        embeddings_path = self.index_dir / INDEX_FILENAME
        metadata_path = self.index_dir / METADATA_FILENAME
        return embeddings_path.exists() and metadata_path.exists()