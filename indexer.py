"""
Indexer module for creating and managing the vector index.
Supports both NumPy (exact search) and FAISS (fast approximate/exact search).
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from config import INDEX_DIR, INDEX_FILENAME, METADATA_FILENAME, FAISS_INDEX_FILENAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import FAISS, but make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Only NumPy backend will be available.")
    logger.warning("Install with: pip install faiss-cpu  (or faiss-gpu for GPU support)")


class VectorIndex:
    """Manages the vector index for multilingual document retrieval."""
    
    def __init__(self, index_dir: Path = INDEX_DIR, backend: str = 'numpy', use_gpu: bool = False):
        """
        Initialize the VectorIndex.
        
        Args:
            index_dir: Directory to store index files
            backend: Indexing backend ('numpy' or 'faiss')
            use_gpu: Use GPU for FAISS (only applicable if backend='faiss')
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Validate backend
        if backend not in ['numpy', 'faiss']:
            raise ValueError(f"Invalid backend: {backend}. Must be 'numpy' or 'faiss'")
        
        if backend == 'faiss' and not FAISS_AVAILABLE:
            logger.error("FAISS backend requested but FAISS is not installed!")
            logger.error("Install with: pip install faiss-cpu  (or faiss-gpu for GPU)")
            raise ImportError("FAISS not available. Install faiss-cpu or faiss-gpu")
        
        self.backend = backend
        self.use_gpu = use_gpu and backend == 'faiss'
        
        # NumPy storage
        self.embeddings = None
        
        # FAISS index
        self.faiss_index = None
        
        # Metadata
        self.metadata = {
            'doc_ids': [],
            'languages': [],
            'doc_texts': [],
            'num_documents': 0,
            'embedding_dim': 0,
            'backend': backend
        }
        
        logger.info(f"Initialized VectorIndex with backend: {backend}")
        if self.use_gpu:
            if FAISS_AVAILABLE and faiss.get_num_gpus() > 0:
                logger.info(f"GPU enabled for FAISS (GPUs available: {faiss.get_num_gpus()})")
            else:
                logger.warning("GPU requested but not available. Falling back to CPU.")
                self.use_gpu = False
    
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
        
        # Store embeddings (always keep for NumPy compatibility and metadata)
        self.embeddings = embeddings
        
        # Build FAISS index if using FAISS backend
        if self.backend == 'faiss':
            self._build_faiss_index(embeddings)
        
        # Store metadata
        self.metadata = {
            'doc_ids': doc_ids,
            'languages': languages,
            'doc_texts': doc_texts if doc_texts is not None else [],
            'num_documents': len(doc_ids),
            'embedding_dim': embeddings.shape[1],
            'backend': self.backend
        }
        
        logger.info(f"Index built with {self.metadata['num_documents']} documents using {self.backend} backend")
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        
        dim = embeddings.shape[1]
        embeddings_float32 = embeddings.astype('float32')
        
        # Use IndexFlatIP for exact cosine similarity (inner product)
        # Embeddings should already be normalized
        logger.info(f"Building FAISS index (dimension: {dim})...")
        
        if self.use_gpu:
            try:
                # GPU index
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(dim)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("FAISS index created on GPU")
            except Exception as e:
                logger.warning(f"Failed to create GPU index: {e}. Falling back to CPU.")
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.use_gpu = False
        else:
            # CPU index
            self.faiss_index = faiss.IndexFlatIP(dim)
            logger.info("FAISS index created on CPU")
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_float32)
        logger.info(f"Added {embeddings.shape[0]} vectors to FAISS index")
    
    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self.embeddings is None:
            raise ValueError("No index to save. Build the index first.")
        
        # Save embeddings as compressed numpy file (always save for compatibility)
        embeddings_path = self.index_dir / INDEX_FILENAME
        np.savez_compressed(embeddings_path, embeddings=self.embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save FAISS index if using FAISS backend
        if self.backend == 'faiss' and self.faiss_index is not None:
            faiss_path = self.index_dir / FAISS_INDEX_FILENAME
            
            # Convert GPU index to CPU before saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
                faiss.write_index(cpu_index, str(faiss_path))
            else:
                faiss.write_index(self.faiss_index, str(faiss_path))
            
            logger.info(f"Saved FAISS index to {faiss_path}")
        
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
            
            # Update backend from metadata if available
            if 'backend' in self.metadata:
                saved_backend = self.metadata['backend']
                if saved_backend != self.backend:
                    logger.warning(f"Index was built with {saved_backend} backend, but {self.backend} backend requested")
                    logger.warning(f"Switching to {saved_backend} backend to match saved index")
                    self.backend = saved_backend
            
            logger.info(f"Loaded metadata for {self.metadata['num_documents']} documents")
            
            # Load FAISS index if using FAISS backend
            if self.backend == 'faiss':
                faiss_path = self.index_dir / FAISS_INDEX_FILENAME
                
                if faiss_path.exists() and FAISS_AVAILABLE:
                    logger.info("Loading FAISS index...")
                    cpu_index = faiss.read_index(str(faiss_path))
                    
                    # Move to GPU if requested
                    if self.use_gpu and faiss.get_num_gpus() > 0:
                        try:
                            res = faiss.StandardGpuResources()
                            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                            logger.info("FAISS index loaded on GPU")
                        except Exception as e:
                            logger.warning(f"Failed to load index on GPU: {e}. Using CPU.")
                            self.faiss_index = cpu_index
                            self.use_gpu = False
                    else:
                        self.faiss_index = cpu_index
                        logger.info("FAISS index loaded on CPU")
                else:
                    logger.warning("FAISS index file not found. Rebuilding from embeddings...")
                    self._build_faiss_index(self.embeddings)
            
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
        
        # Use appropriate backend
        if self.backend == 'faiss' and self.faiss_index is not None:
            return self._search_faiss(query_embedding, top_k)
        else:
            return self._search_numpy(query_embedding, top_k)
    
    def _search_numpy(self, query_embedding: np.ndarray, top_k: int) -> tuple:
        """
        NumPy-based exact search.
        
        Args:
            query_embedding: Query embedding (1, dim)
            top_k: Number of results
        
        Returns:
            Tuple of (indices, scores)
        """
        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> tuple:
        """
        FAISS-based fast search.
        
        Args:
            query_embedding: Query embedding (1, dim)
            top_k: Number of results
        
        Returns:
            Tuple of (indices, scores)
        """
        if not FAISS_AVAILABLE or self.faiss_index is None:
            logger.warning("FAISS not available, falling back to NumPy")
            return self._search_numpy(query_embedding, top_k)
        
        # Convert to float32 for FAISS
        query_float32 = query_embedding.astype('float32')
        
        # Search
        top_k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_float32, top_k)
        
        # Return as 1D arrays
        return indices[0], scores[0]
    
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
