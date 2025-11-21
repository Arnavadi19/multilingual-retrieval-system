"""
Retriever module for query processing and document retrieval.
"""

import logging
from typing import List, Dict, Optional
from embedder import MultilingualEmbedder
from indexer import VectorIndex
from data_loader import DataLoader
from config import DEFAULT_TOP_K, LANG_CODE_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLingualRetriever:
    """Handles cross-lingual information retrieval."""
    
    def __init__(self, embedder: MultilingualEmbedder, index: VectorIndex):
        """
        Initialize the retriever.
        
        Args:
            embedder: MultilingualEmbedder instance for encoding queries
            index: VectorIndex instance for searching
        """
        self.embedder = embedder
        self.index = index
        self.corpus_texts = None  # Will be loaded if needed
    
    def set_corpus_texts(self, corpus_texts: List[str]) -> None:
        """
        Set corpus texts for retrieving full document content.
        
        Args:
            corpus_texts: List of document texts corresponding to the index
        """
        self.corpus_texts = corpus_texts
    
    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, 
                 return_full_text: bool = False) -> List[Dict]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query text (in any language, typically English)
            top_k: Number of documents to retrieve
            return_full_text: If True, include full document text in results
        
        Returns:
            List of result dictionaries with document information and scores
        """
        # Encode the query
        logger.info(f"Processing query: '{query}'")
        query_embedding = self.embedder.encode_query(query)
        
        # Search the index
        indices, scores = self.index.search(query_embedding, top_k=top_k)
        
        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            doc_info = self.index.get_document_info(idx)
            
            # Extract language name from doc_id or metadata
            lang_code = doc_info['doc_id'].split('#')[0]
            language_name = LANG_CODE_MAP.get(lang_code, doc_info['language'])
            
            result = {
                'rank': rank,
                'doc_id': doc_info['doc_id'],
                'language': language_name,
                'score': float(score)
            }
            
            # Add full text if available and requested
            if return_full_text and self.corpus_texts is not None:
                result['text'] = self.corpus_texts[idx]
            
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def print_results(self, results: List[Dict], max_text_length: int = 200) -> None:
        """
        Pretty print retrieval results.
        
        Args:
            results: List of result dictionaries from retrieve()
            max_text_length: Maximum length of text to display
        """
        print(f"\n{'='*80}")
        print(f"Retrieved {len(results)} documents")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"  Document ID: {result['doc_id']}")
            print(f"  Language: {result['language']}")
            
            if 'text' in result:
                text = result['text'][:max_text_length].replace('\n', ' ')
                print(f"  Content: {text}...")
            
            print(f"{'-'*80}")
    
    def batch_retrieve(self, queries: List[str], top_k: int = DEFAULT_TOP_K) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of documents to retrieve per query
        
        Returns:
            List of result lists, one per query
        """
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            all_results.append(results)
        
        return all_results
