"""
Data loader module for loading MIRACL datasets.
"""

import ir_datasets
import logging
from typing import List, Dict, Optional
import random
from config import LANGUAGES, CORPUS_SAMPLE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of multilingual corpus data."""
    
    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize the DataLoader.
        
        Args:
            languages: List of languages to load. If None, loads all supported languages.
        """
        self.languages = languages or list(LANGUAGES.keys())
        self.corpus_docs = []
        
    def load_corpus(self, sample_size: Optional[int] = CORPUS_SAMPLE_SIZE) -> List[Dict[str, str]]:
        """
        Load corpus documents from all specified languages.
        
        Args:
            sample_size: If specified, randomly sample this many documents from the full corpus.
                        If None, load all documents.
        
        Returns:
            List of document dictionaries with 'doc_id', 'text', and 'language' fields.
        """
        all_docs = []
        
        for lang in self.languages:
            logger.info(f"Loading {lang} corpus...")
            dataset_name = LANGUAGES[lang]
            
            try:
                dataset = ir_datasets.load(dataset_name)
                lang_docs = []
                
                for doc in dataset.docs_iter():
                    lang_docs.append({
                        'doc_id': doc.doc_id,
                        'text': doc.text,
                        'language': lang
                    })
                
                logger.info(f"Loaded {len(lang_docs)} {lang} documents")
                all_docs.extend(lang_docs)
                
            except Exception as e:
                logger.error(f"Error loading {lang} corpus: {e}")
                raise
        
        # Sample if requested
        if sample_size is not None and sample_size < len(all_docs):
            logger.info(f"Sampling {sample_size} documents from {len(all_docs)} total documents")
            all_docs = random.sample(all_docs, sample_size)
            random.shuffle(all_docs)  # Shuffle to mix languages
        
        self.corpus_docs = all_docs
        logger.info(f"Total corpus size: {len(all_docs)} documents")
        
        return all_docs
    
    def load_queries(self, language: str = 'en', split: str = 'dev') -> List[Dict[str, str]]:
        """
        Load queries from MIRACL dataset.
        
        Args:
            language: Language code for queries (default: 'en' for English)
            split: Dataset split ('dev' or 'train')
        
        Returns:
            List of query dictionaries with 'query_id' and 'text' fields.
        """
        dataset_name = f"miracl/{language}/{split}"
        
        try:
            logger.info(f"Loading {language} queries from {split} split...")
            dataset = ir_datasets.load(dataset_name)
            queries = []
            
            for query in dataset.queries_iter():
                queries.append({
                    'query_id': query.query_id,
                    'text': query.text
                })
            
            logger.info(f"Loaded {len(queries)} queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            raise
    
    def get_corpus_texts(self) -> List[str]:
        """Get list of corpus document texts."""
        return [doc['text'] for doc in self.corpus_docs]
    
    def get_corpus_ids(self) -> List[str]:
        """Get list of corpus document IDs."""
        return [doc['doc_id'] for doc in self.corpus_docs]
    
    def get_corpus_languages(self) -> List[str]:
        """Get list of corpus document languages."""
        return [doc['language'] for doc in self.corpus_docs]
