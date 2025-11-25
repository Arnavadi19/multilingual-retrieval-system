"""
Evaluation module for the Multilingual Information Retrieval System.
Calculates nDCG@10 and Recall@100 metrics for cross-lingual retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import ir_datasets

from embedder import MultilingualEmbedder
from indexer import VectorIndex
from retriever import CrossLingualRetriever
from config import LANGUAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_recall_at_k(retrieved_doc_ids: List[str], 
                          relevant_doc_ids: List[str], 
                          k: int = 100) -> float:
    """
    Calculate Recall@k metric.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ranked)
        relevant_doc_ids: List of ground truth relevant document IDs
        k: Cutoff for recall calculation
    
    Returns:
        Recall@k score (between 0 and 1)
    """
    if len(relevant_doc_ids) == 0:
        return 0.0
    
    retrieved_k = set(retrieved_doc_ids[:k])
    relevant_set = set(relevant_doc_ids)
    
    num_relevant_retrieved = len(retrieved_k & relevant_set)
    recall = num_relevant_retrieved / len(relevant_set)
    
    return recall


def calculate_ndcg_at_k(retrieved_doc_ids: List[str], 
                        relevant_doc_ids: List[str], 
                        k: int = 10) -> float:
    """
    Calculate nDCG@k (Normalized Discounted Cumulative Gain at k).
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ranked)
        relevant_doc_ids: List of ground truth relevant document IDs
        k: Cutoff for nDCG calculation
    
    Returns:
        nDCG@k score (between 0 and 1)
    """
    if len(relevant_doc_ids) == 0:
        return 0.0
    
    relevant_set = set(relevant_doc_ids)
    
    # Calculate DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], 1):
        if doc_id in relevant_set:
            # Binary relevance: 1 if relevant, 0 otherwise
            # DCG formula: sum(rel_i / log2(i + 1))
            dcg += 1.0 / np.log2(i + 1)
    
    # Calculate IDCG@k (Ideal DCG)
    idcg = 0.0
    for i in range(min(k, len(relevant_doc_ids))):
        idcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
    
    # Normalize
    if idcg == 0.0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


class IREvaluator:
    """Evaluates the multilingual IR system using standard metrics."""
    
    def __init__(self, retriever: CrossLingualRetriever, languages: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            retriever: CrossLingualRetriever instance
            languages: List of languages to evaluate (default: all supported)
        """
        self.retriever = retriever
        self.languages = languages or list(LANGUAGES.keys())
    
    def load_evaluation_data(self, language: str, split: str = 'dev') -> Tuple[List[Tuple], Dict]:
        """
        Load evaluation queries and relevance judgments (qrels) for a language.
        
        Args:
            language: Language code (e.g., 'hi', 'bn', 'te')
            split: Dataset split ('dev' or 'train')
        
        Returns:
            Tuple of (queries, qrels) where:
                queries: List of (query_id, query_text) tuples
                qrels: Dict mapping query_id -> list of relevant doc_ids
        """
        dataset_name = LANGUAGES[language]
        dataset_full = f"{dataset_name}/{split}"
        
        logger.info(f"Loading evaluation data for {dataset_full}...")
        
        try:
            dataset = ir_datasets.load(dataset_full)
            
            # Load queries
            queries = []
            for query in dataset.queries_iter():
                queries.append((query.query_id, query.text))
            
            # Load qrels (relevance judgments)
            qrels = defaultdict(list)
            for qrel in dataset.qrels_iter():
                # Only consider documents marked as relevant (relevance > 0)
                if qrel.relevance > 0:
                    qrels[qrel.query_id].append(qrel.doc_id)
            
            logger.info(f"Loaded {len(queries)} queries and {len(qrels)} query-document pairs")
            return queries, dict(qrels)
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            raise
    
    def evaluate_language(self, language: str, 
                         split: str = 'dev',
                         ndcg_k: int = 10, 
                         recall_k: int = 100,
                         max_queries: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate retrieval performance for a specific language.
        
        Args:
            language: Language to evaluate ('hi', 'bn', or 'te')
            split: Dataset split ('dev' or 'train')
            ndcg_k: k value for nDCG@k calculation
            recall_k: k value for Recall@k calculation
            max_queries: Maximum number of queries to evaluate (None for all)
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {language} on {split} split...")
        
        # Load evaluation data
        queries, qrels = self.load_evaluation_data(language, split)
        
        # Limit queries if specified
        if max_queries is not None:
            queries = queries[:max_queries]
            logger.info(f"Limiting evaluation to {max_queries} queries")
        
        # Evaluate each query
        ndcg_scores = []
        recall_scores = []
        queries_evaluated = 0
        
        for query_id, query_text in queries:
            # Skip queries without relevance judgments
            if query_id not in qrels:
                continue
            
            relevant_docs = qrels[query_id]
            
            # Retrieve documents (get up to recall_k documents)
            try:
                results = self.retriever.retrieve(query_text, top_k=recall_k, return_full_text=False)
                retrieved_doc_ids = [result['doc_id'] for result in results]
                
                # Calculate metrics
                ndcg = calculate_ndcg_at_k(retrieved_doc_ids, relevant_docs, k=ndcg_k)
                recall = calculate_recall_at_k(retrieved_doc_ids, relevant_docs, k=recall_k)
                
                ndcg_scores.append(ndcg)
                recall_scores.append(recall)
                queries_evaluated += 1
                
                if queries_evaluated % 10 == 0:
                    logger.info(f"Evaluated {queries_evaluated}/{len(queries)} queries...")
                    
            except Exception as e:
                logger.warning(f"Error evaluating query {query_id}: {e}")
                continue
        
        # Calculate average metrics
        if len(ndcg_scores) == 0:
            logger.warning("No queries were successfully evaluated!")
            return {
                f'nDCG@{ndcg_k}': 0.0,
                f'Recall@{recall_k}': 0.0,
                'num_queries': 0
            }
        
        avg_ndcg = np.mean(ndcg_scores)
        avg_recall = np.mean(recall_scores)
        
        results = {
            f'nDCG@{ndcg_k}': avg_ndcg,
            f'Recall@{recall_k}': avg_recall,
            'num_queries': queries_evaluated
        }
        
        logger.info(f"Evaluation complete for {language}:")
        logger.info(f"  nDCG@{ndcg_k}: {avg_ndcg:.4f}")
        logger.info(f"  Recall@{recall_k}: {avg_recall:.4f}")
        logger.info(f"  Queries evaluated: {queries_evaluated}")
        
        return results
    
    def evaluate_all_languages(self, 
                              split: str = 'dev',
                              ndcg_k: int = 10, 
                              recall_k: int = 100,
                              max_queries: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval performance across all languages.
        
        Args:
            split: Dataset split ('dev' or 'train')
            ndcg_k: k value for nDCG@k calculation
            recall_k: k value for Recall@k calculation
            max_queries: Maximum number of queries per language (None for all)
        
        Returns:
            Dictionary mapping language -> metrics dict
        """
        all_results = {}
        
        for language in self.languages:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {language.upper()}")
            logger.info(f"{'='*80}\n")
            
            try:
                results = self.evaluate_language(
                    language, 
                    split=split,
                    ndcg_k=ndcg_k, 
                    recall_k=recall_k,
                    max_queries=max_queries
                )
                all_results[language] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {language}: {e}")
                all_results[language] = {
                    f'nDCG@{ndcg_k}': 0.0,
                    f'Recall@{recall_k}': 0.0,
                    'num_queries': 0,
                    'error': str(e)
                }
        
        return all_results
    
    def print_evaluation_summary(self, results: Dict[str, Dict[str, float]]):
        """
        Print a formatted summary of evaluation results.
        
        Args:
            results: Dictionary of evaluation results from evaluate_all_languages()
        """
        print("\n" + "="*80)
        print("CROSS-LINGUAL RETRIEVAL EVALUATION RESULTS")
        print("="*80)
        print("\nMetrics: nDCG@10 and Recall@100")
        print("Setting: Cross-lingual (English queries → Hindi/Bengali/Telugu documents)")
        print("-"*80)
        
        # Table header
        print(f"\n{'Language':<15} {'nDCG@10':<15} {'Recall@100':<15} {'Queries':<15}")
        print("-"*60)
        
        # Per-language results
        total_ndcg = []
        total_recall = []
        
        for lang, metrics in results.items():
            lang_name = LANGUAGES.get(lang, lang).split('/')[-1].upper()
            ndcg = metrics.get('nDCG@10', 0.0)
            recall = metrics.get('Recall@100', 0.0)
            num_queries = metrics.get('num_queries', 0)
            
            print(f"{lang_name:<15} {ndcg:<15.4f} {recall:<15.4f} {num_queries:<15}")
            
            if num_queries > 0:
                total_ndcg.append(ndcg)
                total_recall.append(recall)
        
        # Average across languages
        if total_ndcg:
            print("-"*60)
            avg_ndcg = np.mean(total_ndcg)
            avg_recall = np.mean(total_recall)
            print(f"{'AVERAGE':<15} {avg_ndcg:<15.4f} {avg_recall:<15.4f}")
        
        print("="*80 + "\n")
