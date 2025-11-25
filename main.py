"""
Main application for the Multilingual Information Retrieval System.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import LANGUAGES, CORPUS_SAMPLE_SIZE, DEFAULT_TOP_K
from data_loader import DataLoader
from embedder import MultilingualEmbedder
from indexer import VectorIndex
from retriever import CrossLingualRetriever
from evaluator import IREvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index(sample_size: int = None, force_rebuild: bool = False):
    """
    Build the multilingual vector index.
    
    Args:
        sample_size: Number of documents to sample (None for full corpus)
        force_rebuild: If True, rebuild even if index exists
    """
    logger.info("Starting index building process...")
    
    # Initialize components
    index = VectorIndex()
    
    # Check if index already exists
    if index.index_exists() and not force_rebuild:
        logger.info("Index already exists. Use --force-rebuild to rebuild.")
        return
    
    # Load data
    logger.info("Loading corpus data...")
    data_loader = DataLoader(languages=list(LANGUAGES.keys()))
    corpus_docs = data_loader.load_corpus(sample_size=sample_size)
    
    corpus_texts = data_loader.get_corpus_texts()
    corpus_ids = data_loader.get_corpus_ids()
    corpus_languages = data_loader.get_corpus_languages()
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedder = MultilingualEmbedder()
    embeddings = embedder.encode_corpus(corpus_texts)
    
    # Build and save index (with document texts)
    logger.info("Building index...")
    index.build(embeddings, corpus_ids, corpus_languages, corpus_texts)
    index.save()
    
    logger.info("✅ Index building complete!")


def search(query: str, top_k: int = DEFAULT_TOP_K, show_text: bool = True):
    """
    Search the index with a query.
    
    Args:
        query: Query text
        top_k: Number of results to return
        show_text: If True, display document text
    """
    logger.info("Starting search...")
    
    # Load index
    index = VectorIndex()
    if not index.load():
        logger.error("No index found. Please build the index first using: python main.py build")
        return
    
    # Initialize embedder and retriever
    embedder = MultilingualEmbedder()
    retriever = CrossLingualRetriever(embedder, index)
    
    # Document texts are now stored in the index, no need to reload corpus
    
    # Perform retrieval
    results = retriever.retrieve(query, top_k=top_k, return_full_text=True)
    
    # Display results
    retriever.print_results(results, max_text_length=300)


def interactive_search(top_k: int = DEFAULT_TOP_K):
    """
    Interactive search mode.
    
    Args:
        top_k: Number of results to return per query
    """
    logger.info("Loading index...")
    
    # Load index
    index = VectorIndex()
    if not index.load():
        logger.error("No index found. Please build the index first using: python main.py build")
        return
    
    # Initialize components
    embedder = MultilingualEmbedder()
    retriever = CrossLingualRetriever(embedder, index)
    
    # Document texts are now stored in the index, no need to reload corpus
    
    print("\n" + "="*80)
    print("Multilingual Information Retrieval System")
    print("="*80)
    print(f"Indexed {index.metadata['num_documents']} documents in {len(LANGUAGES)} languages")
    print(f"Languages: {', '.join(LANGUAGES.keys())}")
    print("\nEnter your queries (in English or any supported language)")
    print("Commands: 'quit' or 'exit' to quit, 'help' for help")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nHelp:")
                print("  - Enter any query in English or supported Indian languages")
                print("  - The system will retrieve relevant documents across all languages")
                print("  - Type 'quit' or 'exit' to quit")
                continue
            
            if not query:
                continue
            
            # Retrieve and display results with text snippets
            results = retriever.retrieve(query, top_k=top_k, return_full_text=True)
            
            print(f"\n📄 Top {len(results)} Results:")
            print("=" * 80)
            for result in results:
                print(f"\n[{result['rank']}] {result['language']} | Score: {result['score']:.4f}")
                print(f"Doc ID: {result['doc_id']}")
                if 'text' in result:
                    # Show first 250 characters
                    text_snippet = result['text'][:250].replace('\n', ' ').strip()
                    print(f"Text: {text_snippet}...")
                print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")


def evaluate(languages: list = None, split: str = 'dev', max_queries: int = None):
    """
    Evaluate the retrieval system using nDCG@10 and Recall@100 metrics.
    
    Args:
        languages: List of languages to evaluate (default: all)
        split: Dataset split to use ('dev' or 'train')
        max_queries: Maximum number of queries per language (None for all)
    """
    logger.info("Starting evaluation...")
    
    # Load index
    index = VectorIndex()
    if not index.load():
        logger.error("No index found. Please build the index first using: python main.py build")
        return
    
    # Initialize components
    embedder = MultilingualEmbedder()
    retriever = CrossLingualRetriever(embedder, index)
    
    # Initialize evaluator
    eval_languages = languages or list(LANGUAGES.keys())
    evaluator = IREvaluator(retriever, languages=eval_languages)
    
    # Run evaluation
    logger.info(f"Evaluating on {split} split for languages: {', '.join(eval_languages)}")
    if max_queries:
        logger.info(f"Limiting to {max_queries} queries per language")
    
    results = evaluator.evaluate_all_languages(
        split=split,
        ndcg_k=10,
        recall_k=100,
        max_queries=max_queries
    )
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    logger.info("✅ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Multilingual Information Retrieval System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index with full corpus
  python main.py build

  # Build index with sample of 5000 documents (for testing)
  python main.py build --sample-size 5000

  # Search with a specific query
  python main.py search "What are the common foods in South India?"

  # Interactive search mode
  python main.py interactive

  # Search and show document text
  python main.py search "climate change effects" --show-text --top-k 5

  # Evaluate retrieval performance (nDCG@10 and Recall@100)
  python main.py evaluate

  # Evaluate specific languages only
  python main.py evaluate --languages hindi bengali

  # Evaluate with limited queries (for quick testing)
  python main.py evaluate --max-queries 50
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build the vector index')
    build_parser.add_argument('--sample-size', type=int, default=CORPUS_SAMPLE_SIZE,
                             help='Number of documents to sample (None for full corpus)')
    build_parser.add_argument('--force-rebuild', action='store_true',
                             help='Force rebuild even if index exists')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search with a query')
    search_parser.add_argument('query', type=str, help='Query text')
    search_parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                              help='Number of results to return')
    search_parser.add_argument('--show-text', action='store_true',
                              help='Show document text in results')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive search mode')
    interactive_parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                                   help='Number of results to return per query')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate retrieval performance')
    evaluate_parser.add_argument('--languages', nargs='+', choices=list(LANGUAGES.keys()),
                                help='Languages to evaluate (default: all)')
    evaluate_parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train'],
                                help='Dataset split to use for evaluation')
    evaluate_parser.add_argument('--max-queries', type=int, default=None,
                                help='Maximum number of queries per language (default: all)')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_index(sample_size=args.sample_size, force_rebuild=args.force_rebuild)
    elif args.command == 'search':
        search(args.query, top_k=args.top_k, show_text=args.show_text)
    elif args.command == 'interactive':
        interactive_search(top_k=args.top_k)
    elif args.command == 'evaluate':
        evaluate(languages=args.languages, split=args.split, max_queries=args.max_queries)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
