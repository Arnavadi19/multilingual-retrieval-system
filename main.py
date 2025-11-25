"""
Main application for the Multilingual Information Retrieval System.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import LANGUAGES, CORPUS_SAMPLE_SIZE, DEFAULT_TOP_K, DEFAULT_INDEX_BACKEND, USE_GPU_FOR_FAISS
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


def build_index(sample_size: int = None, force_rebuild: bool = False, backend: str = DEFAULT_INDEX_BACKEND, use_gpu: bool = USE_GPU_FOR_FAISS):
    """
    Build the multilingual vector index.
    
    Args:
        sample_size: Number of documents to sample (None for full corpus)
        force_rebuild: If True, rebuild even if index exists
        backend: Indexing backend ('numpy' or 'faiss')
        use_gpu: Use GPU for FAISS (only applicable if backend='faiss')
    """
    logger.info("Starting index building process...")
    logger.info(f"Backend: {backend}, GPU: {use_gpu if backend == 'faiss' else 'N/A'}")
    
    # Initialize components
    index = VectorIndex(backend=backend, use_gpu=use_gpu)
    
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


def search(query: str, top_k: int = DEFAULT_TOP_K, show_text: bool = True, backend: str = DEFAULT_INDEX_BACKEND, use_gpu: bool = USE_GPU_FOR_FAISS):
    """
    Search the index with a query.
    
    Args:
        query: Query text
        top_k: Number of results to return
        show_text: If True, display document text
        backend: Indexing backend ('numpy' or 'faiss')
        use_gpu: Use GPU for FAISS (only applicable if backend='faiss')
    """
    logger.info("Starting search...")
    
    # Load index
    index = VectorIndex(backend=backend, use_gpu=use_gpu)
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


def interactive_search(top_k: int = DEFAULT_TOP_K, backend: str = DEFAULT_INDEX_BACKEND, use_gpu: bool = USE_GPU_FOR_FAISS):
    """
    Interactive search mode.
    
    Args:
        top_k: Number of results to return per query
        backend: Indexing backend ('numpy' or 'faiss')
        use_gpu: Use GPU for FAISS (only applicable if backend='faiss')
    """
    logger.info("Loading index...")
    
    # Load index
    index = VectorIndex(backend=backend, use_gpu=use_gpu)
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
    print(f"Backend: {backend.upper()}" + (f" (GPU)" if use_gpu and backend == 'faiss' else ""))
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


def evaluate(languages: list = None, split: str = 'dev', max_queries: int = None, backend: str = DEFAULT_INDEX_BACKEND, use_gpu: bool = USE_GPU_FOR_FAISS):
    """
    Evaluate the retrieval system using nDCG@10 and Recall@100 metrics.
    
    Args:
        languages: List of languages to evaluate (default: all)
        split: Dataset split to use ('dev' or 'train')
        max_queries: Maximum number of queries per language (None for all)
        backend: Indexing backend ('numpy' or 'faiss')
        use_gpu: Use GPU for FAISS (only applicable if backend='faiss')
    """
    logger.info("Starting evaluation...")
    
    # Load index
    index = VectorIndex(backend=backend, use_gpu=use_gpu)
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
  # Build index with full corpus (NumPy backend)
  python main.py build

  # Build index with FAISS backend (faster)
  python main.py build --backend faiss

  # Build with FAISS and GPU acceleration
  python main.py build --backend faiss --gpu

  # Build index with sample of 5000 documents (for testing)
  python main.py build --sample-size 5000

  # Search with a specific query (NumPy backend)
  python main.py search "What are the common foods in South India?"

  # Search with FAISS backend
  python main.py search "climate change" --backend faiss

  # Interactive search mode (NumPy backend)
  python main.py interactive

  # Interactive mode with FAISS backend
  python main.py interactive --backend faiss

  # Interactive mode with FAISS GPU acceleration
  python main.py interactive --backend faiss --gpu

  # Search and show document text
  python main.py search "climate change effects" --show-text --top-k 5

  # Evaluate retrieval performance (nDCG@10 and Recall@100)
  python main.py evaluate

  # Evaluate with FAISS backend
  python main.py evaluate --backend faiss

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
    build_parser.add_argument('--backend', type=str, default=DEFAULT_INDEX_BACKEND, 
                             choices=['numpy', 'faiss'],
                             help='Indexing backend (numpy for exact, faiss for fast)')
    build_parser.add_argument('--gpu', action='store_true', default=USE_GPU_FOR_FAISS,
                             help='Use GPU for FAISS (if available)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search with a query')
    search_parser.add_argument('query', type=str, help='Query text')
    search_parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                              help='Number of results to return')
    search_parser.add_argument('--show-text', action='store_true',
                              help='Show document text in results')
    search_parser.add_argument('--backend', type=str, default=DEFAULT_INDEX_BACKEND,
                              choices=['numpy', 'faiss'],
                              help='Indexing backend (numpy for exact, faiss for fast)')
    search_parser.add_argument('--gpu', action='store_true', default=USE_GPU_FOR_FAISS,
                              help='Use GPU for FAISS (if available)')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive search mode')
    interactive_parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                                   help='Number of results to return per query')
    interactive_parser.add_argument('--backend', type=str, default=DEFAULT_INDEX_BACKEND,
                                   choices=['numpy', 'faiss'],
                                   help='Indexing backend (numpy for exact, faiss for fast)')
    interactive_parser.add_argument('--gpu', action='store_true', default=USE_GPU_FOR_FAISS,
                                   help='Use GPU for FAISS (if available)')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate retrieval performance')
    evaluate_parser.add_argument('--languages', nargs='+', choices=list(LANGUAGES.keys()),
                                help='Languages to evaluate (default: all)')
    evaluate_parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train'],
                                help='Dataset split to use for evaluation')
    evaluate_parser.add_argument('--max-queries', type=int, default=None,
                                help='Maximum number of queries per language (default: all)')
    evaluate_parser.add_argument('--backend', type=str, default=DEFAULT_INDEX_BACKEND,
                                choices=['numpy', 'faiss'],
                                help='Indexing backend (numpy for exact, faiss for fast)')
    evaluate_parser.add_argument('--gpu', action='store_true', default=USE_GPU_FOR_FAISS,
                                help='Use GPU for FAISS (if available)')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_index(sample_size=args.sample_size, force_rebuild=args.force_rebuild, 
                   backend=args.backend, use_gpu=args.gpu)
    elif args.command == 'search':
        search(args.query, top_k=args.top_k, show_text=args.show_text,
              backend=args.backend, use_gpu=args.gpu)
    elif args.command == 'interactive':
        interactive_search(top_k=args.top_k, backend=args.backend, use_gpu=args.gpu)
    elif args.command == 'evaluate':
        evaluate(languages=args.languages, split=args.split, max_queries=args.max_queries,
                backend=args.backend, use_gpu=args.gpu)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
