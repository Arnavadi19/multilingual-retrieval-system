"""
Example script demonstrating the Multilingual IR System usage.
"""

from data_loader import DataLoader
from embedder import MultilingualEmbedder
from indexer import VectorIndex
from retriever import CrossLingualRetriever
from config import LANGUAGES

def demo_build_small_index():
    """Build a small demo index with 1000 documents."""
    print("="*80)
    print("DEMO: Building Small Index")
    print("="*80)
    
    # Load a small sample
    print("\n1. Loading data...")
    loader = DataLoader(languages=list(LANGUAGES.keys()))
    corpus_docs = loader.load_corpus(sample_size=1000)
    
    # Generate embeddings
    print("\n2. Generating embeddings...")
    embedder = MultilingualEmbedder()
    embeddings = embedder.encode_corpus(loader.get_corpus_texts())
    
    # Build index
    print("\n3. Building index...")
    index = VectorIndex()
    index.build(embeddings, loader.get_corpus_ids(), loader.get_corpus_languages(), 
                loader.get_corpus_texts())
    
    # Save index
    print("\n4. Saving index...")
    index.save()
    
    print("\n✅ Demo index built successfully!")
    print(f"   Documents indexed: {index.metadata['num_documents']}")
    print(f"   Embedding dimension: {index.metadata['embedding_dim']}")
    

def demo_search():
    """Demonstrate searching the index."""
    print("\n" + "="*80)
    print("DEMO: Searching the Index")
    print("="*80)
    
    # Load index
    print("\n1. Loading index...")
    index = VectorIndex()
    if not index.load():
        print("❌ No index found. Please run demo_build_small_index() first.")
        return
    
    # Initialize retriever
    print("\n2. Initializing retriever...")
    embedder = MultilingualEmbedder()
    retriever = CrossLingualRetriever(embedder, index)
    
    # Example queries
    queries = [
        "What are the common foods in South India?",
        "Indian festivals and celebrations",
        "Effects of climate change"
    ]
    
    print("\n3. Running queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n{'-'*80}")
        print(f"Query {i}: {query}")
        print('-'*80)
        
        results = retriever.retrieve(query, top_k=3)
        
        for result in results:
            print(f"  [{result['rank']}] {result['language']} | Score: {result['score']:.4f}")
            print(f"      Doc ID: {result['doc_id']}")
        

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        demo_build_small_index()
    elif len(sys.argv) > 1 and sys.argv[1] == 'search':
        demo_search()
    else:
        print("Usage:")
        print("  python example.py build   # Build demo index")
        print("  python example.py search  # Search demo index")
