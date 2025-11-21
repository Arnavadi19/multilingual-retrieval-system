# Multilingual Information Retrieval System

A professional-grade cross-lingual information retrieval (CLIR) system that enables English queries to retrieve relevant documents from Hindi, Bengali, and Telugu corpora using multilingual dense embeddings.

## Features

- **Cross-Lingual Retrieval**: Query in English, retrieve documents in Hindi, Bengali, and Telugu
- **Dense Vector Embeddings**: Uses state-of-the-art sentence-transformers for semantic similarity
- **Scalable Architecture**: Modular design with separate components for data loading, embedding, indexing, and retrieval
- **Persistent Index**: Save and load pre-computed embeddings for fast retrieval
- **Interactive & CLI Modes**: Both command-line and interactive search interfaces
- **GPU Support**: Automatic GPU acceleration when available

## Architecture

The system is built with a modular architecture:

```
├── config.py          # Configuration and constants
├── data_loader.py     # MIRACL dataset loading and preprocessing
├── embedder.py        # Multilingual embedding generation
├── indexer.py         # Vector index creation and management
├── retriever.py       # Query processing and document retrieval
└── main.py            # CLI application and entry point
```

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster processing)
- At least 8GB RAM (16GB+ recommended for full corpus)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ir-project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Build the Index

First, you need to build the vector index from the MIRACL datasets:

```bash
# Build index with full corpus (this may take time and memory)
python main.py build

# OR build with a sample for testing (recommended for first run)
python main.py build --sample-size 5000

# Force rebuild if index already exists
python main.py build --force-rebuild
```

The index will be saved to the `index/` directory for reuse.

### 2. Search with Queries

#### Single Query Search:
```bash
# Basic search
python main.py search "What are the common foods in South India?"

# Show document text in results
python main.py search "climate change effects" --show-text --top-k 5
```

#### Interactive Mode:
```bash
python main.py interactive

# Then enter queries interactively:
# 🔍 Query: What are the major festivals in India?
```

### 3. Example Queries

Try these example queries to test the system:

- "What are the common foods eaten in South India and Sri Lanka?"
- "Indian independence movement and freedom fighters"
- "Climate change effects on agriculture"
- "Traditional dance forms in India"
- "Renewable energy sources and solar power"

## Project Structure

```
ir-project/
├── config.py              # System configuration
├── data_loader.py         # Dataset loading
├── embedder.py            # Embedding generation
├── indexer.py             # Index management
├── retriever.py           # Search functionality
├── main.py                # CLI application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── APPROACH.md            # Technical approach documentation
├── .gitignore             # Git ignore rules
├── index/                 # Generated index files (created automatically)
│   ├── multilingual_index.npz
│   └── document_metadata.json
├── cache/                 # Cache directory (created automatically)
└── data/                  # Data directory (created automatically)
```

## Configuration

Edit `config.py` to customize:

- **Model**: Change `MODEL_NAME` to use different sentence-transformers models
- **Languages**: Add or remove languages in the `LANGUAGES` dictionary
- **Sample Size**: Set `CORPUS_SAMPLE_SIZE` to limit corpus size for testing
- **Retrieval Parameters**: Adjust `DEFAULT_TOP_K` for number of results

## How It Works

1. **Data Loading**: The system loads documents from the MIRACL dataset for Hindi, Bengali, and Telugu
2. **Embedding Generation**: Each document is encoded into a dense vector using a multilingual sentence-transformer model
3. **Indexing**: Document embeddings are stored in a searchable index with metadata
4. **Query Processing**: User queries are encoded into the same vector space
5. **Retrieval**: Cosine similarity is used to find the most relevant documents
6. **Ranking**: Results are ranked by similarity score and returned

## Performance Tips

- **Use GPU**: The system automatically uses CUDA if available (much faster)
- **Sample Mode**: For testing, use `--sample-size` to work with a subset
- **Batch Size**: Adjust `BATCH_SIZE` in `config.py` based on your GPU memory
- **Normalized Embeddings**: Embeddings are normalized for efficient cosine similarity

## Evaluation

The system uses the MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) dataset, which provides:
- Large-scale multilingual corpora
- Development and test queries with relevance judgments
- Standard benchmarks for cross-lingual IR evaluation

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is intended for educational and research purposes.

## Acknowledgments

- **MIRACL Dataset**: [https://project-miracl.github.io/](https://project-miracl.github.io/)
- **Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **IR Datasets**: [https://ir-datasets.com/](https://ir-datasets.com/)

## Support

For questions or issues, please open an issue on GitHub.

---

