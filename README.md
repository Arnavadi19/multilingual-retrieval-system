Multilingual Information Retrieval SystemA professional-grade cross-lingual information retrieval (CLIR) system that enables English queries to retrieve relevant documents from Hindi, Bengali and Telugu corpora using multilingual dense embeddings.FeaturesCross-Lingual Retrieval: Queries are in English and are retrieved documents in Hindi, Bengali and TeluguDense Vector Embeddings: Uses state-of-the-art sentence-transformers for the semantic similarityScalable Architecture: Modular design with separate components for data loading, embedding, indexing and retrievalPersistent Index: Save and load pre-computed embeddings for fast retrievalInteractive & CLI Modes: Both command-line and interactive search interfacesWeb API: Exposes retrieval functionality via a FastAPI endpoint for use by the frontend.GPU Support: Automatic GPU acceleration when availableArchitectureThe system is built with a modular architecture:├── config.py          # Configuration and constants
├── data_loader.py     # MIRACL dataset loading and preprocessing
├── embedder.py        # Multilingual embedding generation
├── indexer.py         # Vector index creation and management
├── retriever.py       # Query processing and document retrieval
├── main.py            # CLI application and entry point
└── main_api.py        # FastAPI server entry point
PrerequisitesPython 3.8 or higherNode.js and npm/yarn (for frontend)CUDA-compatible GPU (optional, but recommended for faster processing)At least 8GB RAM (16GB+ recommended for full corpus)InstallationClone the repository:   bash    git clone <your-repo-url>    cd ir-project    Create a virtual environment (recommended):   bash    python -m venv venv    source venv/bin/activate  # On Windows: venv\Scripts\activate    Install Python dependencies:   bash    pip install -r requirements.txt    💻 Usage: Backend (Python)1. Build the IndexFirst, you need to build the vector index from the MIRACL datasets. This must be done once before the API can run.# Build index with full corpus (this may take time and memory)
python main.py build

# OR build with a sample for testing (recommended for first run)
python main.py build --sample-size 5000

# Force rebuild if index already exists
python main.py build --force-rebuild
The index will be saved to the index/ directory for reuse.2. Start the Web API (Required for Frontend)The FastAPI server exposes the retrieval function to the React frontend on port 8000.# Start the uvicorn server with autoreload
python -m uvicorn main_api:app --reload --port 8000
Keep this terminal window running while using the frontend.3. Search with Queries (CLI)Single Query Search:# Basic search
python main.py search "What are the common foods in South India?"

# Show document text in results
python main.py search "climate change effects" --show-text --top-k 5
Interactive Mode:python main.py interactive

# Then enter queries interactively:
# 🔍 Query: What are the major festivals in India?
🖥️ Usage: Frontend (React UI)The frontend is a separate application that relies on the Web API (step 2 above) to be running.1. InstallationNavigate into the frontend project directory:cd multilingual-retrieval-ui
Install the necessary Node.js dependencies:npm install
# or
yarn install
2. Run the UIStart the development server:npm run dev
# or
yarn dev
The application will typically open at http://localhost:5173. You can now enter English queries and view the multilingual results in the browser.📚 Project Structureir-project/
├── config.py              # System configuration
├── data_loader.py         # Dataset loading
├── embedder.py            # Embedding generation
├── indexer.py             # Index management
├── retriever.py           # Search functionality
├── main.py                # CLI application
├── main_api.py            # FastAPI server
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── ... (other root files)
├── index/                 # Generated index files
├── cache/                 # Cache directory
├── data/                  # Data directory
└── multilingual-retrieval-ui/ # Frontend Directory
    ├── src/
    └── package.json
Example QueriesTry these example queries to test the system:"What are the common foods eaten in South India and Sri Lanka?""Indian independence movement and freedom fighters""Climate change effects on agriculture""Traditional dance forms in India""Renewable energy sources and solar power"ConfigurationEdit config.py to customize:Model: Change MODEL_NAME to use different sentence-transformers modelsLanguages: Add or remove languages in the LANGUAGES dictionarySample Size: Set CORPUS_SAMPLE_SIZE to limit corpus size for testingRetrieval Parameters: Adjust DEFAULT_TOP_K for number of resultsHow It WorksData Loading: The system loads documents from the MIRACL dataset for Hindi, Bengali, and TeluguEmbedding Generation: Each document is encoded into a dense vector using a multilingual sentence-transformer modelIndexing: Document embeddings are stored in a searchable index with metadataQuery Processing: User queries are encoded into the same vector spaceRetrieval: Cosine similarity is used to find the most relevant documentsRanking: Results are ranked by similarity score and returnedPerformance TipsUse GPU: The system automatically uses CUDA if available (much faster)Sample Mode: For testing, use --sample-size to work with a subsetBatch Size: Adjust BATCH_SIZE in config.py based on your GPU memoryNormalized Embeddings: Embeddings are normalized for efficient cosine similarityEvaluationThe system uses the MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) dataset that provides:Large-scale multilingual corporaDevelopment and test queries with relevance judgmentsStandard benchmarks for cross-lingual IR evaluationContributingContributions are welcome! Please feel free to submit issues or pull requests.LicenseThis project is intended for educational and research purposes.AcknowledgmentsMIRACL Dataset: https://project-miracl.github.io/Sentence Transformers: https://www.sbert.net/IR Datasets: https://ir-datasets.com/FastAPI: For the lightweight web API.React/Vite: For the modern frontend user interface.SupportFor questions or issues, please open an issue on GitHub.
