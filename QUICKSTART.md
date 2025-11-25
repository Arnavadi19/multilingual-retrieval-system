# Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Build the Index
```bash
# Build with a small sample (recommended for first run)
python main.py build --sample-size 5000

# This will:
# - Download MIRACL datasets (cached locally)
# - Load Hindi, Bengali, and Telugu documents
# - Generate multilingual embeddings
# - Save the index to index/ directory
```

### Step 3: Search!
```bash
# Single query search
python main.py search "What are the common foods in South India?"

# Interactive mode (recommended)
python main.py interactive
```

## Example Session

```bash
$ python main.py interactive

================================================================================
Multilingual Information Retrieval System
================================================================================
Indexed 5000 documents in 3 languages
Languages: hindi, bengali, telugu

Enter your queries (in English or any supported language)
Commands: 'quit' or 'exit' to quit, 'help' for help
================================================================================

🔍 Query: What are the common foods in South India?

📄 Top 10 Results:
--------------------------------------------------------------------------------
  [1] Telugu | Score: 0.7234
      Doc ID: te#12345...
  [2] Tamil | Score: 0.6891
      Doc ID: hi#67890...
  [3] Bengali | Score: 0.6543
      Doc ID: bn#11111...
...
```

## Example Queries to Try

1. **Food & Culture**
   - "What are the common foods in South India?"
   - "Traditional Indian festivals and celebrations"
   - "Indian wedding customs and traditions"

2. **History & Politics**
   - "Indian independence movement leaders"
   - "Ancient Indian civilizations"
   - "Indian constitution and democracy"

3. **Science & Environment**
   - "Effects of climate change on agriculture"
   - "Renewable energy sources in India"
   - "Water conservation methods"

4. **Technology**
   - "Digital India initiatives"
   - "Space research in India"
   - "Mobile technology adoption"

## ⚙️ Configuration Options

### Using Full Corpus (Production)
```bash
# Build with all documents (may take time and memory)
python main.py build
```

### Adjusting Results Count
```bash
# Get more results
python main.py search "your query" --top-k 20

# In interactive mode
python main.py interactive --top-k 15
```

### Showing Document Text
```bash
# Display document snippets in results
python main.py search "your query" --show-text
```

## 🔧 Troubleshooting

### Out of Memory?
- Use a smaller `--sample-size` when building
- Close other applications
- Edit `config.py` and set `BATCH_SIZE` to a smaller value (e.g., 16)

### Slow Performance?
- Check if GPU is being used (look for "device: cuda" in logs)
- If on CPU, use a smaller corpus sample
- Consider using a machine with more RAM/GPU

### Index Not Found?
```bash
# Make sure to build the index first
python main.py build --sample-size 5000
```

## Project Files

- `main.py` - Main application (run this)
- `config.py` - Configuration settings
- `data_loader.py` - Dataset loading
- `embedder.py` - Embedding generation
- `indexer.py` - Index management
- `retriever.py` - Search functionality
- `example.py` - Example usage script
- `index/` - Saved index files (created after build)

## Learn More

- See `README.md` for full documentation
- See `APPROACH.md` for technical details
- Check `example.py` for programmatic usage

## 💡 Tips

1. **First Time**: Use `--sample-size 5000` to test quickly
2. **Production**: Remove sample size limit for full corpus
3. **GPU**: System automatically uses GPU if available
4. **Persistence**: Index is saved and reused (no need to rebuild)
5. **Languages**: System works with any query language, not just English!
6. **Evaluation**: Run `python main.py evaluate` to measure performance with nDCG@10 and Recall@100

## 📊 Evaluating Your System

After building the index, you can evaluate retrieval performance:

```bash
# Evaluate on all languages (Hindi, Bengali, Telugu)
python main.py evaluate

# Quick test with limited queries
python main.py evaluate --max-queries 20

# Evaluate specific languages only
python main.py evaluate --languages hindi bengali
```

See [`EVALUATION.md`](EVALUATION.md ) for detailed evaluation instructions and interpreting results.

---

**Ready to explore multilingual information retrieval! 🌍**
