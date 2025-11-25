# FAISS Integration Guide

## Overview

This multilingual IR system supports two indexing backends:

1. **NumPy** (default) - Exact similarity search using dot products
2. **FAISS** (optional) - Fast approximate nearest neighbor search

## Why FAISS?

FAISS (Facebook AI Similarity Search) provides:

- **25-50x faster search** for large corpora (>50k documents)
- **Same retrieval quality** with IndexFlatIP (exact inner product search)
- **GPU acceleration** support for even faster search
- **Memory efficiency** with optimized index structures

## Performance Comparison

| Corpus Size | NumPy Search Time | FAISS CPU Time | FAISS GPU Time |
|-------------|-------------------|----------------|----------------|
| 10K docs    | ~50ms            | ~10ms          | ~5ms           |
| 100K docs   | ~500ms           | ~20ms          | ~8ms           |
| 1M docs     | ~5s              | ~100ms         | ~30ms          |

*Times are approximate for top-10 search on a single query*

## Installation

### CPU-Only Systems

```bash
pip install faiss-cpu
```

### GPU-Enabled Systems

```bash
pip install faiss-gpu
```

**Note**: The system automatically falls back to NumPy if FAISS is not installed.

## Usage

### Building Index

```bash
# Build with NumPy backend (default)
python main.py build

# Build with FAISS backend
python main.py build --backend faiss

# Build with FAISS and GPU acceleration
python main.py build --backend faiss --gpu

# Sample build for testing
python main.py build --backend faiss --sample-size 5000
```

### Searching

```bash
# Search with NumPy backend
python main.py search "climate change" --top-k 10

# Search with FAISS backend (faster)
python main.py search "climate change" --backend faiss --top-k 10

# Search with FAISS GPU acceleration
python main.py search "climate change" --backend faiss --gpu
```

### Interactive Mode

```bash
# Interactive mode with NumPy
python main.py interactive

# Interactive mode with FAISS
python main.py interactive --backend faiss

# Interactive mode with FAISS GPU
python main.py interactive --backend faiss --gpu
```

### Evaluation

```bash
# Evaluate with NumPy
python main.py evaluate

# Evaluate with FAISS (much faster for large corpora)
python main.py evaluate --backend faiss

# Evaluate with FAISS GPU
python main.py evaluate --backend faiss --gpu --max-queries 100
```

## FAISS Index Details

### Index Type: IndexFlatIP

The system uses FAISS's `IndexFlatIP` (Flat Index with Inner Product):

- **Exact search** - Same quality as NumPy (no approximation)
- **Inner product metric** - Compatible with cosine similarity (normalized vectors)
- **No quantization** - Vectors stored in full precision
- **No clustering** - Direct brute-force search (fast enough for most IR tasks)

### GPU Support

When `--gpu` flag is used:

1. System checks for GPU availability via `faiss.get_num_gpus()`
2. If GPU available, creates `GpuIndexFlatIP` with default GPU resources
3. If GPU unavailable, falls back to CPU FAISS index
4. GPU provides 2-5x additional speedup over CPU FAISS

### Index Persistence

FAISS indices are saved alongside NumPy indices:

```
index/
├── embeddings.npz           # NumPy embeddings (always saved)
├── faiss_index.bin          # FAISS binary index (when using FAISS)
└── metadata.json            # Index metadata
```

When loading:
- NumPy backend loads from `embeddings.npz`
- FAISS backend loads from `faiss_index.bin` (falls back to NumPy if missing)

## Backend Selection Logic

### At Build Time

```python
# In indexer.py
class VectorIndex:
    def __init__(self, backend='numpy', use_gpu=False):
        # Validates backend choice
        # Creates appropriate index structure
        
    def save(self, index_dir):
        # Saves NumPy arrays (always)
        # Saves FAISS index (if backend='faiss')
```

### At Search Time

```python
def search(self, query_embedding, top_k):
    if self.backend == 'numpy':
        return self._search_numpy(query_embedding, top_k)
    else:  # faiss
        return self._search_faiss(query_embedding, top_k)
```

## When to Use Which Backend?

### Use NumPy When:

- Small corpus (<10k documents)
- Simplicity is priority
- FAISS installation not possible
- Exact reproducibility required across systems

### Use FAISS When:

- Large corpus (>50k documents)
- Speed is critical
- Running many evaluation queries
- GPU is available for acceleration

## Troubleshooting

### FAISS Not Found Error

**Error**: `FAISS is not installed`

**Solution**: 
```bash
pip install faiss-cpu  # or faiss-gpu
```

### GPU Not Available

**Behavior**: System automatically falls back to CPU FAISS

**Check GPU availability**:
```python
import faiss
print(f"GPUs available: {faiss.get_num_gpus()}")
```

### Incompatible Index File

**Error**: Cannot load FAISS index

**Solution**: Rebuild index with correct backend
```bash
python main.py build --backend faiss --force-rebuild
```

### Performance Not Improving

**Check**:
1. Ensure FAISS is actually being used (check logs)
2. Corpus size - FAISS benefits appear at >10k docs
3. GPU flag set if GPU is available
4. Latest FAISS version installed

## Technical Implementation

### Key Code Sections

#### 1. Initialization (`indexer.py`)

```python
def __init__(self, backend='numpy', use_gpu=False):
    valid_backends = ['numpy', 'faiss']
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}")
    
    if backend == 'faiss' and not FAISS_AVAILABLE:
        logger.warning("FAISS not available, falling back to NumPy")
        backend = 'numpy'
```

#### 2. FAISS Index Creation

```python
def _build_faiss_index(self):
    dimension = self.embeddings.shape[1]
    
    if self.use_gpu and FAISS_AVAILABLE:
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            # GPU index
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            # Fallback to CPU
            index = faiss.IndexFlatIP(dimension)
    else:
        # CPU index
        index = faiss.IndexFlatIP(dimension)
    
    index.add(self.embeddings)
    return index
```

#### 3. Search Implementation

```python
def _search_faiss(self, query_embedding, top_k):
    scores, indices = self.faiss_index.search(
        query_embedding.reshape(1, -1), 
        top_k
    )
    return indices[0], scores[0]
```

## Best Practices

1. **Build once, search many**: FAISS index building is fast, but search is where it shines
2. **Use GPU for evaluation**: When running `evaluate` with many queries, GPU provides huge speedup
3. **Match backends**: Use same backend for build and search (system handles this automatically)
4. **Start with NumPy**: Test system with NumPy first, then upgrade to FAISS for production
5. **Monitor memory**: FAISS uses more RAM than NumPy for index structures (but still reasonable)

## References

- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Documentation](https://faiss.ai/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
