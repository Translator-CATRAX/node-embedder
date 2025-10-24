# KG2 Node Embedding and Analysis Pipeline

This repository contains two coordinated scripts for embedding and analyzing free-text descriptions of KG2 concept nodes using **BioLinkBERT** ([model documentation](https://huggingface.co/michiyasunaga/BioLinkBERT-base)) and **ChromaDB**.  
The goal is to generate a reusable, persistent biomedical vector store that supports concept canonicalization, cross-reference discovery, and downstream reasoning or visualization.

---

## Overview

| File | Purpose | Typical Usage |
|------|----------|----------------|
| `embed_kg2nodes.py` | Builds embeddings from KG2 node descriptions and stores them in a persistent ChromaDB collection (locally). | Run once per KG version to generate a new vector store. |
| `analyze_embeddings.py` | Connects to an existing vector store for validation, inspection, and visualization of the embeddings. | Run after embeddings are generated to explore semantic structure and relationships. |

---

## Inputs

### For `embed_kg2nodes.py`

**Inputs:**
- TSV file containing `CURIE`, `Name`, and `Description` fields. Must be **tab-delimited** for the script to work.
- Output directory.
- Collection name to create, update, or overwrite.
- Mode indicating whether you are creating, updating, or overwriting a collection.

**Processing:**
- Uses `BioLinkBERT-base` to generate normalized embeddings.
- Falls back to `Name` or `CURIE` when `Description` is missing.
- Metadata (CURIE + name) is preserved for traceability.

**Vector Store Configuration:**
```
"hnsw": {
    "space": "cosine",
    "ef_construction": 200,
    "ef_search": 150,
    "max_neighbors": 32
}
```

**Modes:**
- `new` — create a new collection (default)
- `add` — append to an existing collection
- `overwrite` — replace an existing collection

**Output:**
- Embeddings stored in a Chroma collection named after the current KG version (e.g., `kg2103`).
- The vector store is created as a `PersistentClient` and saved locally in the directory specified by `output_dir`.

---

### For `analyze_embeddings.py`

**Inputs:**
- `-c` (Collection name): name of the Chroma collection to analyze.
- `-d` (Directory): path to the ChromaDB persistent store.
- `-m` (Mode): specifies the type of analysis to run.

**Available Modes:**
- `info` — Summarizes collection statistics and sample embeddings.
- `similar` — Finds semantically similar concepts to the input query using existing embeddings.  
  Example: find concepts semantically similar to `'lactulose'`.  
  Uses the embedding already stored for `'lactulose'` rather than embedding the query text.
- `pair` — Computes cosine similarity between two named concepts.
- `clusters` — Runs **PCA → KMeans** to identify semantic clusters, outputs representative concepts, and saves cluster visualizations.
- `umap` — Performs **UMAP** (or t-SNE if UMAP is not installed) dimensionality reduction and generates labeled 2D visualizations of embedding space.

---

## Example Usage

### Generate embeddings
```
python embed_kg2nodes.py -i nodes_cleaned.tsv -c kg2103 -o chromadb -m new -v
```

### Inspect or visualize embeddings
```
python analyze_embeddings.py -c kg2103 -d chromadb -m info
python analyze_embeddings.py -c kg2103 -d chromadb -m similar -q "lactulose"
python analyze_embeddings.py -c kg2103 -d chromadb -m umap
```

---

## Implementation Details

- **Model:** [BioLinkBERT-base](https://huggingface.co/michiyasunaga/BioLinkBERT-base)  
- **Embedding Normalization:** All embeddings are L2-normalized to ensure consistent cosine similarity comparisons.  
- **Backend:** Persistent ChromaDB store using the HNSW index with cosine distance.  
- **Dependencies:**  
  `chromadb`, `torch`, `sentence-transformers`, `scikit-learn`, `numpy`, `matplotlib`, `umap-learn` (optional).

