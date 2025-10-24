#!/usr/bin/env python3

#####################################################################
#
#  KG2x Node Description Embedding Pipeline
#                                                                   
#  Copyright 2025
#
#  Author: Frankie Hodges
#      Maintained by: Ramsey Lab, Oregon State Unvierstiy 
#
#  College of Engineering
#  Oregon State University
#  Corvallis, OR 97331
#
#  email: hodgesf@oregonstate.edu
# 
#  Extended ChromaDB analysis tool for biomedical embeddings.
#  Supports collection stats, querying, pairwise similarity,
#  KMeans cluster summaries, and UMAP visualization.
# 
#  Usage:
#     python3 analyze_embeddings.py -c kg2103_bert -m info
#         # Show basic stats and sample entries from the collection
#
#     python3 analyze_embeddings.py -c kg2103_bert -m similar -q "lactulose"
#         # Find concepts semantically similar to the concept named 'lactulose'
#         # (Uses the embedding already stored for 'lactulose' rather than embedding the query text)
#
#     python3 analyze_embeddings.py -c kg2103_bert -m pair -a "lactulose" -b "sorbitol"
#         # Compute cosine similarity between the stored embeddings for two concept names
#
#     python3 analyze_embeddings.py -c kg2103_bert -m clusters
#         # Perform KMeans clustering on the stored embeddings and show top terms per cluster
#
#     python3 analyze_embeddings.py -c kg2103_bert -m umap
#         # Generate a UMAP or t-SNE visualization of the embedding space
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#                                                                                                                         
#####################################################################

import sys
import getopt
import chromadb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


#####################################################################
# Helper Functions
#####################################################################
def print_help():
    print("\nUsage:")
    print("  analyze_embeddings.py -c <collection> -d <directory> -m <mode> [options]\n")
    print("\nOptions:")
    print("  -d, --dir <persist_dir>      Directory containing the persistent Chroma store")
    print("Modes:")
    print("  info                         Find concepts semantically similar to the concept named 'lactulose'")
    print("  similar -q <text>            Query for similar embeddings")
    print("  pair -a <textA> -b <textB>   Compute cosine similarity between the stored embeddings for two concept names")
    print("  clusters                     Perform KMeans clustering on the stored embeddings and show top terms per cluster")
    print("  umap                         2D visualization via UMAP (or t-SNE fallback)")
    print("\nExamples:")
    print("  python3 analyze_embeddings.py -c kg2103_bert -d ./chromadb -m info")
    print("  python3 analyze_embeddings.py -c kg2103_bert -d ./chromadb -m similar -q 'lactulose'")
    print("  python3 analyze_embeddings.py -c kg2103_bert -d ./chromadb -m pair -a 'lactulose' -b 'sorbitol'")
    print("  python3 analyze_embeddings.py -c kg2103_bert -d ./chromadb -m clusters")
    print("  python3 analyze_embeddings.py -c kg2103_bert -d ./chromadb -m umap")
    sys.exit()

#####################################################################
# Parse Command-line Arguments
#####################################################################
collection_name = ""
mode = ""
directory = "./chroma_dir"
query_text = None
textA = None
textB = None

opts, args = getopt.getopt(sys.argv[1:], "hc:d:m:q:a:b:", ["collection=", "directory=", "mode=", "query=", "a=", "b="])
for opt, arg in opts:
    if opt == "-h":
        print_help()
    elif opt in ("-c", "--collection"):
        collection_name = arg
    elif opt in ("-d", "--directory"):
        directory = arg
    elif opt in ("-m", "--mode"):
        mode = arg
    elif opt in ("-q", "--query"):
        query_text = arg
    elif opt in ("-a", "--a"):
        textA = arg
    elif opt in ("-b", "--b"):
        textB = arg

if not collection_name or not directory or not mode:
    print_help()


#####################################################################
# GPU device setup and model loading                                #
#####################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the same models used for embedding creation
biolink = SentenceTransformer("michiyasunaga/BioLinkBERT-base", device=device)

# if UMAP installed, use UMAP, otherwise we will use t-SNE later. 
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

#####################################################################
# Connect to Chroma
#####################################################################
print(f"[INFO] Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=directory)


try:
    collection = client.get_collection(collection_name)
    print(f"[INFO] Connected to collection '{collection_name}'")
except Exception as e:
    print(f"[ERROR] Could not access collection '{collection_name}': {e}")
    sys.exit(1)

#####################################################################
# Mode: Info
#####################################################################
if mode == "info":
    count = collection.count()
    print(f"\nCollection '{collection_name}' Summary:")
    print("----------------------------------------")
    print(f"Total embeddings: {count}")

    sample = collection.get(limit=3, include=["embeddings", "documents"])
    embed_dim = len(sample["embeddings"][0]) if len(sample["embeddings"]) > 0 else 0
    print(f"Embedding dimension: {embed_dim}")

    norms = [np.linalg.norm(vec) for vec in sample["embeddings"]]
    print(f"Sample embedding norms: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")

    print("\nSample Documents:")
    for d in sample["documents"]:
        print(" -", d.replace("\n", " ") + "...")
    sys.exit()

#####################################################################
# Mode: Query (SapBERT-only)
#####################################################################
if mode == "query":
    if not query_text:
        print("[ERROR] Query text required. Use -q '<text>'")
        sys.exit(1)

    print(f"\n[INFO] Querying for: \"{query_text}\"")

    # Generate SapBERT embedding only
    query_vec = sapbert.encode(query_text, normalize_embeddings=True)
    query_vec /= np.linalg.norm(query_vec)
    query_vec = query_vec.tolist()

    results = collection.query(query_embeddings=[query_vec], n_results=10)

    print("\nTop 10 results:")
    print("----------------------------------------")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        sim = 1 - dist
        clean_doc = doc.replace("\n", " ")
        print(f"Similarity: {sim:.4f}")
        print(f"Text: {clean_doc}\n")
    sys.exit()


#####################################################################
# Mode: Pairwise Similarity
#####################################################################
if mode == "pair":
    if not (textA and textB):
        print("[ERROR] Must specify -a and -b texts for pair mode")
        sys.exit(1)

    print(f"\n[INFO] Comparing '{textA}' and '{textB}'")

    # Retrieve stored embeddings for A and B
    data = collection.get(include=["embeddings", "documents"])
    docs = data["documents"]
    embeds = np.array(data["embeddings"])

    # Locate rows matching names in metadata
    embA = None
    embB = None
    for doc, emb in zip(docs, embeds):
        if textA.lower() in doc.lower() and embA is None:
            embA = emb
        if textB.lower() in doc.lower() and embB is None:
            embB = emb
        if embA is not None and embB is not None:
            break

    if embA is None or embB is None:
        print("[WARN] Could not find one or both concepts in metadata.")
        sys.exit(1)

    # Compute cosine similarity
    embA = np.array(embA)
    embB = np.array(embB)
    similarity = np.dot(embA, embB) / (np.linalg.norm(embA) * np.linalg.norm(embB))
    print(f"\nCosine Similarity: {similarity:.4f}")

    print(f"\nConcept A: {textA}")
    print(f"Concept B: {textB}")
    sys.exit()

#####################################################################
# Mode: Clusters
#####################################################################
if mode == "clusters":
    print(f"[INFO] Fetching all embeddings and metadata...")
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    embeds = np.array(data["embeddings"])
    docs = data["documents"]
    metas = data["metadatas"]

    # Normalize embeddings for cosine similarity consistency
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)

    print("[INFO] Running PCA to 50 dims for clustering...")
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(embeds)

    n_clusters = 10
    print(f"[INFO] Performing KMeans (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(reduced)

    print(f"\nCluster summaries:")
    print("----------------------------------------")
    for i in range(n_clusters):
        cluster_indices = [j for j, label in enumerate(labels) if label == i]

        # Extract cluster data
        cluster_docs = [docs[j] for j in cluster_indices]
        cluster_metas = [metas[j] for j in cluster_indices if metas[j] is not None]

        names = [m.get("name", "N/A") for m in cluster_metas]
        curies = [m.get("curie", "N/A") for m in cluster_metas]
        descriptions = cluster_docs

        # TF-IDF summary from descriptions only
        text = " ".join(descriptions)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
        try:
            words = vectorizer.fit([text]).get_feature_names_out()
        except ValueError:
            words = []

        print(f"\nCluster {i} — top terms: {', '.join(words) if len(words) > 0 else '[no clear terms]'}")
        print(f"Representative concepts: {', '.join(names[:5]) if len(names) > 0 else '[no names]'}")
        print(f"Total items: {len(cluster_docs)}")

    print("\n[INFO] Reducing to 2D for visualization...")
    reduced_2d = PCA(n_components=2).fit_transform(reduced)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=labels, cmap="tab10", s=8)
    plt.title(f"KMeans Clusters for '{collection_name}'")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)
    plt.savefig(f"{collection_name}_clusters.png", dpi=300)
    print(f"[INFO] Saved plot to {collection_name}_clusters.png")
    sys.exit()


#####################################################################
# Mode: UMAP Visualization
#####################################################################
if mode == "umap":
    print(f"[INFO] Fetching embeddings and metadata...")
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    embeds = np.array(data["embeddings"])
    docs = data["documents"]
    metas = data["metadatas"]

    print(f"[INFO] Reducing dimensions with {'UMAP' if UMAP_AVAILABLE else 't-SNE'}...")

    # Dimensionality reduction
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        reduced = reducer.fit_transform(embeds)
    else:
        reduced = TSNE(n_components=2, metric="cosine", random_state=42).fit_transform(embeds)

    # Cluster for coloring
    n_clusters = 10
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(embeds)

    # Extract representative names
    names = [m.get("name", f"Concept_{i}") if m else f"Concept_{i}" for i, m in enumerate(metas)]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )

    plt.title(f"UMAP/t-SNE Visualization for '{collection_name}' (colored by cluster)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    # Annotate a few representative points per cluster
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        # pick one representative name near the cluster center
        center = np.mean(reduced[cluster_indices], axis=0)
        closest_idx = cluster_indices[np.argmin(np.linalg.norm(reduced[cluster_indices] - center, axis=1))]
        plt.text(
            reduced[closest_idx, 0],
            reduced[closest_idx, 1],
            names[closest_idx],
            fontsize=8,
            weight="bold",
            alpha=0.9,
        )

    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig(f"{collection_name}_umap.png", dpi=300)
    print(f"[INFO] Saved plot to {collection_name}_umap.png")

    sys.exit()


#####################################################################
# Mode: Find Similar Concepts by Name
#####################################################################
if mode == "similar":
    if not query_text:
        print("[ERROR] Must specify -q '<name>' for similarity lookup")
        sys.exit(1)

    print(f"\n[INFO] Finding concepts similar to '{query_text}'")

    # Step 1: Retrieve the vector by metadata name match
    results = collection.get(where={"name": query_text}, include=["embeddings", "documents", "metadatas"])

    if len(results["embeddings"]) == 0:
        print(f"[WARN] No embeddings found for name '{query_text}'")
        sys.exit(1)


    base_embedding = np.array(results["embeddings"][0])

    # Step 2: Query collection for nearest neighbors using that embedding
    neighbors = collection.query(query_embeddings=[base_embedding.tolist()], n_results=10)

    print("\nTop 10 similar concepts:")
    print("----------------------------------------")
    for doc, meta, dist in zip(
        neighbors["documents"][0],
        neighbors["metadatas"][0],
        neighbors["distances"][0]
    ):
        sim = 1 - dist
        print(f"Similarity: {sim:.4f}")
        print(f"Name: {meta.get('name', 'N/A')}")
        print(f"CURIE: {meta.get('curie', 'N/A')}")
        print("Description:", doc.replace("\n", " "), "\n")
    sys.exit()


#####################################################################
# Unknown mode
#####################################################################
print(f"[ERROR] Unknown mode '{mode}'. Use one of: info, query, pair, clusters, umap.")
print_help()
