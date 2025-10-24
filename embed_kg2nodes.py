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
# 
# takes KG2 nodes (already processed to only include the CURIE, description and name fields)
# and generates hybrid embeddings which are then stored in a chromaDB vector store.

# usage: 
#     python embed.py -i <input_file> -o <output_dur> -c <collection> -m <mode>
#
# This program is not free software; you can not redistribute it
# and/or modify it at all.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#                                                                   
#                                                               
#####################################################################
from datetime import datetime
import time
import glob
import os
import sys
import getopt
import re
import subprocess
import chromadb
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

#####################################################################
#                                                                   #
#                         Start of Program                          #
#                                                                   #
#####################################################################

#####################################################################
# Verbose setup. Silent run by default                              #
#####################################################################
verbose = "No"
total_processed = 0

#####################################################################
# Helper function for timestamped verbose output                    #
#####################################################################
verbose = "No"
def vprint(message):
    """Prints messages only when verbose mode is enabled."""
    if verbose.lower() == "yes":
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} [VERBOSE] {message}")

#####################################################################
# GPU device setup and model loading                                #
#####################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
vprint(f"Using device: {device}")

biolink = SentenceTransformer("michiyasunaga/BioLinkBERT-base", device=device)

#####################################################################
# Set the paths                                                     #
#####################################################################
path = os.getcwd()
script_path = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(script_path, os.pardir))
input_file = ""
output_dir = f"{path}/chromadb"
collection = ""
vector_store = ""
mode = "new"  # default behavior if user doesn't specify

#####################################################################
# datetime object containing current date and time                  #
#####################################################################
current_time = datetime.now()
current_unix = (time.mktime(current_time.timetuple()))

#####################################################################
# Help message                                                      #
#####################################################################
def print_help():
    print ("\n\tUsage:\n")
    print ("\t" + str(sys.argv[0]) + " -i <input_file> -o <output_dir> -c <collection_name> -m <mode> -h -v\n")
    print ("\t-i Path to the TSV file you want to process and embed.")
    print ("\t\t-->Expected to contain CURIE, Name, and Description fields.")
    print ("\n\t-o Directory to save vector store")
    print (f"\t\t default: {output_dir}")
    print ("\n\t-c Collection to create/save embeddings to")
    print ("\t\tdefault: expected to be kg + current version, eg. kg2103")
    print ("\n\t-m Mode:")
    print ("\t\tadd: add to a collection that already exists.")
    print ("\t\toverwrite: overwrite a collection that already exists.")
    print ("\t\tnew: create a new collection (default).")
    print ("\n\t-v Verbose mode / Print dialog")
    print ("\t-h Help message")
    print ("\n")

#####################################################################
# Parse arguments                                                   #
#####################################################################
opts, args = getopt.getopt(
    sys.argv[1:], 
    "hvi:o:c:m:", 
    ["input_file=", "output_dir=", "collection=", "mode="]
)

for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '-v':
        verbose = "Yes"
        vprint("Verbose mode activated.")
    elif opt in ("-i", "--input_file"):
        input_file = arg
        vprint(f"Input file set to: {input_file}")
    elif opt in ("-o", "--output_dir"):
        output_dir = arg
        vprint(f"Output directory set to: {output_dir}")
    elif opt in ("-c", "--collection"):
        collection = arg
        vprint(f"Collection name set to: {collection}")
    elif opt in ("-m", "--mode"):
        mode = arg.lower()
        vprint(f"Collection handling mode set to: {mode}")

#####################################################################
# Validate required inputs                                          #
#####################################################################
if not input_file:
    print("\nError: No input file specified.")
    print_help()
    sys.exit(1)

if not collection:
    print("\nError: No collection specified.")
    print_help()
    sys.exit(1)

if mode not in ["add", "overwrite", "new"]:
    print(f"\nError: Invalid mode '{mode}'. Use one of: add, overwrite, new.\n")
    print_help()
    sys.exit(1)

#####################################################################
# Connect to Chroma and handle collection                           #
#####################################################################
vprint("Attempting connection to Chroma client...")

chroma_client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=f"{output_dir}"
    )
)

try:
    exists = chroma_client.get_collection(name=collection)
    collection_exists = True
    vprint(f"Collection '{collection}' found in Chroma database.")
except Exception:
    collection_exists = False
    vprint(f"Collection '{collection}' does not exist.")


if mode == "new":
    if collection_exists:
        print(f"Error: Collection '{collection}' already exists. Use a different name, or run with --mode overwrite to replace it.")
        sys.exit(1)
    else:
        print(f"Creating new collection: {collection}")
        vector_store = chroma_client.create_collection(
            name=collection, 
            metadata={"source": "kg2-node-descriptions"},
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200,
                    "ef_search": 150, 
                    "max_neighbors": 32, 
                    
                }
            }
        )
        vprint(f"New collection '{collection}' created successfully.")

elif mode == "add" and collection_exists:
    print(f"Adding to existing collection: {collection}")
    vector_store = exists
    vprint("Reusing existing collection without modification.")

elif mode == "overwrite" and collection_exists:
    print(f"Overwriting collection: {collection}")
    chroma_client.delete_collection(name=collection)
    vector_store = chroma_client.create_collection(
            name=collection, 
            metadata={"source": "kg2-node-descriptions"},
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200,
                    "ef_search": 150, 
                    "max_neighbors": 32, 
                }
            }
    )
    vprint("Collection deleted and recreated successfully.")

elif mode in ("add", "overwrite") and not collection_exists:
    print(f"Error: Collection '{collection}' not found for mode '{mode}'. Collection must exist to overwrite or add to it.")
    sys.exit(1)

else:
    print(f"Error: Unknown mode '{mode}'. Must be one of 'add', 'overwrite', or 'new'.")
    sys.exit(1)


#####################################################################
# Start the run                                                     #
#####################################################################
print("\n\tStart of Run!", file=sys.stdout, flush=True)
vprint("Run started successfully. Beginning data processing...")

print("\tProcessing Data: ", file=sys.stdout, flush=True)


# ===============================================================
# Read TSV file: UMLS_ID, Name, Description
# ===============================================================
documents = []
if input_file:
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                umls_id, name, description = parts[0], parts[1], parts[2]
                documents.append((umls_id, name, description))
            elif len(parts) == 2:
                umls_id, name = parts[0], parts[1]
                documents.append((umls_id, name, ""))
            else:
                continue

    ids = [f"doc_{i}" for i in range(len(documents))]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        # ===============================================================
        # Use only the DESCRIPTION as the embedding source.
        # If description is empty, fallback to name or CURIE.
        # ===============================================================
        descriptions = [
            d if d.strip() else f"{u} {n}" for (u, n, d) in batch
        ]

        desc_vecs = biolink.encode(descriptions, normalize_embeddings=True)

        # Optional: safety normalization
        desc_vecs = np.array([v / np.linalg.norm(v) for v in desc_vecs])

        # ===============================================================
        # Store CURIE and Name as metadata, keep description as document
        # ===============================================================
        metadatas = [{"curie": u, "name": n} for (u, n, _) in batch]

        vector_store.add(
            ids=batch_ids,
            embeddings=desc_vecs.tolist(),
            documents=descriptions,  # stored as text body for readability
            metadatas=metadatas      # CURIE + name kept for retrieval
        )

        vprint(f"Added batch {i // batch_size + 1} ({len(batch)} docs)")

else:
    print("\t\tExiting - No input file")


print("\tEnd of Run!\n", file=sys.stdout, flush=True)

sys.exit()

#####################################################################
#                                                                   #
#                           End of Program                          #
#                                                                   #
#####################################################################