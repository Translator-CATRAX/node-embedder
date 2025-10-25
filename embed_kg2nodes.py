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
from tqdm import tqdm
import math
import time



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
        print(f"\t[VERBOSE] {timestamp} {message}")
        time.sleep(.5)

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
output_dir = f"{os.getcwd()}/chromadb"
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
    elif opt in ("-i", "--input_file"):
        input_file = arg
    elif opt in ("-o", "--output_dir"):
        output_dir = arg
    elif opt in ("-c", "--collection"):
        collection = arg
    elif opt in ("-m", "--mode"):
        mode = arg.lower()

print("\n\n")
vprint(" Verbose mode activated.")
vprint(f" Input file set to: {input_file}")
vprint(f" Collection name set to: {collection}")
vprint(f" Collection handling mode set to: {mode}")
vprint(f" Output directory set to: {output_dir}")
print("\n\n")
#####################################################################
# Validate required inputs                                          #
#####################################################################
if not input_file:
    print("\n[ERROR] No input file specified.")
    print_help()
    sys.exit(1)

if not collection:
    print("\n[ERROR] No collection specified.")
    print_help()
    sys.exit(1)

if mode not in ["add", "overwrite", "new"]:
    print(f"\n[ERROR] Invalid mode '{mode}'. Use one of: add, overwrite, new.\n")
    print_help()
    sys.exit(1)



#####################################################################
# Connect to Chroma and handle collection                           #
#####################################################################
vprint(" Attempting connection to Chroma client...")
os.makedirs(output_dir, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=output_dir)

try:
    exists = chroma_client.get_collection(name=collection)
    collection_exists = True
    vprint(f" Collection '{collection}' found in Chroma database.")
except Exception:
    collection_exists = False
    vprint(f" Collection '{collection}' does not exist.")


if mode == "new":
    if collection_exists:
        print(f"\t[ERROR] Collection '{collection}' already exists. Use a different name, or run with --mode overwrite to replace it.")
        sys.exit(1)
    else:
        print(f"\t[INFO]                           Creating new collection: {collection}")
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
        if vector_store != "":
            vprint(f" New collection '{collection}' created successfully.")
        else: 
            print(f"\t[ERROR] vector store {collection} not created")

elif mode == "add" and collection_exists:
    print(f"\t[INFO]             Adding to existing collection: {collection}")
    vector_store = exists
    

elif mode == "overwrite" and collection_exists:
    print(f"\t[INFO] Overwriting collection: {collection}")
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
    print(f"[ERROR] Collection '{collection}' not found for mode '{mode}'. Collection must exist to overwrite or add to it.")
    sys.exit(1)

else:
    print(f"[ERROR] Unknown mode '{mode}'. Must be one of 'add', 'overwrite', or 'new'.")
    sys.exit(1)


#####################################################################
# Start the run                                                     #
#####################################################################
print("\n\t[INFO]                           Start of Run!", file=sys.stdout, flush=True)


# ===============================================================
# Read TSV file: ID, Name, Description
# ===============================================================
documents = []
if input_file:
    vprint(" Opening input file to get data")
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                id, name, description = parts[0], parts[1], parts[2]
                documents.append((id, name, description))
            elif len(parts) == 2:
                id, name = parts[0], parts[1]
                documents.append((id, name, ""))
            else:
                continue
    if len(documents) < 1: 
        print(f"\t[ERROR] failed to extract documents from input file")

    ids = [f"doc_{i}" for i in range(len(documents))]
    batch_size = 100
    total = math.ceil(len(documents) / batch_size)
    print("\n\nBeginning vector store embeddings. This may take a while....\n\n")
    start = time.time()
    for i in tqdm(range(0, len(documents), batch_size),
              total=total,
              desc=f"Embedding batches ({batch_size} docs each)",
              unit="batch"):

        batch = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        descriptions = [
            d if d.strip() else f"{u} {n}"
            for (u, n, d) in batch]
        desc_vecs = biolink.encode(descriptions, normalize_embeddings=True)

        # Optional: safety normalization
        desc_vecs = np.array([v / np.linalg.norm(v) for v in desc_vecs])

        metadatas = [{"curie": u, "name": n} for (u, n, _) in batch]

        vector_store.add(
            ids=batch_ids,
            embeddings=desc_vecs.tolist(),
            documents=descriptions,  # stored as text body for readability
            metadatas=metadatas      # CURIE + name kept for retrieval
        )

    vprint(" Finished embedding")
    end = time.time()

    elapsed = end - start
    minutes, seconds = divmod(int(elapsed), 60)
    vprint(f" Total runtime: {minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}")
else:
    print("\t[INFO] Exiting - No input file")

print("\t\t\t\n[INFO]                          End of Run!\n", file=sys.stdout, flush=True)
vprint(f" Vector store saved to: {path}{output_dir}")
# Give Chroma a brief moment to flush data before exit
time.sleep(.5)

#####################################################################
#                                                                   #
#                           End of Program                          #
#                                                                   #
#####################################################################