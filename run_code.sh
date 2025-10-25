#!/usr/bin/env bash

#####################################################################
#  KG2x Node Embedding Runner
#
#  Wrapper script for embed_kg2nodes.py
#  Uses nodes_cleaned.tsv as input and generates embeddings into ./chromadb
#
#  Usage:
#      bash run_embedder.sh [-v]
#
#  Options:
#      -v    Enable verbose mode (passes through to embed_kg2nodes.py)
#
#####################################################################

# Fixed parameters
INPUT_FILE="nodes_cleaned.tsv"
OUTPUT_DIR="./chromadb"
COLLECTION="kg2103_test"
MODE="new"
VERBOSE=""

# Parse optional -v flag
while getopts "v" opt; do
  case $opt in
    v)
      VERBOSE="-v"
      ;;
  esac
done

# Check for required input file
if [ ! -f "$INPUT_FILE" ]; then
  echo "[ERROR] Input file '$INPUT_FILE' not found."
  exit 1
fi

# Ensure Python environment exists
if ! command -v python3 &> /dev/null; then
  echo "[ERROR] Python3 not found. Please activate your virtual environment."
  exit 1
fi

# Run the embedder
python3 embed_kg2nodes.py -i "$INPUT_FILE" -o "$OUTPUT_DIR" -c "$COLLECTION" -m "$MODE" $VERBOSE

# Exit status
if [ $? -ne 0 ]; then
  echo "[ERROR] Embedding process failed. Check log output above."
  exit 1
fi

