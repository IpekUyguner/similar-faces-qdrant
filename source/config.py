import os

DATA_DIR = "/home/ipek/PycharmProjects/similar_faces/data/faces"

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)

EMBEDDINGS_LOCATION = "/home/ipek/PycharmProjects/similar_faces/embeddings.npy"
EMBEDDINGS_IDS = "/home/ipek/PycharmProjects/similar_faces/embeddings_ids.txt"
