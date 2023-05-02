import logging
import numpy as np

from config import EMBEDDINGS_LOCATION, EMBEDDINGS_IDS
from searcher import QDrant

def read_embeddings():
    '''
    Reads the embedding and id files respectively.
    :return:
    '''
    embeddings = np.load(EMBEDDINGS_LOCATION).tolist()
    readfile = open(EMBEDDINGS_IDS, "r")
    lines = readfile.readlines()
    readfile.close()
    payload = [{"name": file[0:-1]} for file in lines]
    return embeddings, payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting embeddings")
    embeddings, payload = read_embeddings()
    vector_size = len(embeddings[0])
    collection_name = "Faces"
    qdrant_client = QDrant(collection_name)
    qdrant_client.recreate(collection_name=collection_name, vector_size=vector_size)
    qdrant_client.upload(embeddings=embeddings, payload=payload)
    logger.info("The embeddings uploaded!")
