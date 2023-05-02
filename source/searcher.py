from PIL import Image
from img2vec_pytorch import Img2Vec
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

from config import QDRANT_HOST, QDRANT_PORT


class QDrant:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def recreate(self, collection_name, vector_size):
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine"),
        )

    def upload(self, embeddings, payload):
        self.qdrant_client.upload_collection(
            collection_name="Faces", vectors=embeddings, payload=payload, ids=None
        )

    def find_similar_face(self, path_image):
        im = Image.open(path_image)
        img2vec = Img2Vec()
        embedding = img2vec.get_vec(im).tolist()

        result = self.qdrant_client.search(
            collection_name="Faces",
            query_vector=embedding,
            limit=2,
        )

        face = result[0]
        face_score, face_name = face.score, face.payload["name"]
        return face_name, face_score
