import os

import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec

from config import DATA_DIR


class embedding_extractor:
    """Extract embeddings from img2vec model.
    """

    def __init__(self):
        self.img2vec = Img2Vec()

    def get_embedding(self, image_path):
        embedding = self.img2vec.get_vec(image_path)
        return embedding.tolist()

    def embeddings_dataset(self, path_image):
        """
        Extract all the embeddigns for the dataset.
        :param path_image: path to data image folder
        :return: embedding list and their ids
        """
        result = []
        ids = []
        for img in os.listdir(path_image):
            current_img = str(os.path.join(path_image, img))
            im = Image.open(current_img)
            #print(current_img)
            img_embedding = self.get_embedding(im)
            result.append(img_embedding)
            ids.append(img)
        return result, ids

    def save_embeddings(self, embeddings, ids):
        """
        Saves the embeddins to npy file and its ids. Later, they
        will be used in QDrant search vector database.
        :param embeddings: embeddings list
        :param ids: embeddings ids
        """
        np.save("../embeddings.npy", embeddings)
        outfile = open("../embeddings_ids.txt", "w")
        outfile.writelines([str(i) + "\n" for i in ids])
        outfile.close()


if __name__ == "__main__":
    extractor = embedding_extractor()
    image_path = os.path.join(DATA_DIR)
    embeddings, ids = extractor.embeddings_dataset(image_path)
    extractor.save_embeddings(embeddings, ids)
