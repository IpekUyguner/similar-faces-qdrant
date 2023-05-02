import os

import streamlit as st
from PIL import Image

from config import DATA_DIR
from searcher import QDrant

qdrant_client = QDrant("Faces")


def run_demo():
    st.set_page_config(
        page_title="Find most similar face",
    )

    st.title("Find most similar celebrity to you!")
    st.caption(
        "Hey you can find out the most similar celebrity face to you! Vector space involves 10K faces. You should upload a photo where your face is zoomed for best result."
    )

    uploaded_file = st.file_uploader("Choose a image file in jpg format.", type="jpg")

    # if upload, then search
    if uploaded_file:
        face_name, face_score = qdrant_client.find_similar_face(uploaded_file)
        st.subheader("Most similar face to your photo from our database:")
        image = Image.open(os.path.join(DATA_DIR + "/" + face_name))
        st.image(image, channels="BGR")
    st.caption("Made by Ipek Uyguner")


if __name__ == "__main__":
    run_demo()
