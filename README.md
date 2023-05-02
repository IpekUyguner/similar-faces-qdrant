# similar-faces-qdrant
This is a Streamlit api demo project where you can upload a face photo and finds the most similar face to your input photo by using QDrant Search Engine and Streamlit.

#### Requirements
Download the python requirements for the project.
```
pip install -r requirements.txt
```
#### Dataset 
In this project, 10K faces are used from CelebA dataset as database.
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html                                 
The embeddings&label ids are in 'embeddings.npy", "embeddings_ids.txt" respectively. 

If you want to produce the embeddings from scratch or more embeddings, you can run the script below after you put your data location under the
config file. 
 
 ```
 python source/embedding_extractor.py
 ```
 
 
 #### Start Qdrant
 For searching similar photos, [QDrant Vector Database](https://qdrant.tech/) is used.                                   
 After install [Docker](https://docs.docker.com/get-docker/), you can start the qdrant service in port 6333.
 ```
 sudo docker run -p 6333:6333 qdrant/qdrant:latest
 ```
 To upload the vector embeddings to qdrant search engine, run the command below.
 ```
 python source/embedding_loader.py
 ```
 
 #### Start the application
 You can start the streamlit api in your localhost.
 ```
 streamlit run source/app.py 
 ```
 At the end, you should be able to see the interface as follow:
![Screenshot from 2023-05-02 19-53-41](https://user-images.githubusercontent.com/40366759/235745802-8345d9a1-a3c9-4be2-9e51-3761927a3d5b.png)

 
