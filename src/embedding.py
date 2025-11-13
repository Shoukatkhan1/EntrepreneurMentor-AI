# src/embedding.py
from langchain_huggingface import HuggingFaceEmbeddings

# Download HuggingFace embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
