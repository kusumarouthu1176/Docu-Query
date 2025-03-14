from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
import faiss
import numpy as np

def split_text(text):
    """Splits long text into smaller chunks for processing."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return splitter.split_text(text)

def get_embeddings(text_chunks):
    """Generates embeddings for text chunks using Google PALM."""
    model = GooglePalmEmbeddings()
    return [model.embed(chunk) for chunk in text_chunks]

def store_embeddings(embeddings):
    """Stores embeddings in FAISS vector store for retrieval."""
    dimension = len(embeddings[0])  # Get embedding dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index
