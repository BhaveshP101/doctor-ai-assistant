import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_index(index, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]