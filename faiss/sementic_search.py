import faiss
import numpy as np

faiss_index = faiss.read_index("faiss.index")


def sementic_search(context):
    # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
    question_embedding = sentence_embedding(context)
    question_embedding = np.expand_dims(question_embedding, axis=0)
    D, I = faiss_index.search(question_embedding, 3)
    return D[0], I[0]
