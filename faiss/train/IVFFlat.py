import faiss
import numpy as np

nlist = 256
n_probe = 3

embeddings = np.random.rand(768, 10000).astype(np.float32)

embedding_dim = embeddings.shape[1]
quantizer = faiss.IndexFlatIP(embedding_dim)

index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
index.train(embeddings)
index.add(embeddings)
index.nprobe = n_probe
