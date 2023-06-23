import numpy as np
import faiss
from pathlib import Path


class FaissIVFPQ:
    """
    FaissIVFPQ 는 Faiss 라이브러리를 이용하여 IVF + PQ 알고리즘을 구현한 클래스입니다.
    IVF - 벡터를 미리 분류하여 벡터 전체를 검색하지 않고 특정 클러스터에서만 검색하는 방법입니다.
     * IVF를 사용하면 검색 속도를 높일 수 있습니다.

     : n_list : 클러스터의 수를 지정합니다.
     : n_probe : 검색할 최대 클러스터의 수를 지정합니다.

    PQ -  벡터를 양자화하는 방법입니다.
     * PQ를 사용하면 벡터의 차원을 줄일 수 있습니다.(faiss index의 메모리 사용량 감소)
     : m : 최종 벡터의 중심 수
     : bits : 각 중심에서 벡터를 양자화하는데 사용되는 비트 수를 지정합니다.

    """

    def __init__(self, nlist: int = 256, n_probe: int = 3, m: int = 384, bits: int = 8):
        self.nlist = nlist
        self.n_probe = n_probe
        self.m = m
        self.bits = bits

    def train(self, embedding: np.ndarray):
        """
        IVFPQ 모델을 학습합니다.
        IndexFlatIP : 내적을 사용하여 벡터를 정렬합니다.
        IndexIVFPQ : IVF + PQ 알고리즘을 사용합니다.
        faiss.normalize_L2 : 벡터를 L2 정규화합니다.
         -> Faiss의 결과를 dot product로 비교하기 위해 사용합니다.

        : embedding : 학습에 사용할 벡터를 지정합니다.
        """
        embedding_dim = embedding.shape[1]

        if embedding_dim % self.m != 0:
            raise ValueError(f"embedding_dim({embedding_dim}) % m({self.m}) != 0")

        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            embedding_dim,
            self.nlist,
            self.m,
            self.bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = self.n_probe

        faiss.normalize_L2(embedding)
        index.train(embedding)
        index.add(embedding)
        return index

    def save(self, index, name: str, export_path: Path):
        """
        : name : 학습된 모델의 이름을 지정합니다.
        : export_path : 학습된 모델을 저장할 경로를 지정합니다.
        """
        index_file_name = (
            f"{name}_n{self.nlist}_nprobe{self.n_probe}_m{self.m}_b{self.bits}.index"
        )
        faiss.write_index(index, str(export_path) + index_file_name)
