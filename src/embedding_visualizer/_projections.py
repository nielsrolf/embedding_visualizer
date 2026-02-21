import numpy as np

from ._embed import get_embedding


class PrincipalComponent:
    """Project embeddings onto the nth principal component (1-indexed)."""

    def __init__(self, n: int):
        self.n = n

    def project(self, embeddings: np.ndarray, model: str) -> np.ndarray:
        from sklearn.decomposition import PCA

        n_components = min(self.n, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(embeddings)
        return transformed[:, self.n - 1]

    def axis_label(self) -> str:
        return f"PC{self.n}"


class TextEmbedding:
    """Project embeddings onto cosine similarity with a reference text."""

    def __init__(self, text: str):
        self.text = text

    def project(self, embeddings: np.ndarray, model: str) -> np.ndarray:
        ref = np.array(get_embedding(self.text, model=model))
        ref_norm = ref / np.linalg.norm(ref)
        # Normalize each row for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        return normalized @ ref_norm

    def axis_label(self) -> str:
        short = self.text[:40]
        if len(self.text) > 40:
            short += "..."
        return f'Similarity to "{short}"'
