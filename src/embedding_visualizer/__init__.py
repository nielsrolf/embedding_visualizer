from ._visualize import visualize_embeddings
from ._projections import PrincipalComponent, TextEmbedding
from ._plot import EmbeddingPlot

plot_embeddings = visualize_embeddings

__all__ = [
    "visualize_embeddings",
    "plot_embeddings",
    "PrincipalComponent",
    "TextEmbedding",
    "EmbeddingPlot",
]
