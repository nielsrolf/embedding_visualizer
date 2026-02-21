from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._embed import get_embeddings
from ._plot import EmbeddingPlot
from ._projections import PrincipalComponent, TextEmbedding

DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#808080", "#ffe119", "#000000",
]


def visualize_embeddings(
    docs: list[dict],
    projection: str | None = None,
    x_projection: PrincipalComponent | TextEmbedding | None = None,
    y_projection: PrincipalComponent | TextEmbedding | None = None,
    title: str = "Embedding Visualization",
    model: str = "text-embedding-3-small",
    figsize: tuple[int, int] | None = None,
) -> EmbeddingPlot:
    """Embed documents and create an interactive 2D scatter plot.

    Args:
        docs: List of dicts with keys: text (required), label, color, line-id, hover.
        projection: "t-sne" or "pca" for both axes.
        x_projection: Custom projection for x-axis (PrincipalComponent or TextEmbedding).
        y_projection: Custom projection for y-axis (PrincipalComponent or TextEmbedding).
        title: Plot title.
        model: OpenAI embedding model name.
        figsize: Optional (width, height) in pixels for the figure.
    """
    texts = [doc["text"] for doc in docs]

    print(f"Computing embeddings for {len(texts)} documents...")
    embeddings = get_embeddings(texts, model=model)

    # Compute projections
    if x_projection is not None and y_projection is not None:
        print("Computing custom projections...")
        x_coords = x_projection.project(embeddings, model)
        y_coords = y_projection.project(embeddings, model)
        x_label = x_projection.axis_label()
        y_label = y_projection.axis_label()
    else:
        proj = projection or "t-sne"
        if proj == "t-sne":
            from sklearn.manifold import TSNE

            perplexity = min(30, len(texts) - 1)
            print(f"Computing t-SNE (perplexity={perplexity})...")
            tsne = TSNE(
                n_components=2,
                metric="cosine",
                init="pca",
                perplexity=perplexity,
                random_state=42,
                max_iter=1500,
            )
            coords = tsne.fit_transform(embeddings)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            x_label = "t-SNE 1"
            y_label = "t-SNE 2"
        elif proj == "pca":
            from sklearn.decomposition import PCA

            print("Computing PCA...")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            x_label = "PC1"
            y_label = "PC2"
        else:
            raise ValueError(f"Unknown projection: {proj!r}. Use 't-sne' or 'pca'.")

    fig = _build_figure(docs, x_coords, y_coords, title, x_label, y_label, figsize)
    return EmbeddingPlot(fig)


def _build_figure(
    docs: list[dict],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    figsize: tuple[int, int] | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Build label -> color mapping
    label_colors: dict[str, str] = {}
    color_idx = 0
    for doc in docs:
        label = doc.get("label")
        if label and label not in label_colors:
            color = doc.get("color")
            if color:
                label_colors[label] = color
            else:
                label_colors[label] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
                color_idx += 1

    # Draw lines first (so they appear behind points)
    line_groups: dict[str, list[int]] = {}
    for i, doc in enumerate(docs):
        line_id = doc.get("line-id")
        if line_id is not None:
            line_groups.setdefault(line_id, []).append(i)

    for line_id, indices in line_groups.items():
        # Determine line color from label of first point, or grey
        first_label = docs[indices[0]].get("label")
        line_color = label_colors.get(first_label, "rgba(150,150,150,0.5)")
        # Make line color semi-transparent
        fig.add_trace(
            go.Scatter(
                x=[x_coords[i] for i in indices],
                y=[y_coords[i] for i in indices],
                mode="lines",
                line=dict(color=line_color, width=1.5),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Group docs by label
    label_indices: dict[str, list[int]] = {}
    unlabeled: list[int] = []
    for i, doc in enumerate(docs):
        label = doc.get("label")
        if label:
            label_indices.setdefault(label, []).append(i)
        else:
            unlabeled.append(i)

    # Add scatter traces per label
    for label, indices in label_indices.items():
        color = label_colors[label]
        hover_texts = [docs[i].get("hover", docs[i]["text"][:100]) for i in indices]
        fig.add_trace(
            go.Scatter(
                x=[x_coords[i] for i in indices],
                y=[y_coords[i] for i in indices],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    # Add unlabeled points
    if unlabeled:
        colors = [docs[i].get("color", "#999") for i in unlabeled]
        hover_texts = [docs[i].get("hover", docs[i]["text"][:100]) for i in unlabeled]
        fig.add_trace(
            go.Scatter(
                x=[x_coords[i] for i in unlabeled],
                y=[y_coords[i] for i in unlabeled],
                mode="markers",
                name="(unlabeled)",
                marker=dict(color=colors, size=7, line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )

    layout_kwargs: dict = dict(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
    )
    if figsize is not None:
        layout_kwargs["width"] = figsize[0]
        layout_kwargs["height"] = figsize[1]
    fig.update_layout(**layout_kwargs)

    return fig
