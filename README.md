# embedding-visualizer

Interactive 2D visualization of text embeddings using OpenAI's embedding API and Plotly.

## Install

```bash
cd embedding_visualizer
uv sync
```

Requires an `OPENAI_API_KEY` environment variable.

## Usage

```python
from embedding_visualizer import visualize_embeddings, PrincipalComponent, TextEmbedding

docs = [
    {
        "text": "This text will be embedded",
        "label": "optional, groups points in the legend",
        "color": "optional, color for the point/label group",
        "line-id": "optional, connects points with the same id",
        "hover": "optional hover text",
    },
    # ...
]

# t-SNE or PCA projection
plot = visualize_embeddings(docs=docs, projection="t-sne")  # or "pca"

# Custom per-axis projection
plot = visualize_embeddings(
    docs=docs,
    x_projection=PrincipalComponent(1),
    y_projection=TextEmbedding("some text to project onto"),
    title="My Embedding Plot",
)

plot.display()              # show in Jupyter or browser
plot.to_html("plot.html")   # self-contained HTML file
```

## Doc fields

| Field | Required | Description |
|-------|----------|-------------|
| `text` | yes | Text to embed |
| `label` | no | Legend group name; points with the same label share a color |
| `color` | no | Point color. If `label` is set, applies to the whole group |
| `line-id` | no | Connects points with the same id in document order |
| `hover` | no | Custom hover text (defaults to first 100 chars of `text`) |

## Projections

- **`projection="t-sne"`** — t-SNE with cosine metric (default)
- **`projection="pca"`** — PCA
- **`PrincipalComponent(n)`** — project onto the nth principal component (1-indexed)
- **`TextEmbedding("text")`** — cosine similarity with a reference text's embedding

## Example

`examples/repo_files.py` embeds every Python file in this repository at multiple truncation points and connects versions of the same file with lines:

```bash
uv run python examples/repo_files.py
```

## Caching

Embeddings are cached to `~/.cache/embedding_visualizer/` so repeated runs don't re-call the API.
