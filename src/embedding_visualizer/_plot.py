import plotly.graph_objects as go


class EmbeddingPlot:
    """Wrapper around a Plotly figure with convenience methods."""

    def __init__(self, fig: go.Figure):
        self.fig = fig

    def display(self):
        """Show the plot. In Jupyter, renders inline."""
        self.fig.show()

    def to_html(self, path: str):
        """Write a self-contained HTML file."""
        self.fig.write_html(path, full_html=True, include_plotlyjs=True)

    def _repr_html_(self):
        """Jupyter auto-display support."""
        return self.fig.to_html(full_html=False, include_plotlyjs="cdn")
