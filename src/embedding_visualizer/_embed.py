import openai
import tiktoken
import numpy as np

from ._cache import get_cached, set_cached

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.encoding_for_model("gpt-4o")
    return _enc


def truncate(text: str, max_tokens: int = 8192) -> str:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    cached = get_cached(text, model)
    if cached is not None:
        return cached
    client = openai.OpenAI()
    response = client.embeddings.create(model=model, input=truncate(text))
    embedding = response.data[0].embedding
    set_cached(text, model, embedding)
    return embedding


def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings for multiple texts, batching uncached ones."""
    results: list[list[float] | None] = [None] * len(texts)
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        cached = get_cached(text, model)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append((i, text))

    if uncached:
        client = openai.OpenAI()
        # Batch in groups of 2048 (OpenAI limit)
        for batch_start in range(0, len(uncached), 2048):
            batch = uncached[batch_start : batch_start + 2048]
            truncated_texts = [truncate(text) for _, text in batch]
            response = client.embeddings.create(model=model, input=truncated_texts)
            for j, (orig_idx, orig_text) in enumerate(batch):
                embedding = response.data[j].embedding
                results[orig_idx] = embedding
                set_cached(orig_text, model, embedding)

    return np.array(results)
