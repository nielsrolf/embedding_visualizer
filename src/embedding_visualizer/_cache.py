import hashlib
import json
import os

CACHE_DIR = os.path.expanduser("~/.cache/embedding_visualizer")


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


def get_cached(text: str, model: str) -> list[float] | None:
    key = _cache_key(text, model)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def set_cached(text: str, model: str, embedding: list[float]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(text, model)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(embedding, f)
