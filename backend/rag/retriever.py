"""Embedding and ChromaDB retrieval for Gemmy's knowledge base."""

from typing import Any

import chromadb
import requests

from backend.config import (
	CHROMA_COLLECTION_NAME,
	CHROMA_PATH,
	OLLAMA_BASE_URL,
	OLLAMA_EMBED_MODEL,
	RAG_TOP_K,
	detect_program_category,
)


class EmbeddingError(Exception):
	"""Raised when Ollama cannot generate embeddings."""


class ChromaError(Exception):
	"""Raised when the local ChromaDB cannot be initialized or queried."""


def get_collection() -> Any:
	"""Open the persistent Gemmy collection, creating it if needed."""
	try:
		CHROMA_PATH.mkdir(parents=True, exist_ok=True)
		client = chromadb.PersistentClient(path=str(CHROMA_PATH))
		return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
	except Exception as error:
		raise ChromaError(f"Unable to initialize ChromaDB: {error}") from error


def embed_texts(texts: list[str]) -> list[list[float]]:
	"""Generate embeddings for text values using Ollama's embed endpoint."""
	if not texts:
		return []

	try:
		response = requests.post(
			f"{OLLAMA_BASE_URL}/embed",
			json={"model": OLLAMA_EMBED_MODEL, "input": texts},
			timeout=(10, 300),
		)
		response.raise_for_status()
		data = response.json()
		embeddings = data["embeddings"]
	except requests.RequestException as error:
		raise EmbeddingError(
			f"Unable to generate embeddings with Ollama model "
			f"'{OLLAMA_EMBED_MODEL}': {error}"
		) from error
	except (ValueError, KeyError, TypeError) as error:
		raise EmbeddingError(
			"Ollama returned an invalid embedding response. "
			"Check that the embedding model is installed."
		) from error

	if not isinstance(embeddings, list) or len(embeddings) != len(texts):
		raise EmbeddingError("Ollama returned an unexpected number of embeddings.")
	return embeddings


def retrieve(question: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
	"""Return the most relevant stored chunks for a user question."""
	collection = get_collection()
	if collection.count() == 0:
		return []

	query_embedding = embed_texts([question])[0]
	query_options = {
		"query_embeddings": [query_embedding],
		"n_results": max(1, top_k),
		"include": ["documents", "metadatas", "distances"],
	}
	program_category = detect_program_category(question)
	if program_category != "General":
		query_options["where"] = {"category": program_category}
	results = collection.query(**query_options)

	documents = results.get("documents", [[]])[0]
	metadatas = results.get("metadatas", [[]])[0]
	distances = results.get("distances", [[]])[0]
	return [
		{
			"text": document,
			"metadata": metadatas[index] or {},
			"distance": distances[index] if index < len(distances) else None,
		}
		for index, document in enumerate(documents)
	]
