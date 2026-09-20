"""Ingest local knowledge-base files into the persistent Chroma collection."""

import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

from backend.config import KNOWLEDGE_BASE_PATH, PROGRAM_CATEGORY_ALIASES
from backend.rag.retriever import embed_texts, get_collection


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json"}
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


def _extract_file(path: Path) -> tuple[str, str]:
	if path.suffix.lower() == ".pdf":
		try:
			reader = PdfReader(str(path))
			text = "\n".join(page.extract_text() or "" for page in reader.pages)
		except Exception as error:
			raise RuntimeError(f"Unable to extract PDF '{path.name}': {error}") from error
	else:
		try:
			if path.suffix.lower() == ".json":
				text = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
			else:
				text = path.read_text(encoding="utf-8")
		except Exception as error:
			raise RuntimeError(f"Unable to read '{path.name}': {error}") from error

	title = next(
		(
			line.lstrip("# ").strip()
			for line in text.splitlines()
			if line.strip() and len(re.findall(r"[A-Za-z0-9]", line)) > 3
		),
		path.stem.replace("_", " "),
	)
	return re.sub(r"\s+", " ", text).strip(), title


def _chunk_text(text: str, start_offset: int = 0) -> list[tuple[str, int]]:
	if not text:
		return []
	chunks = []
	start = 0
	while start < len(text):
		end = min(start + CHUNK_SIZE, len(text))
		if end < len(text):
			boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
			if boundary > start + CHUNK_SIZE // 2:
				end = boundary + 1
		chunk = text[start:end].strip()
		if chunk:
			chunks.append((chunk, start_offset + start))
		if end >= len(text):
			break
		start = max(end - CHUNK_OVERLAP, start + 1)
	return chunks


def _program_sections(text: str) -> list[tuple[int, int, str]]:
	"""Return non-overlapping ranges for explicit Program N sections."""
	heading_pattern = re.compile(
		r"program\s+\d+\s*[-–—:]\s*(student development program|job readiness program|"
		r"hackathon\s*/?\s*ideathon|corporate training|faculty development program|"
		r"child development program)",
		re.IGNORECASE,
	)
	top_level_pattern = re.compile(
		r"(?:^|\s)(?:[1-9]|1[0-7])\.\s*(?:■\s*)?"
		r"(?:Organization|Community|Programs|Internships|Events|Hackathons|Mentorship|"
		r"Learning|Projects|Certificates|Registration|Fees|Frequently|Support|Important|"
		r"Current|Notes)\b",
		re.IGNORECASE,
	)
	program_matches = list(heading_pattern.finditer(text))
	top_level_positions = [match.start() for match in top_level_pattern.finditer(text)]
	sections = []
	for index, match in enumerate(program_matches):
		heading_text = match.group(1).casefold()
		category = next(
			(
				candidate
				for candidate, aliases in PROGRAM_CATEGORY_ALIASES.items()
				if any(alias in heading_text for alias in aliases)
			),
			"General",
		)
		next_program = (
			program_matches[index + 1].start()
			if index + 1 < len(program_matches)
			else len(text)
		)
		next_top_level = next(
			(position for position in top_level_positions if position > match.start()),
			len(text),
		)
		end = min(next_program, next_top_level)
		sections.append((match.start(), end, category))
	return sections


def ingest() -> int:
	if not KNOWLEDGE_BASE_PATH.exists():
		raise RuntimeError(f"Knowledge-base directory does not exist: {KNOWLEDGE_BASE_PATH}")
	files = sorted(
		path for path in KNOWLEDGE_BASE_PATH.iterdir()
		if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
	)
	if not files:
		raise RuntimeError(f"No supported knowledge-base files found in {KNOWLEDGE_BASE_PATH}")

	print(f"Found {len(files)} knowledge-base file(s)...")
	records = []
	for path in files:
		text, title = _extract_file(path)
		program_sections = _program_sections(text)
		section_ranges = []
		cursor = 0
		for section_start, section_end, category in program_sections:
			if cursor < section_start:
				section_ranges.append((cursor, section_start, "General"))
			section_ranges.append((section_start, section_end, category))
			cursor = section_end
		if cursor < len(text):
			section_ranges.append((cursor, len(text), "General"))
		chunks = [
			(chunk, offset, category)
			for section_start, section_end, category in section_ranges
			for chunk, offset in _chunk_text(
				text[section_start:section_end], start_offset=section_start
			)
		]
		print(f"Extracted {path.name}; created {len(chunks)} chunks...")
		for index, (chunk, offset, category) in enumerate(chunks):
			digest = hashlib.sha256(f"{path.name}:{index}:{chunk}".encode()).hexdigest()
			records.append(
				{
					"id": digest,
					"text": chunk,
					"metadata": {
						"source_filename": path.name,
						"chunk_index": index,
						"source_title": title,
						"category": category,
						"program_category": category,
					},
				}
			)

	if not records:
		raise RuntimeError("Knowledge-base files contained no extractable text.")

	print(f"Generating embeddings for {len(records)} chunks...")
	embeddings = embed_texts([record["text"] for record in records])
	collection = get_collection()
	for path in files:
		collection.delete(where={"source_filename": path.name})
	collection.upsert(
		ids=[record["id"] for record in records],
		documents=[record["text"] for record in records],
		metadatas=[record["metadata"] for record in records],
		embeddings=embeddings,
	)
	print(f"Stored {len(records)} chunks in ChromaDB...")
	print("Ingestion completed successfully.")
	return len(records)


if __name__ == "__main__":
	try:
		ingest()
	except Exception as error:
		print(f"Ingestion failed: {error}")
		raise SystemExit(1)
