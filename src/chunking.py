import re
from typing import Any, Dict, List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [sentence.strip() for sentence in _SENT_SPLIT.split(text) if sentence.strip()]


def chunk_text(text: str, max_words: int = 220, overlap: int = 40) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if current_words + len(sentence_words) > max_words and current:
            chunks.append(" ".join(current).strip())
            if overlap > 0:
                overlap_words = " ".join(current).split()[-overlap:]
                current = [" ".join(overlap_words)]
                current_words = len(overlap_words)
            else:
                current = []
                current_words = 0
        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def chunk_documents(
    docs: List[Dict[str, Any]], max_words: int = 220, overlap: int = 40
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_text(doc["text"], max_words=max_words, overlap=overlap)
        if len(chunks) == 1:
            out.append({**doc, "chunk_id": 0, "text": chunks[0]})
            continue

        for chunk_id, chunk in enumerate(chunks):
            out.append(
                {
                    "id": doc["id"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "metadata": doc["metadata"],
                }
            )
    return out
