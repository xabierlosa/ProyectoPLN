import argparse
from pathlib import Path

from .chunking import chunk_documents
from .data_loader import build_documents, load_recipes
from .embeddings import embed_passages, load_embedder
from .faiss_index import build_index, save_docs, save_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for recipes")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON file")
    parser.add_argument(
        "--out_dir", type=str, default="artifacts", help="Output directory"
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Embedding model name",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Embedding device (cuda or cpu)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_words", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)

    recipes = load_recipes(data_path)
    docs = build_documents(recipes)
    chunks = chunk_documents(docs, max_words=args.max_words, overlap=args.overlap)

    texts = [doc["text"] for doc in chunks]
    embedder = load_embedder(args.embed_model, device=args.device)
    embeddings = embed_passages(embedder, texts, batch_size=args.batch_size)

    index = build_index(embeddings)
    save_index(index, out_dir / "index.faiss")
    save_docs(chunks, out_dir / "docs.jsonl")

    print(f"Saved index with {len(chunks)} chunks to {out_dir}")


if __name__ == "__main__":
    main()
