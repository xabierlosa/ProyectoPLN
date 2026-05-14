import argparse
from pathlib import Path
from typing import List, Tuple

from .embeddings import embed_query, load_embedder
from .faiss_index import load_docs, load_index
from .rag_pipeline import generate_answer, load_llm


def retrieve(
    query: str,
    embedder,
    index,
    docs: List[dict],
    top_k: int = 6,
) -> Tuple[List[str], List[float]]:
    query_vec = embed_query(embedder, query).astype("float32")
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)

    contexts: List[str] = []
    for idx in indices[0]:
        if idx < 0:
            continue
        contexts.append(docs[idx]["text"])

    return contexts, scores[0].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the recipe RAG")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--docs", type=str, required=True)
    parser.add_argument(
        "--embed_model",
        type=str,
        default="intfloat/multilingual-e5-base",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--show_context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index = load_index(Path(args.index))
    docs = load_docs(Path(args.docs))
    embedder = load_embedder(args.embed_model, device=args.device)

    contexts, scores = retrieve(args.query, embedder, index, docs, top_k=args.top_k)

    if args.show_context:
        print("\n".join(contexts))
        print("Scores:", scores)

    tokenizer, model = load_llm(args.llm_model)
    answer = generate_answer(tokenizer, model, args.query, contexts)
    print(answer)


if __name__ == "__main__":
    main()
