import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_documents
from src.data_loader import build_documents, load_recipes


def main() -> None:
    data_path = ROOT / "ChefGPT_Dataset_Random_Sample.json"
    recipes = load_recipes(data_path)
    docs = build_documents(recipes)
    chunks = chunk_documents(docs, max_words=220, overlap=40)

    print(f"recipes={len(recipes)} docs={len(docs)} chunks={len(chunks)}")
    if chunks:
        print(chunks[0]["text"][:200])


if __name__ == "__main__":
    main()
