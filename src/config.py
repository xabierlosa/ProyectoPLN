from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "ChefGPT_Dataset_Random_Sample.json"
ARTIFACTS_DIR = ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
DOCS_PATH = ARTIFACTS_DIR / "docs.jsonl"

EMBED_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
