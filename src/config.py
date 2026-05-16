from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "sesgos" / "ChefGPT_Dataset_downsampled_top10pct_authors.json"
ARTIFACTS_DIR = ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
DOCS_PATH = ARTIFACTS_DIR / "docs.jsonl"

EMBED_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
