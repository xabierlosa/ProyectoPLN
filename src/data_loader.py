import json
from pathlib import Path
from typing import Any, Dict, List


def load_recipes(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_recipe_text(item: Dict[str, Any]) -> str:
    title = str(item.get("titulo", "")).strip()
    ingredients = item.get("ingredientes", []) or []
    instructions = str(item.get("instrucciones", "")).strip()
    total_time = str(item.get("tiempo_total", "")).strip()
    portions = str(item.get("porciones", "")).strip()
    author = str(item.get("autor", "")).strip()

    ing_text = ", ".join([str(x).strip() for x in ingredients if str(x).strip()])

    parts = []
    if title:
        parts.append(f"Titulo: {title}")
    if ing_text:
        parts.append(f"Ingredientes: {ing_text}")
    if instructions:
        parts.append(f"Instrucciones: {instructions}")
    if total_time:
        parts.append(f"Tiempo total: {total_time}")
    if portions:
        parts.append(f"Porciones: {portions}")
    if author:
        parts.append(f"Autor: {author}")

    return "\n".join(parts).strip()


def build_documents(recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for idx, item in enumerate(recipes):
        text = build_recipe_text(item)
        metadata = {
            "id": idx,
            "titulo": item.get("titulo"),
            "autor": item.get("autor"),
            "tiempo_total": item.get("tiempo_total"),
            "porciones": item.get("porciones"),
            "n_ingredientes": item.get("n_ingredientes"),
        }
        docs.append({"id": idx, "text": text, "metadata": metadata})
    return docs
