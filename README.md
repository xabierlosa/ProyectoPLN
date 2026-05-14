# RAG MVP (recetas)

Este MVP crea un indice FAISS a partir de un JSON de recetas y permite responder preguntas de cocina con un modelo abierto.

## Estructura

- src/: modulos del pipeline (ingesta, chunking, embeddings, FAISS, RAG)
- scripts/: pruebas rapidas
- notebooks/: orquestacion en cuaderno
- artifacts/: se crea al construir el indice

## Instalacion rapida

```bash
pip install -r requirements.txt
```

## Construir indice

```bash
python -m src.cli_build_index --data ChefGPT_Dataset_Random_Sample.json --out_dir artifacts
```

## Probar consulta

```bash
python -m src.cli_chat --index artifacts/index.faiss --docs artifacts/docs.jsonl --query "Tengo pollo y arroz, que puedo cocinar?"
```

## Notas para Colab

- En Colab puedes instalar faiss-gpu en lugar de faiss-cpu.
- Si quieres mas calidad, cambia el modelo a Qwen/Qwen2.5-14B-Instruct.
- Para embeddings, se recomienda intfloat/multilingual-e5-base.
