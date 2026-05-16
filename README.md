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

## Analisis de sesgos

El proyecto incorpora el cuaderno [notebooks/sesgos.ipynb](notebooks/sesgos.ipynb) para revisar sesgos relevantes en un corpus de recetas web: concentracion por autor o fuente, sesgo tematico, sesgo geografico y sesgo de accesibilidad.

Las mitigaciones documentadas incluyen diversificar las fuentes de scraping, limitar la repeticion por dominio, anotar metadatos de procedencia, evaluar por subgrupos y penalizar la redundancia en el retrieval.
