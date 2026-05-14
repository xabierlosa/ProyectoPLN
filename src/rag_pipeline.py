from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .embeddings import embed_query


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


def build_prompt(query: str, contexts: List[str]) -> List[dict]:
    context_block = "\n\n".join(contexts)
    system_prompt = (
        "Eres un asistente de cocina. Responde en espanol claro y directo. "
        "Usa el contexto disponible para responder y evita inventar datos. "
        "Si el contexto no es suficiente, pide una aclaracion o propone una opcion "
        "segura basada en tecnicas generales de cocina."
    )
    user_prompt = (
        "Consulta del usuario:\n"
        f"{query}\n\n"
        "Contexto disponible:\n"
        f"{context_block}\n\n"
        "Respuesta:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_llm(model_name: str, device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
    )
    return tokenizer, model


def generate_answer(
    tokenizer,
    model,
    query: str,
    contexts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    messages = build_prompt(query, contexts)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    do_sample = temperature > 0
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    outputs = model.generate(input_ids, **gen_kwargs)

    generated = outputs[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
