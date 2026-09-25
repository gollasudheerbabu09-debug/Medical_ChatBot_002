# 🩺 MedAI — Medical QA with Qwen2.5 + QLoRA + RAG (FAISS)

A medical question-answering assistant built by fine-tuning **Qwen2.5-7B-Instruct** with **QLoRA** on the
**MedQuAD** dataset and grounding answers with **retrieval-augmented generation** (sentence embeddings + FAISS),
served through a **Gradio** app on **Hugging Face Spaces**.

> Educational project — not medical advice.

**Live demo:** `https://huggingface.co/spaces/<your-hf-username>/MedAI`  
**LoRA adapter:** `https://huggingface.co/<your-hf-username>/qwen2.5-7b-medquad-qlora`

## Architecture
```
TRAINING   MedQuAD -> clean/dedupe -> Qwen chat template -> train/val/test (80/10/10)
           -> Qwen tokenizer -> 4-bit NF4 frozen Qwen + LoRA (r=16, attn + MLP)
           -> cross-entropy on answer tokens -> backprop -> paged 8-bit AdamW -> LoRA adapter

RAG INDEX  train-split answers -> 180-word chunks (40 overlap) -> bge-small-en-v1.5 -> FAISS IndexFlatIP

INFERENCE  question -> query embedding -> FAISS top-K -> question + context
           -> Qwen2.5 + merged LoRA -> streamed answer + sources -> Gradio
```

## Results
Fill in from the last cell of the notebook after your run (`eval_results.csv`).

| Model | ROUGE-L | BERTScore-F1 |
|---|---|---|
| Qwen2.5-7B-Instruct (base) | – | – |
| + QLoRA fine-tuning | – | – |
| + QLoRA + RAG | – | – |

Validation loss / perplexity (fine-tuned): – / –

## Repository layout
```
notebooks/MedAI_QLoRA_RAG_pipeline.ipynb   end-to-end: data -> QLoRA -> eval -> FAISS -> deploy
notebooks/00_original_exploration.ipynb    first version (kept for reference)
space/app.py            Gradio app (ZeroGPU-ready, streaming, shows retrieved sources)
space/rag.py            cleaning, split, chunking, embeddings, FAISS, prompt building
space/requirements.txt  Space dependencies
space/README.md         Hugging Face Space config
docs/                   project flow notes (PDF) and DEPLOY.md
```

## Reproduce
See [docs/DEPLOY.md](docs/DEPLOY.md). In short: open the notebook on Kaggle/Colab with a GPU, set your
GitHub URL and HF username in the config cell, and run all cells.

## Design notes
- **Loss on answers only**: prompt tokens are masked with `-100`, and Qwen's own pad token is kept, so the
  model learns the `<|im_end|>` stop token.
- **Dynamic padding at 512 tokens** instead of padding to 2048 cuts training compute several-fold.
- **No test leakage in RAG**: answers that appear in the test split are excluded from the FAISS index.
- **4-bit for training, bf16 for serving**: QLoRA trains cheaply; on a GPU Space the adapter is merged into a bf16 base.
