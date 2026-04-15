import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunConfig:
    experiment_name: str = "Exp_MoI_Distributed"
    output_dir: str = "./University_MATH/MoI_Distributed"
    dataset_path: str = "./x.json"
    base_model_id: str = "openai-community/gpt2"
    samples_in_balance: int = 24
    max_seq_len: int = 96
    train_steps: int = 60
    lr: float = 2e-4
    batch_size: int = 2
    seed: int = 42
    num_eval_prompts: int = 12
    max_new_tokens: int = 20


DEFAULT_MOI_GRID = {
    "betas": [0.1, 1.0, 5.0],
    "temperatures": [0.7, 1.0],
    "repetition_penalties": [1.0, 1.2],
    "top_ks": [0, 50],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed MoI experiment with Hugging Face Accelerate")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--no-train", action="store_true", help="Skip LoRA finetuning and only run MoI grid")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig()
    if args.dataset_path:
        cfg.dataset_path = args.dataset_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model_id:
        cfg.base_model_id = args.model_id
    if args.train_steps is not None:
        cfg.train_steps = args.train_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    cfg.seed = args.seed
    cfg.max_new_tokens = args.max_new_tokens
    return cfg


def load_json_or_synthetic(path: str, n: int) -> pd.DataFrame:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "instruction" not in df.columns:
            raise ValueError("Dataset must contain an 'instruction' column")
        if "response" not in df.columns:
            df["response"] = ""
        return df

    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "instruction": [
                "Explain in detail: " + "token " * int(rng.integers(4, 18)) for _ in range(n * 6)
            ],
            "response": ["This is a sample response." for _ in range(n * 6)],
        }
    )


def compute_ttr_pools(df: pd.DataFrame, nlp, samples_in_balance: int) -> Dict[str, Dataset]:
    records: List[dict] = []
    for _, row in df.iterrows():
        text = str(row.get("instruction", ""))
        doc = nlp(text)
        words = [tok.text.lower() for tok in doc if tok.is_alpha]
        ttr = len(set(words)) / len(words) if words else 0.0
        records.append(
            {
                "instruction": text,
                "response": str(row.get("response", "")),
                "ttr": float(ttr),
            }
        )

    df_ttr = pd.DataFrame(records).dropna(subset=["ttr"]).sort_values("ttr").reset_index(drop=True)
    n = min(samples_in_balance, max(1, len(df_ttr) // 3))
    mid_start = max(0, (len(df_ttr) - n) // 2)

    pools = {
        "Low_Rich_Lexicon": Dataset.from_pandas(df_ttr.head(n), preserve_index=False),
        "Middle_Rich_Lexicon": Dataset.from_pandas(df_ttr.iloc[mid_start: mid_start + n], preserve_index=False),
        "High_Rich_Lexicon": Dataset.from_pandas(df_ttr.tail(n), preserve_index=False),
    }
    return pools


def build_lora_model(model_id: str, tokenizer) -> torch.nn.Module:
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(base_model, lora_cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def tokenize_batch(batch: dict, tokenizer, max_len: int) -> dict:
    responses = batch.get("response", [""] * len(batch["instruction"]))
    texts = [f"{inst}\n{resp}" for inst, resp in zip(batch["instruction"], responses)]
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors=None,
    )
    encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
    return encoded


def make_train_loader(ds: Dataset, tokenizer, max_len: int, batch_size: int) -> DataLoader:
    tokenized = ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_len),
        batched=True,
        remove_columns=ds.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)


def train_lora_distributed(accelerator: Accelerator, model, loader: DataLoader, lr: float, max_steps: int) -> List[dict]:
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    log_history: List[dict] = []
    model.train()
    step = 0

    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process, desc="Train")
    while step < max_steps:
        for batch in loader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_scalar = accelerator.gather(loss.detach()).mean().item()
            log_history.append({"step": step + 1, "loss": loss_scalar})
            step += 1
            progress.update(1)

            if step >= max_steps:
                break
    progress.close()
    return log_history


class BayesianMoIDirichletMultinomial:
    def __init__(self, model, tokenizer, beta: float, temp: float, rep_penalty: float, top_k: int):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.temp = temp
        self.rep_penalty = rep_penalty
        self.top_k = top_k

        emb = model.get_input_embeddings()
        if emb is None:
            raise RuntimeError("Model has no input embeddings")
        self.embed_layer = emb
        self.vocab_size = model.config.vocab_size

    def _posterior_embedding(self, logits: torch.Tensor, sampled_token_id: torch.Tensor) -> torch.Tensor:
        logits_adj = logits / max(self.temp, 1e-7)

        if self.top_k > 0:
            k = min(self.top_k, logits_adj.size(-1))
            threshold = torch.topk(logits_adj, k)[0][..., -1, None]
            logits_adj = logits_adj.masked_fill(logits_adj < threshold, -float("inf"))

        probs = torch.softmax(logits_adj, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
        entropy_norm = entropy / math.log(self.vocab_size)

        one_hot = torch.zeros_like(probs).scatter_(1, sampled_token_id.unsqueeze(-1), 1.0)
        weights = (entropy_norm * probs + (self.beta + 1.0 - entropy_norm) * one_hot) / (self.beta + 1.0)
        return torch.matmul(weights, self.embed_layer.weight).unsqueeze(1)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> torch.Tensor:
        self.model.eval()
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        current_embeds = self.embed_layer(inputs.input_ids)
        generated_ids = inputs.input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.model(inputs_embeds=current_embeds).logits[:, -1, :].clone()

            if self.rep_penalty > 1.0:
                used_tokens = set(generated_ids[0].tolist())
                for tok in used_tokens:
                    if logits[0, tok] > 0:
                        logits[0, tok] /= self.rep_penalty
                    else:
                        logits[0, tok] *= self.rep_penalty

            next_token_id = torch.argmax(logits, dim=-1)
            post_emb = self._posterior_embedding(logits, next_token_id)
            current_embeds = torch.cat([current_embeds, post_emb], dim=1)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            if self.tokenizer.eos_token_id is not None and generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break

        return generated_ids


@torch.no_grad()
def compute_text_ppl(model, tokenizer, text: str, max_len: int) -> float:
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    out = model(**encoded, labels=encoded.input_ids)
    loss = out.loss
    if torch.isnan(loss):
        return float("inf")
    return float(torch.exp(loss).item())


def evaluate_moi_grid_distributed(
    accelerator: Accelerator,
    model,
    tokenizer,
    prompts: List[str],
    grid: Dict[str, List[float]],
    max_new_tokens: int,
    max_seq_len: int,
) -> pd.DataFrame:
    combos = list(product(grid["betas"], grid["temperatures"], grid["repetition_penalties"], grid["top_ks"]))

    world_size = accelerator.num_processes
    rank = accelerator.process_index

    local_combos = [c for idx, c in enumerate(combos) if idx % world_size == rank]
    local_records: List[dict] = []

    for beta, temp, rep_penalty, top_k in local_combos:
        moi = BayesianMoIDirichletMultinomial(model, tokenizer, beta, temp, rep_penalty, top_k)
        ppl_values = []
        for prompt in prompts:
            out_ids = moi.generate(prompt, max_new_tokens=max_new_tokens)
            decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            ppl_values.append(compute_text_ppl(model, tokenizer, decoded, max_seq_len))

        local_records.append(
            {
                "beta": float(beta),
                "temperature": float(temp),
                "repetition_penalty": float(rep_penalty),
                "top_k": int(top_k),
                "ppl_mean": float(np.mean(ppl_values)),
                "ppl_std": float(np.std(ppl_values)),
                "num_prompts": int(len(prompts)),
            }
        )

    gathered = gather_object(local_records)
    flat: List[dict] = []

    # gather_object can return different shapes depending on backend/world size:
    # - single process: list[dict]
    # - multi process: list[list[dict]]
    if isinstance(gathered, list):
        for chunk in gathered:
            if isinstance(chunk, list):
                flat.extend([row for row in chunk if isinstance(row, dict)])
            elif isinstance(chunk, dict):
                flat.append(chunk)
    elif isinstance(gathered, dict):
        flat.append(gathered)

    return pd.DataFrame(flat)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    accelerator = Accelerator()
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    accelerator.print(f"=== Starting {cfg.experiment_name} ===")
    accelerator.print(f"Processes: {accelerator.num_processes} | Device: {accelerator.device}")

    t0 = time.time()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required. Run: python -m spacy download en_core_web_sm"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df_raw = load_json_or_synthetic(cfg.dataset_path, cfg.samples_in_balance)
    pools = compute_ttr_pools(df_raw, nlp, cfg.samples_in_balance)

    all_results: List[pd.DataFrame] = []

    for treat_name, pool_ds in pools.items():
        accelerator.print(f"\n[{treat_name}] Preparing model and loaders")

        model = build_lora_model(cfg.base_model_id, tokenizer)
        train_loader = make_train_loader(pool_ds, tokenizer, cfg.max_seq_len, cfg.batch_size)

        log_history: List[dict] = []
        if not args.no_train:
            log_history = train_lora_distributed(
                accelerator=accelerator,
                model=model,
                loader=train_loader,
                lr=cfg.lr,
                max_steps=cfg.train_steps,
            )

        # Model is already prepared by accelerate during training. In no-train mode, prepare only model.
        if args.no_train:
            model = accelerator.prepare(model)

        prompts = [pool_ds[i]["instruction"] for i in range(min(len(pool_ds), cfg.num_eval_prompts))]
        if not prompts:
            prompts = ["Explain gravity in simple terms."]

        df_grid = evaluate_moi_grid_distributed(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            grid=DEFAULT_MOI_GRID,
            max_new_tokens=cfg.max_new_tokens,
            max_seq_len=cfg.max_seq_len,
        )

        if accelerator.is_main_process and not df_grid.empty:
            df_grid.insert(0, "Treatment", treat_name)
            sort_cols = [
                c
                for c in ["ppl_mean", "beta", "temperature", "repetition_penalty", "top_k"]
                if c in df_grid.columns
            ]
            if sort_cols:
                df_grid = df_grid.sort_values(sort_cols)
            all_results.append(df_grid)

            treat_dir = os.path.join(cfg.output_dir, treat_name)
            os.makedirs(treat_dir, exist_ok=True)

            df_grid.to_csv(os.path.join(treat_dir, "moi_grid_results.csv"), index=False)
            if log_history:
                pd.DataFrame(log_history).to_csv(os.path.join(treat_dir, "train_log.csv"), index=False)

        accelerator.wait_for_everyone()

        # Free memory between treatments.
        del model
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        if all_results:
            summary = pd.concat(all_results, ignore_index=True)
            summary.to_csv(os.path.join(cfg.output_dir, "moi_results_all_treatments.csv"), index=False)
            summary.to_excel(os.path.join(cfg.output_dir, "moi_results_all_treatments.xlsx"), index=False)
            best = summary.sort_values("ppl_mean").groupby("Treatment", as_index=False).first()
            best.to_csv(os.path.join(cfg.output_dir, "moi_best_by_treatment.csv"), index=False)
            print("\n[INFO] Results written to:", cfg.output_dir)
        else:
            print("\n[WARN] No results were produced.")

    elapsed = time.time() - t0
    accelerator.print(f"[INFO] Finished in {elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()
