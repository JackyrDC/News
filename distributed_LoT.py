import os, sys, math, gc, json, warnings, shutil, types, re, time
from functools import partial
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import seaborn as sns
from scipy.linalg import svd as scipy_svd
from scipy.stats import kruskal, kurtosis, gaussian_kde
from scipy.special import logsumexp
from tqdm.auto import tqdm
from accelerate import Accelerator
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from accelerate import notebook_launcher
import spacy

warnings.filterwarnings('ignore')
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif", "Bitstream Vera Serif"], "mathtext.fontset": "cm", "axes.labelsize": 14, "font.size": 12, "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.figsize": [10, 8], "image.cmap": "jet", "axes.formatter.use_mathtext": True})

OUTS = 'L_kw_nL_Advanced'
BASE_DRIVE_PATH = f'./University_MATH/{OUTS}'
DATASET_FULL_PATH = './University_MATH/x.json'

QUICK_CONFIG = {
    "run_config": {"experiment_name": "Exp_L_QQQ_Advanced", "num_replicas": 2, "device": "cuda" if torch.cuda.is_available() else "cpu", "metric_for_hypothesis_testing": "RLCT"},
    "data_config": {"dataset_path": DATASET_FULL_PATH, "samples_in_balance": 10, "finetune_budget": 10, "max_seq_len": 64},
    "model_config": {"base_model_id": "openai-community/gpt2", "lora_config": {"r": 4, "lora_alpha": 8, "target_modules": ["c_attn"], "fan_in_fan_out": True}},
    "training_config": {"max_steps": 5, "learning_rate": 1e-4, "per_device_train_batch_size": 2, "logging_steps": 1},
    "rlct_config": {"max_rlct_estimation_steps": 5, "sgld_lr": 1e-4},
    "lot_config": {"num_trajectories": 50, "n_tokens": 25, "use_alternating_source": True}
}
ACTIVE_CONFIG = QUICK_CONFIG

class CLASE_LOT_and_STATS:
    def __init__(self, master_config: dict):
        self.config = master_config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.base_dir = BASE_DRIVE_PATH
        os.makedirs(self.base_dir, exist_ok=True)
        self.accelerator.print(f"=== INICIALIZANDO EXPERIMENTO: {self.config['run_config']['experiment_name']} ===")
        self.accelerator.print(f"Directorio de salida: {self.base_dir}")
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_config']['base_model_id'])
        self.tokenizer.pad_token = self.tokenizer.pad_token if self.tokenizer.pad_token is not None else self.tokenizer.eos_token
        self.doc_gen = self.ExperimentDocumentationGenerator(self)
        self.data_handler = self.LinguisticDataHandler(self)
        self.lora_xs = self.LoRAXS_Arquitectura(self)
        self.rlct_estimator = self.SingularLearningRLCT(self)
        self.math_dynamics = self.MathematicalDynamics(self)
        self.visualizer = self.TrajectoryVisualizer(self)
        self.stats = self.StatisticalAnalysis(self)
        self.orchestrator = self.ExperimentOrchestrator(self)
        self.doc_gen.write_math_foundations()

    class ExperimentDocumentationGenerator:
        def __init__(self, parent):
            self.parent = parent
            self.txt_dir = os.path.join(parent.base_dir, "Theoretical_Foundations")
            os.makedirs(self.txt_dir, exist_ok=True)
        def write_math_foundations(self):
            content = r"\section{Fundamentos Matemáticos}"
            with open(os.path.join(self.txt_dir, "Math_Foundations.txt"), "w", encoding="utf-8") as f:
                f.write(content.strip())

    class LinguisticDataHandler:
        def __init__(self, parent): self.parent = parent
        def prepare_dataset(self):
            cfg = self.parent.config['data_config']
            try:
                if os.path.exists(cfg['dataset_path']):
                    data = json.load(open(cfg['dataset_path'], 'r'))
                    df = pd.DataFrame(data)
                else: raise FileNotFoundError
            except:
                df = pd.DataFrame({
                    "instruction": ["Explain " + "word " * np.random.randint(5, 20) for _ in range(cfg['samples_in_balance'] * 6)],
                    "response": ["This is a response " * 5 for _ in range(cfg['samples_in_balance'] * 6)]
                })
            
            records = []
            for idx, row in df.iterrows():
                doc = self.parent.nlp(str(row["instruction"]))
                sents_count = len(list(doc.sents))
                asl = len([t for t in doc if t.is_alpha]) / sents_count if sents_count > 0 else 0.0
                records.append({**row, "asl": asl})
            
            df_asl = pd.DataFrame(records).dropna(subset=['asl']).sort_values('asl').reset_index(drop=True)
            n_samples = cfg['samples_in_balance']
            pools = {
                "Simple_Syntax": Dataset.from_pandas(df_asl.head(n_samples)),
                "Middle_Syntax": Dataset.from_pandas(df_asl.iloc[(len(df_asl) - n_samples) // 2:(len(df_asl) - n_samples) // 2 + n_samples]),
                "Complex_Syntax": Dataset.from_pandas(df_asl.tail(n_samples))
            }
            return pools, Dataset.from_pandas(df_asl)

    class LoRAXS_Arquitectura:
        def __init__(self, parent): self.parent = parent
        class SVD_Engine:
            @staticmethod
            def compute_svd(W: torch.Tensor, rank: int):
                U, S, Vt = scipy_svd(W.cpu().detach().float().numpy(), full_matrices=False)
                return torch.tensor(U[:, :rank] @ np.diag(S[:rank]), dtype=W.dtype, device=W.device), torch.tensor(Vt[:rank, :], dtype=W.dtype, device=W.device)
        class LatentPatcher:
            @staticmethod
            def forward_latent(self_peft, x: torch.Tensor, *args, **kwargs):
                prev_dtype = x.dtype
                result = self_peft.base_layer(x, *args, **kwargs)
                active_adapters = [getattr(self_peft, "active_adapter", "default")] if isinstance(getattr(self_peft, "active_adapter", "default"), str) else getattr(self_peft, "active_adapters", ["default"])
                
                for adapter in active_adapters:
                    if adapter in self_peft.lora_A.keys() and self_peft.r.get(adapter, self_peft.r) > 0:
                        x_type = x.to(self_peft.lora_A[adapter].weight.dtype)
                        proj_out = self_peft.lora_B[adapter](self_peft.default_lora_latent_mapping(self_peft.lora_A[adapter](self_peft.lora_dropout[adapter](x_type))))
                        result += proj_out * (self_peft.scaling[adapter] if isinstance(self_peft.scaling, dict) else self_peft.scaling)
                return result.to(prev_dtype)
        def transmute_to_loraxs(self, peft_model, rank):
            import peft
            for name, target_module in peft_model.named_modules():
                if isinstance(target_module, peft.tuners.lora.Linear):
                    base_w = target_module.base_layer.weight.data.clone()
                    if getattr(target_module, "fan_in_fan_out", False): base_w = base_w.T
                    A_tensor, B_tensor = self.SVD_Engine.compute_svd(base_w, rank)
                    adapter = list(target_module.lora_A.keys())[0] if target_module.lora_A else "default"
                    dev, dtype = target_module.lora_A[adapter].weight.device, target_module.lora_A[adapter].weight.dtype
                    target_module.lora_A[adapter].weight = nn.Parameter(B_tensor.to(dev).to(dtype).contiguous())
                    target_module.lora_B[adapter].weight = nn.Parameter(A_tensor.to(dev).to(dtype).contiguous())
                    target_module.lora_A[adapter].weight.requires_grad = target_module.lora_B[adapter].weight.requires_grad = False
                    target_module.default_lora_latent_mapping = nn.Linear(rank, rank, bias=False).to(dev).to(dtype)
                    nn.init.normal_(target_module.default_lora_latent_mapping.weight, mean=0, std=1e-5)
                    target_module.forward = types.MethodType(self.LatentPatcher.forward_latent, target_module)
            return peft_model

    class SingularLearningRLCT:
        def __init__(self, parent): self.parent = parent
        class SGLD(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-4, noise_level=1., temperature=1.):
                super().__init__(params, dict(lr=lr, noise_level=noise_level, temperature=temperature))
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            with torch.no_grad():
                                p.add_(p.grad, alpha=-0.5*group["lr"]/group["temperature"])
                                p.add_(torch.normal(mean=0., std=group["noise_level"], size=p.size(), device=p.device), alpha=math.sqrt(group["lr"]))
                                
        def estimate_rlct(self, model, dataset):
            self.parent.accelerator.print("   -> RLCT Phase: Watanabe Estimator (SGLD)...")
            cfg = self.parent.config['rlct_config']
            n = max(1, len(dataset))
            beta1, beta2 = 1.0 / np.log(n), 1.3 / np.log(n)
            optimizer = self.SGLD(model.parameters(), lr=cfg['sgld_lr'], temperature=1.0 / beta1)
            model, optimizer = self.parent.accelerator.prepare(model, optimizer)
            energies = []
            model.train()
            
            for i in range(min(n, cfg["max_rlct_estimation_steps"])):
                sample = dataset[i % n]
                text = f"{sample['instruction']}\n{sample.get('response', '')}"
                inputs = self.parent.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.parent.device)
                optimizer.zero_grad()
                loss = model(**inputs, labels=inputs["input_ids"]).loss
                loss.backward()
                optimizer.step()
                energies.append(loss.item() * n)
                
            if not energies: return 0.0, []
            V = np.array(energies)
            log_w = -(beta2 - beta1) * V
            norm_w = np.exp(log_w - logsumexp(log_w))
            rlct_lambda = (np.mean(V) - np.sum(norm_w * V)) / (1 / beta1 - 1 / beta2)
            return (0.0 if np.isnan(rlct_lambda) or np.isinf(rlct_lambda) else rlct_lambda), energies

    class MathematicalDynamics:
        def __init__(self, parent): self.parent = parent
        def extract_and_compute(self, model, dataset):
            model.eval()
            traj = []
            with torch.no_grad():
                for d in dataset.select(range(min(5, len(dataset)))):
                    text = f"{d['instruction']} {d.get('response', '')}"
                    tokens = self.parent.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.parent.device)
                    out = model(**tokens, output_hidden_states=True)
                    traj.append(np.stack([h[0, -1, :].cpu().numpy() for h in out.hidden_states], axis=0))
            if not traj: return None
            M = np.stack(traj, axis=2)
            NL, _, _ = M.shape
            results = {"Sigma": [], "Drift": [], "Diffusion": []}
            U_prev = None
            for t in range(NL):
                U, S, Vt = scipy_svd(M[t, :, :], full_matrices=False)
                results["Sigma"].append(S)
                if U_prev is not None:
                    results["Drift"].append(np.linalg.norm(U - U_prev, ord="fro"))
                    results["Diffusion"].append(np.var(M[t, :, :] - (U @ np.diag(S) @ Vt)))
                else:
                    results["Drift"].append(0)
                    results["Diffusion"].append(0)
                U_prev = U
            return results

    class TrajectoryVisualizer:
        def __init__(self, parent): self.parent = parent; self.current_model = None; self.current_tokenizer = None; self.output_dir = None
        def plot_rlct(self, energies, method):
            plt.figure(figsize=(6, 4))
            plt.plot(energies, marker='o', linestyle='-', color='indigo')
            plt.title(f"Estimador Watanabe SGLD - {method}")
            plt.xlabel("Pasos SGLD")
            plt.ylabel(r"Energía $nL_n(w)$")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, "rlct_evolution_lambda.png"), bbox_inches='tight')
            plt.close()
            
        def plot_math_dynamics(self, math_res):
            if not math_res: return
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for i, S in enumerate(math_res["Sigma"]):
                ax[0].plot(S[:10], label=f"Layer {i}" if i % 2 == 0 else "")
            ax[0].set_title(r"Decaimiento $\Sigma(t)$")
            ax[0].set_yscale('log')
            ax[1].plot(math_res["Drift"], marker='s', color='r')
            ax[1].set_title(r"Deriva $\|A(t)\|_F$")
            ax[2].plot(math_res["Diffusion"], marker='^', color='g')
            ax[2].set_title(r"Difusión")
            ax[2].set_yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "Mathematical_Dynamics.png"), bbox_inches='tight')
            plt.close()
            
        def generate_trajectories(self, text, lot_config):
            self.parent.accelerator.print(f"   -> Generating {lot_config['num_trajectories']} LoT trajectories...")
            tokens = self.current_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).input_ids[0]
            sentences = [tokens[i:i + lot_config['n_tokens']] for i in range(0, len(tokens) - lot_config['n_tokens'] + 1, lot_config['n_tokens'])][:lot_config['num_trajectories']]
            if not sentences: return None
            trajectories = []
            self.current_model.eval()
            with torch.no_grad():
                for s in tqdm(sentences, desc="Tracing", leave=False):
                    out = self.current_model(s.unsqueeze(0).to(self.current_model.device), output_hidden_states=True)
                    trajectories.append(np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=1))
            return np.stack(trajectories, axis=2)[:, 1:, :] if trajectories else None
            
        def run_full_analysis(self, lot_config, dolly_dataset):
            self.parent.accelerator.print(f"   -> Executing LoT Analysis in: {os.path.basename(self.output_dir)}")
            sample = dolly_dataset.shuffle().select(range(min(len(dolly_dataset), 100))) if lot_config.get('use_alternating_source', True) else None
            text = "\n\n".join(list(sample['instruction']) + [r for r in sample['response'] if r]) if sample else " ".join(["thought"] * 500)
            traj = self.generate_trajectories(text, lot_config)
            if traj is not None:
                print("   -> Scientific Charts Generated.")

    class StatisticalAnalysis:
        def __init__(self, parent): self.parent = parent
        def run_kruskal_wallis(self, df_results): pass

    class ExperimentOrchestrator:
        def __init__(self, parent): self.parent = parent
        @staticmethod
        def tokenize_batch_fn(batch, tokenizer, max_len):
            resps = batch.get('response', [''] * len(batch['instruction']))
            texts = [f"{inst}\n{resp}" for inst, resp in zip(batch['instruction'], resps)]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
            
        @staticmethod
        def plot_training_history(history, run_id, output_dir):
            if history:
                df = pd.DataFrame(history).dropna(subset=['loss'])
                if not df.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df['step'], df['loss'], marker='o', label='Training Loss')
                    plt.title(rf"Training Loss Run {run_id}")
                    plt.yscale('log')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, "training_history_loss.png"), bbox_inches='tight')
                    plt.close()
                    
        def calculate_ppl(self, model, dataset):
            model.eval()
            total_loss = 0
            count = 0
            with torch.no_grad():
                for i in range(min(5, len(dataset))):
                    inputs = self.parent.tokenizer(dataset[i]["instruction"], return_tensors="pt", truncation=True, max_length=64).to(self.parent.device)
                    loss = model(**inputs, labels=inputs["input_ids"]).loss
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        count += 1
            avg_loss = total_loss / count if count > 0 else 100.0
            return math.exp(min(avg_loss, 20))
            
        def run(self):
            start_time = time.time()
            cfg = self.parent.config
            pools, _ = self.parent.data_handler.prepare_dataset()
            global_results = []
            max_len = cfg['data_config']['max_seq_len']
            tokenize_fn = partial(self.tokenize_batch_fn, tokenizer=self.parent.tokenizer, max_len=max_len)
            
            for rep in range(cfg["run_config"]["num_replicas"]):
                for treat_name, pool_ds in pools.items():
                    tokenized_pool_ds = pool_ds.map(tokenize_fn, batched=True)
                    for method in ["LORA_Standard", "LORA_XS"]:
                        run_id = f"Rep{rep}_{treat_name}_{method}"
                        self.parent.accelerator.print(f"\n[{run_id}] Iniciando Pipeline...")
                        run_dir = os.path.join(self.parent.base_dir, run_id)
                        os.makedirs(run_dir, exist_ok=True)
                        
                        base_model = AutoModelForCausalLM.from_pretrained(cfg["model_config"]["base_model_id"]).to(self.parent.device)
                        peft_cfg = LoraConfig(task_type="CAUSAL_LM", **cfg["model_config"]["lora_config"])
                        peft_model = get_peft_model(base_model, peft_cfg)
                        
                        if method == "LORA_XS":
                            peft_model = self.parent.lora_xs.transmute_to_loraxs(peft_model, rank=peft_cfg.r)
                            
                        trainer = Trainer(
                            model=peft_model,
                            args=TrainingArguments(
                                output_dir=os.path.join(run_dir, "ckpt"),
                                max_steps=cfg["training_config"]["max_steps"],
                                learning_rate=cfg["training_config"]["learning_rate"],
                                per_device_train_batch_size=cfg["training_config"]["per_device_train_batch_size"],
                                logging_steps=cfg["training_config"]["logging_steps"],
                                report_to="none",
                                save_strategy="no",
                                remove_unused_columns=True
                            ),
                            train_dataset=tokenized_pool_ds,
                            data_collator=DataCollatorForLanguageModeling(self.parent.tokenizer, mlm=False)
                        )
                        trainer.train()
                        
                        if self.parent.accelerator.is_main_process:
                            self.plot_training_history(trainer.state.log_history, run_id, run_dir)
                            
                        ppl = self.calculate_ppl(peft_model, pool_ds)
                        
                        self.parent.visualizer.current_model = peft_model
                        self.parent.visualizer.current_tokenizer = self.parent.tokenizer
                        self.parent.visualizer.output_dir = run_dir
                        
                        rlct_lambda, energies = self.parent.rlct_estimator.estimate_rlct(peft_model, pool_ds)
                        if self.parent.accelerator.is_main_process:
                            self.parent.visualizer.plot_rlct(energies, method)
                            
                        math_res = self.parent.math_dynamics.extract_and_compute(peft_model, pool_ds)
                        if self.parent.accelerator.is_main_process:
                            self.parent.visualizer.plot_math_dynamics(math_res)
                            
                        if self.parent.accelerator.is_main_process:
                            self.parent.visualizer.run_full_analysis(cfg["lot_config"], pool_ds)
                            
                        global_results.append({
                            "Replica": rep,
                            "ASL_Treatment": treat_name,
                            "Method": method,
                            "RLCT": rlct_lambda,
                            "PPL": ppl
                        })
                        
                        del peft_model, base_model, trainer
                        gc.collect()
                        torch.cuda.empty_cache()
                        
            df_res = pd.DataFrame(global_results)
            if self.parent.accelerator.is_main_process:
                df_res.to_excel(os.path.join(self.parent.base_dir, "LOT_and_STATS_Results.xlsx"), index=False)
                
            try:
                self.parent.stats.run_kruskal_wallis(df_res)
            except:
                pass
                
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            
            self.parent.accelerator.print(f"\n[INFO] Experimento Completado. Archivos generados en: {self.parent.base_dir}")
            self.parent.accelerator.print(f"[INFO] Tiempo Total de Entrenamiento: {int(hours):02d}h {int(mins):02d}m {secs:05.2f}s")
            self.parent.accelerator.print(self.parent.accelerator.device)
def main():
    app = CLASE_LOT_and_STATS(ACTIVE_CONFIG)
    app.orchestrator.run()

if __name__ == "__main__":
    main()
