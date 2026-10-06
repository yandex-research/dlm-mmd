from typing import Dict, List, Union

import numpy as np
import torch
import transformers
from tqdm import tqdm

from utils.logging_utils import log_for_0


# ============================================
# Perplexity / entropy metrics (PyTorch)
# ============================================
class NLL:
    """PyTorch implementation of NLL metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean_value = torch.tensor(0.0, dtype=torch.float64)
        self.weight = torch.tensor(0.0, dtype=torch.float64)

    def update(self, value: Union[float, torch.Tensor], weight: Union[float, torch.Tensor] = 1.0):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float64)
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float64)
        weight = torch.broadcast_to(weight, value.shape)
        if value.numel() == 0:
            return
        self.mean_value = self.mean_value + value.sum()
        self.weight = self.weight + weight.sum()


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value / self.weight)


class Metrics:
    def __init__(
        self,
        gen_ppl_eval_model_name_or_path=None,
        eval_ppl_batch_size=None,
        eval_context_size=1024,
    ) -> None:
        self.gen_ppl = Perplexity()
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
        self.eval_context_size = eval_context_size
        self._eval_model = None
        self._eval_device = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(gen_ppl_eval_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def reset(self):
        self.gen_ppl.reset()

    def _eval_retokenize(self, text_samples, max_length):
        out = self.tokenizer(
            text_samples,
            return_tensors="np",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        return out["input_ids"], out["attention_mask"]

    @torch.no_grad()
    def _compute_batch_nlls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        model = self._eval_model
        eos_token_id = self.tokenizer.eos_token_id
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        targets = input_ids[:, 1:]
        logits_pred = logits[:, :-1, :]
        log_normalizers = torch.logsumexp(logits_pred.to(torch.float32), dim=-1)
        target_logits = logits_pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1).to(torch.float32)
        nlls = log_normalizers - target_logits
        is_eos = (input_ids == eos_token_id)
        first_eos = (is_eos.to(torch.int32).cumsum(dim=-1) == 1)
        token_mask = (input_ids != eos_token_id)
        valid_tokens = first_eos[:, 1:].to(torch.int32) + token_mask[:, 1:].to(torch.int32)
        return nlls, valid_tokens

    def record_generative_perplexity(
        self,
        text_samples: List[str],
        max_length: int,
    ) -> Dict:
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self._eval_model is None:
            from transformers import AutoModelForCausalLM
            log_for_0(f"Loading PyTorch model: {self.gen_ppl_eval_model_name_or_path}")
            self._eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._eval_model = AutoModelForCausalLM.from_pretrained(
                self.gen_ppl_eval_model_name_or_path,
                torch_dtype=torch.bfloat16,
            ).to(self._eval_device).eval()
            log_for_0("PPL model cached for reuse")

        device = self._eval_device

        samples, attn_mask = self._eval_retokenize(text_samples, max_length=max_length)
        eval_context_size = self.eval_context_size

        batch_size = self.eval_ppl_batch_size or samples.shape[0]
        batch_size = min(batch_size, samples.shape[0]) or 1
        num_batches = (samples.shape[0] + batch_size - 1) // batch_size
        log_for_0(f"PPL: batch_size={batch_size}, {num_batches} batches")

        for i in tqdm(range(num_batches), desc="Evaluating perplexity"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, samples.shape[0])
            batch_samples = samples[batch_start:batch_end]
            batch_attn_mask = attn_mask[batch_start:batch_end]

            for chunk_start in range(0, batch_samples.shape[1], eval_context_size):
                chunk_end = min(chunk_start + eval_context_size, batch_samples.shape[1])
                sample_chunk = batch_samples[:, chunk_start:chunk_end]
                attn_mask_chunk = batch_attn_mask[:, chunk_start:chunk_end]

                input_ids = torch.from_numpy(sample_chunk).to(device).long()
                attn = torch.from_numpy(attn_mask_chunk).to(device).long()
                nlls, valid_tokens = self._compute_batch_nlls(input_ids, attn)

                nlls_np = nlls.detach().cpu().numpy().astype(np.float64)
                valid_tokens_np = valid_tokens.detach().cpu().numpy().astype(np.float64)
                weighted_nlls = nlls_np * valid_tokens_np

                self.gen_ppl.update(torch.from_numpy(weighted_nlls),
                                    torch.from_numpy(valid_tokens_np))

        per_sample_entropy = []
        for i in range(samples.shape[0]):
            valid_len = int(attn_mask[i].sum())
            valid_tokens = samples[i, :valid_len]
            _, counts = np.unique(valid_tokens, return_counts=True)
            probs = counts.astype(np.float32) / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            per_sample_entropy.append(entropy)

        return {
            "ppl": float(self.gen_ppl.compute()),
            "mean_entropy": sum(per_sample_entropy) / len(per_sample_entropy),
        }
