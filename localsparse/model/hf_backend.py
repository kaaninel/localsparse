"""HuggingFace generation backend (transformers).

Lazy-imports transformers so the rest of LocalSparse stays usable on
machines that don't have it installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List


class HFBackend:
    def __init__(self, model_path: Path, *, device: str = "auto", torch_dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        dtype = (torch.bfloat16 if torch_dtype == "auto" else getattr(torch, torch_dtype))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
        )
        self.model.eval()

    def generate(self, prompt: str, *, max_new_tokens: int = 512,
                 stop: Optional[List[str]] = None) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=False)
        if stop:
            for s in stop:
                i = text.find(s)
                if i != -1:
                    text = text[:i]
                    break
        return text
