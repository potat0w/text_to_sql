from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def compute_dtype(quant: dict) -> torch.dtype:
    name = quant["compute_dtype"]
    if name not in DTYPES:
        raise ValueError(f"Unknown compute_dtype {name!r}. Expected one of {sorted(DTYPES)}.")
    return DTYPES[name]


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, quant: dict, *, train_mode: bool):
    dtype = compute_dtype(quant)
    kwargs = {"device_map": "auto", "torch_dtype": dtype, "attn_implementation": "sdpa"}
    if quant["load_in_4bit"]:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bool(quant["bnb_4bit_use_double_quant"]),
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = not train_mode
    if train_mode and quant["load_in_4bit"]:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    if not train_mode:
        model.eval()
    return model


def load_adapter(model, adapter_path: str):
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model
