import os
import threading
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

LOCK = threading.Lock()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRANSLATION_DEVICE = os.getenv("BHARATDOC_TRANSLATION_DEVICE", "cpu")


@lru_cache(maxsize=1)
def llm(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16 if DEVICE == "mps" else torch.float32).to(DEVICE).eval()
    return tokenizer, model


def generate(path, prompt, max_tokens):
    with LOCK:
        tokenizer, model = llm(path)
        messages = [{"role": "system", "content": "You are BharatDoc AI. Answer faithfully from the supplied document. Never invent document facts."}, {"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


LANGUAGES = {
    "eng_Latn": "eng_Latn", "hin_Deva": "hin_Deva", "ben_Beng": "ben_Beng",
    "tam_Taml": "tam_Taml", "tel_Telu": "tel_Telu", "mar_Deva": "mar_Deva",
    "guj_Gujr": "guj_Gujr", "kan_Knda": "kan_Knda", "mal_Mlym": "mal_Mlym",
    "pan_Guru": "pan_Guru", "ory_Orya": "ory_Orya", "asm_Beng": "asm_Beng",
}


@lru_cache(maxsize=1)
def translator(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True, dtype=torch.float16 if TRANSLATION_DEVICE == "mps" else torch.float32).to(TRANSLATION_DEVICE).eval()
    return tokenizer, model


def translate_text(path, text, source, target):
    if source not in LANGUAGES or target not in LANGUAGES:
        raise ValueError("Unsupported language")
    with LOCK:
        tokenizer, model = translator(path)
        tokenizer.src_lang = source
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(TRANSLATION_DEVICE)
        with torch.inference_mode():
            output = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target), max_new_tokens=256)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
