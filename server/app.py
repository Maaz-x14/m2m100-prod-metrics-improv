"""
M2M100 Urdu → Roman Urdu Transliteration Server.
The tokenizer defaults to the fine-tuned tokenizer repo you provided.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = os.environ.get("HF_MODEL", "Mavkif/m2m100_rup_ur_to_rur")
TOKENIZER_ID  = os.environ.get("HF_TOKENIZER", "Mavkif/m2m100_rup_tokenizer_both")
SRC_LANG      = os.environ.get("SRC_LANG", "ur")
TGT_LANG      = os.environ.get("TGT_LANG", "ru")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# FP16 on CUDA for speed; FP32 on CPU
DTYPE         = torch.float16 if DEVICE == "cuda" else torch.float32

# Generation defaults (can be overridden per-request)
DEFAULT_BEAMS      = 4
DEFAULT_MAX_NEW    = 128
MAX_INPUT_CHARS    = 1024     # guard against runaway inputs
MAX_BATCH_SIZE     = 16

# ── Global model state ────────────────────────────────────────────────────────
_model:     Optional[M2M100ForConditionalGeneration] = None
_tokenizer: Optional[M2M100Tokenizer]                = None
_forced_bos: Optional[int]                           = None


def load_model() -> None:
    global _model, _tokenizer, _forced_bos

    if not MODEL_ID:
        raise RuntimeError(
            "HF_MODEL is not set. Provide the Urdu→Roman Urdu checkpoint id before starting the server."
        )

    log.info("Loading tokenizer from %s …", TOKENIZER_ID)
    _tokenizer = M2M100Tokenizer.from_pretrained(TOKENIZER_ID)

    if SRC_LANG not in _tokenizer.lang_code_to_id:
        raise RuntimeError(
            f"Source language '{SRC_LANG}' is not supported by tokenizer. "
            f"Available language codes: {sorted(_tokenizer.lang_code_to_id.keys())[:20]}..."
        )
    if TGT_LANG not in _tokenizer.lang_code_to_id:
        raise RuntimeError(
            f"Target language '{TGT_LANG}' is not supported by tokenizer. "
            f"Available language codes: {sorted(_tokenizer.lang_code_to_id.keys())[:20]}..."
        )

    log.info("Loading model from %s onto %s (dtype=%s) …", MODEL_ID, DEVICE, DTYPE)
    _model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    _model.eval()

    if getattr(_model, 'generation_config', None) is not None:
        _model.generation_config.max_length = None
        if getattr(_model.generation_config, 'no_repeat_ngram_size', None) is None:
            _model.generation_config.no_repeat_ngram_size = 3
        if getattr(_model.generation_config, 'repetition_penalty', None) is None:
            _model.generation_config.repetition_penalty = 1.1

    _tokenizer.src_lang = SRC_LANG
    _forced_bos = _tokenizer.get_lang_id(TGT_LANG)
    if DEVICE == "cuda":
        log.info("Running warm-up pass …")
        dummy = _tokenizer("hello", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _model.generate(
                **dummy,
                forced_bos_token_id=_forced_bos,
                num_beams=1,
                max_new_tokens=8,
            )
        torch.cuda.synchronize()
        log.info("Warm-up done. GPU memory allocated: %.1f MB",
                 torch.cuda.memory_allocated() / 1e6)

    log.info("Model ready on %s. tokenizer=%s model=%s src=%s tgt=%s", DEVICE, TOKENIZER_ID, MODEL_ID, SRC_LANG, TGT_LANG)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Urdu → Roman Urdu Transliteration API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: Optional[str] = Field(None,  description="Single input string")
    texts: Optional[list[str]] = Field(None, description="Batch of input strings")
    num_beams:    int = Field(DEFAULT_BEAMS,   ge=1, le=10)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW, ge=1, le=512)


class TranslateResponse(BaseModel):
    translation:  Optional[str]       = None
    translations: Optional[list[str]] = None
    latency_ms: dict


# ── Core inference helper ──────────────────────────────────────────────────────
def _run_inference(
    texts: list[str],
    num_beams: int,
    max_new_tokens: int,
) -> tuple[list[str], dict]:
    """Run tokenise → generate → decode and return (outputs, latency_dict)."""
    t0 = time.perf_counter()

    _tokenizer.src_lang = SRC_LANG
    encoded = _tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    t1 = time.perf_counter()

    with torch.no_grad():
        generated = _model.generate(
            **encoded,
            forced_bos_token_id=_forced_bos,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t2 = time.perf_counter()

    decoded = _tokenizer.batch_decode(generated, skip_special_tokens=True)
    import re
    decoded = [re.sub(r'^__[a-z]{2,4}__\s*', '', text) for text in decoded]

    t3 = time.perf_counter()

    latency = {
        "tokenize_ms":  round((t1 - t0) * 1000, 2),
        "generate_ms":  round((t2 - t1) * 1000, 2),
        "decode_ms":    round((t3 - t2) * 1000, 2),
        "total_ms":     round((t3 - t0) * 1000, 2),
    }
    log.info("batch=%d beams=%d | %s", len(texts), num_beams, latency)
    return decoded, latency


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_ID,
        "gpu_mem_mb": round(torch.cuda.memory_allocated() / 1e6, 1)
                      if DEVICE == "cuda" else None,
    }


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    # Validate input
    if req.text is None and (req.texts is None or len(req.texts) == 0):
        raise HTTPException(400, "Provide 'text' or 'texts'.")

    if req.text is not None:
        inputs = [req.text]
        is_single = True
    else:
        inputs = req.texts
        is_single = False

    if len(inputs) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Batch size exceeds maximum of {MAX_BATCH_SIZE}.")

    # Guard against empty / oversized strings
    for i, t in enumerate(inputs):
        if not t or not t.strip():
            raise HTTPException(400, f"texts[{i}] is empty.")
        if len(t) > MAX_INPUT_CHARS:
            raise HTTPException(
                400,
                f"texts[{i}] exceeds {MAX_INPUT_CHARS} chars "
                f"(got {len(t)}). Split into shorter segments.",
            )

    outputs, latency = _run_inference(inputs, req.num_beams, req.max_new_tokens)

    if is_single:
        return TranslateResponse(translation=outputs[0], latency_ms=latency)
    return TranslateResponse(translations=outputs, latency_ms=latency)


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
