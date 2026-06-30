# M2M100 Urdu → Roman Urdu — CPU / GPU Inference Server

`Mavkif/m2m100_rup_ur_to_rur` fine-tuned Urdu-to-Roman-Urdu model deployed with FastAPI.

```
m2m100-deploy/
├── server/
│   ├── app.py                # FastAPI inference server
│   ├── requirements.txt      # GPU / generic Python dependencies
│   └── requirements-cpu.txt  # Local CPU Python dependencies
├── scripts/
│   └── setup.sh          # One-shot server bootstrap
├── client/
│   └── client.py         # Translation client and sanity-check runner
├── Dockerfile            # Container alternative
└── README.md
```

## Local CPU Usage

To run the server locally on CPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r server/requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python server/app.py
```

> Use a dedicated Python virtual environment rather than a shared Conda/base environment to avoid package conflicts such as `pydantic`, `numpy`, and `protobuf`.
>
> The server defaults to `Mavkif/m2m100_rup_ur_to_rur` for Urdu-to-Roman-Urdu transliteration. This repo contains only fine-tuned model weights, so the tokenizer is loaded from `Mavkif/m2m100_rup_tokenizer_both` by default. If you use a different weights-only checkpoint, set `HF_MODEL` and `HF_TOKENIZER` appropriately.

Install the client dependency and run the test-pair client:

```bash
pip install -r client/requirements.txt
python client/client.py --url http://localhost:8000 --batch-size 4 --timeout 300
```

For local CPU usage, smaller request batches are recommended to avoid request timeouts and long inference waits.

To translate a single sentence instead of the built-in test set:

```bash
python client/client.py --url http://localhost:8000 --text "مجھے کھانا چاہیے"
```

---

---

## 1 — Kaggle Setup

### Prerequisites

1. Go to [dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup) and grab your free **authtoken**.
2. Create a new Kaggle notebook → **Settings → Accelerator → GPU T4 x2**.

### Steps

1. Upload `scripts/kaggle_setup.ipynb` to your Kaggle notebook (or copy-paste cells manually).
2. In **Cell 3**, paste your ngrok authtoken.
3. Run cells 1 → 5 in order.
4. Cell 4 prints your public URL, e.g. `https://abc123.ngrok-free.app` — use this as your `BASE_URL`.

**Expected startup time**: ~2–3 min (pip install + model download on first run; subsequent runs load from Kaggle's cache faster).

> Kaggle sessions last up to **12 hours**. The ngrok URL changes each session, so update your client's `BASE_URL` after each restart.

---

## 3 — API Reference

### `GET /health`

```bash
curl https://<NGROK_URL>/health
```

```json
{
  "status": "ok",
  "device": "cuda",
  "model": "Mavkif/m2m100_rup_ur_to_rur",
  "gpu_mem_mb": 1821.4
}
```

---

### `POST /translate`

**Single input:**

```bash
curl -X POST https://<NGROK_URL>/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "مجھے کھانا چاہیے"}'
```

```json
{
  "translation": "mujhe khana chahiye",
  "translations": null,
  "latency_ms": {
    "tokenize_ms": 2.1,
    "generate_ms": 38.6,
    "decode_ms": 0.4,
    "total_ms": 41.1
  }
}
```

**Batch input:**

```bash
curl -X POST https://<NGROK_URL>/translate \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "مجھے کھانا چاہیے",
      "آپ کہاں جا رہے ہیں",
      "پاکستان زندہ باد"
    ],
    "num_beams": 4
  }'
```

```json
{
  "translation": null,
  "translations": [
    "مجھے کھانا چاہیے",
    "آپ کہاں جا رہے ہیں",
    "پاکستان زندہ باد"
  ],
  "latency_ms": {
    "tokenize_ms": 3.2,
    "generate_ms": 52.8,
    "decode_ms": 0.8,
    "total_ms": 56.8
  }
}
```

**Request fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | — | Single input (mutually exclusive with `texts`) |
| `texts` | list[str] | — | Batch of up to 16 inputs |
| `num_beams` | int 1–10 | 4 | Beam search width |
| `max_new_tokens` | int 1–512 | 128 | Max output tokens |

---

## 4 — Python Client

```python
import httpx

BASE_URL = "https://<NGROK_URL>"

def transliterate(text: str, beams: int = 4) -> str:
    r = httpx.post(
        f"{BASE_URL}/translate",
        json={"text": text, "num_beams": beams},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["translation"]

def transliterate_batch(texts: list[str], beams: int = 4) -> list[str]:
    r = httpx.post(
        f"{BASE_URL}/translate",
        json={"texts": texts, "num_beams": beams},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()["translations"]

# Examples
print(transliterate("mujhe khana chahiye"))
# → مجھے کھانا چاہیے

print(transliterate_batch(["aap kaisy hain", "mera naam Ali hai"]))
# → ['آپ کیسے ہیں', 'میرا نام علی ہے']
```

---

## 5 — Benchmarks & Latency

Run the included benchmark script from your local machine:

```bash
pip install httpx
python client/benchmark.py --host https://<NGROK_URL> --runs 50
```

### Expected numbers on L40S GPU (FP16, `num_beams=4`)

| Metric | GPU (L40S) | CPU baseline |
|---|---|---|
| p50 single-request (wall) | ~45–65 ms | ~2–4 s |
| p99 single-request (wall) | ~80–100 ms | ~5–8 s |
| Server-side generate time | ~35–55 ms | ~1.8–3.5 s |
| Batch-8 total latency | ~80–120 ms | — |
| Batch-8 per-item amortised | ~10–15 ms | — |

> Numbers are estimates based on typical M2M100-418M performance at this model size
> on an L40S. Actual numbers depend on sequence length and system load.

---

## 6 — Generation Parameters & Trade-offs

### `num_beams` (key knob)

| Beams | Latency impact | Quality |
|---|---|---|
| 1 (greedy) | Fastest — ~20–30 ms generate | Acceptable for short inputs |
| 4 (default) | ~35–55 ms | Best quality/speed balance ✅ |
| 8 | ~70–100 ms | Marginal gain for transliteration |

For a transliteration task (constrained vocabulary, mostly 1:1 mapping), `num_beams=4` is the recommended default. Greedy (`num_beams=1`) works well for high-throughput pipelines where quality is slightly less critical.

### `max_new_tokens`

The model is tuned for sentence-level transliteration. In practice:
- Urdu output is typically 0.9–1.1× the token count of the Roman input.
- Default of 128 tokens covers sentences up to ~80 words.
- Very long paragraphs (>150 words) should be split at sentence boundaries before sending.

### Input length limits

- Hard limit in the server: 1024 characters per string.
- Tokenizer truncation: 256 tokens.
- Practical recommendation: keep inputs under 100 words / 600 characters per request for best quality.

### Out-of-domain behaviour

This model was fine-tuned specifically for Urdu → Roman Urdu transliteration. On out-of-domain inputs:
- **English words**: typically passed through or partially transliterated (unpredictable).
- **Numbers/dates**: usually passed through in Latin script.
- **Code-switched sentences** (Urdu + English): transliterates the Urdu words, English words may be garbled — pre-process to strip English tokens if needed.
- Pure Urdu Nastaliq input: this model is designed for Urdu input, so Urdu script is expected.

---

## 7 — Cost

Kaggle provides **30 free GPU hours/week** (T4). Sessions run up to 12 hours. No billing required.
