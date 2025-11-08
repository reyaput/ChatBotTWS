import os
import time
import logging
from functools import lru_cache
from string import Template
from typing import List, Tuple

import numpy as np
import google.generativeai as genai
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load config model (opsional)
try:
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        EMBED_MODEL = cfg["models"]["embedding"]
        CHAT_MODEL = cfg["models"]["chat"]
except FileNotFoundError:
    EMBED_MODEL = "gemini-embedding-exp-03-07"
    CHAT_MODEL = "gemini-2.5-flash"

# set API key dari UI
def set_api_key(key: str):
    """dipanggil dari Streamlit sidebar."""
    key = (key or "").strip()
    if not key:
        raise ValueError("API key kosong.")
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)
    logger.info("Google API key set from UI.")

# helper untuk pastikan ada API key
def ensure_api_key():
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        # ini yang nanti ketangkep di app.py dan ditampilkan ke user
        raise RuntimeError("Google API Key belum diisi. Masukkan di sidebar dulu.")
    return key

# helper untuk retry dengan exponential backoff
def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))

# embed text
def embed_text(text: str, max_retries: int = 3) -> np.ndarray:
    ensure_api_key()

    def _call():
        return genai.embed_content(model=EMBED_MODEL, content=text)

    res = _with_retry(_call, max_retries=max_retries)
    return np.array(res["embedding"], dtype=np.float32)

# cached embed text
@lru_cache(maxsize=1000)
def cached_embed_text(text: str) -> np.ndarray:
    return embed_text(text)

# cosine similarity
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# retrieve top-k docs
def retrieve(query: str, docs: List[str], embs: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, int]]:
    ensure_api_key()
    q_emb = embed_text(query)
    sims = []
    for i, emb in enumerate(embs):
        sims.append((cosine_sim(q_emb, emb), i))
    sims.sort(reverse=True, key=lambda x: x[0])
    return [(docs[i], score, i) for score, i in sims[:top_k]]

# prompt template
PROMPT_TEMPLATE = Template(
    """
Kamu adalah asisten audio yang ramah. Kamu membantu pengguna memilih atau mengetahui produk TWS / earphone berdasarkan data katalog (jangan pernah menyebutkan kata "katalog").

PENTING:
- Jawab dengan bahasa yang sama seperti pertanyaan pengguna.
- Jangan membuat asumsi preferensi pengguna yang tidak ditulis (misal "kamu tidak suka open-ear").
- Jangan mengarang produk di luar data.
- Jika pengguna mau list produk, berikan maksimal 10 rekomendasi dengan highlight spesifikasi harga saja.

DATA KATALOG:
$context

PERTANYAAN PENGGUNA:
$query

TULISKAN JAWABAN:
""".strip()
)

def ask_gemini(context: str, query: str, max_retries: int = 3) -> str:
    ensure_api_key()
    model = genai.GenerativeModel(CHAT_MODEL)
    prompt = PROMPT_TEMPLATE.substitute(context=context, query=query)

    def _call():
        return model.generate_content(prompt)

    resp = _with_retry(_call, max_retries=max_retries)
    return resp.text.strip()

# chunking
def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks
