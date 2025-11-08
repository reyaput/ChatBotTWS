import os
import time
import logging
from functools import lru_cache
from string import Template
from typing import List, Tuple

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import yaml

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load .env
load_dotenv()
_initial_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

# kalo ada API key di .env, configure genai dari situ
if _initial_api_key:
    genai.configure(api_key=_initial_api_key)
else:
    logger.warning("GOOGLE_API_KEY tidak ditemukan di .env. Harus diisi dari UI.")

# load model config dari config.yaml
try:
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        EMBED_MODEL = cfg["models"]["embedding"]
        CHAT_MODEL = cfg["models"]["chat"]
except FileNotFoundError:
    EMBED_MODEL = "gemini-embedding-exp-03-07"
    CHAT_MODEL = "gemini-2.5-flash"

# API key management
def ensure_api_key():
    """Raise error kalau belum ada API key."""
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "API key Gemini belum di-set. Silakan isi di sidebar Streamlit terlebih dahulu."
        )
    return key

# ganti API key saat runtime
def set_api_key(key: str):
    """Ubah API key Gemini saat runtime dari Streamlit."""
    key = (key or "").strip()
    if not key:
        # kalau dikasih kosong ya jangan configure
        raise ValueError("API key kosong. Isi API key yang valid.")
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)
    logger.info("Google API key updated from Streamlit.")

# retry helper
def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"Gemini call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# embedding
def embed_text(text: str, max_retries: int = 3) -> np.ndarray:
    """Bikin embedding dari teks pakai Gemini dengan retry."""
    # pastikan ada API key
    ensure_api_key()

    def _call():
        return genai.embed_content(model=EMBED_MODEL, content=text)

    res = _with_retry(_call, max_retries=max_retries)
    return np.array(res["embedding"], dtype=np.float32)


@lru_cache(maxsize=1000)
def cached_embed_text(text: str) -> np.ndarray:
    return embed_text(text)

# similarity
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# retrieval
def retrieve(
    query: str,
    docs: List[str],
    embs: np.ndarray,
    top_k: int = 3,
) -> List[Tuple[str, float, int]]:
    """
    Ambil dokumen paling relevan berdasarkan cosine-similarity.
    Returns: [(doc_text, score, original_index), ...]
    """
    logger.info(f"[RAG] query: {query[:80]!r}")
    q_emb = embed_text(query)  # embed_text sudah cek API key
    sims: List[Tuple[float, int]] = []

    for i, emb in enumerate(embs):
        sim = cosine_sim(q_emb, emb)
        sims.append((sim, i))

    sims.sort(reverse=True, key=lambda x: x[0])
    top_items = sims[:top_k]
    results = [(docs[i], score, i) for score, i in top_items]
    return results

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

# chat dengan Gemini
def ask_gemini(context: str, query: str, max_retries: int = 3) -> str:
    """Panggil model chat Gemini dengan prompt yang sudah ditentukan."""
    # pastikan ada API key
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
