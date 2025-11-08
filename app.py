import os
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Optional, Tuple, List

from data_loader import load_excel_all_sheets, build_docs_from_df
from rag_core import embed_text, retrieve, ask_gemini, set_api_key
from storage_utils import load_embeddings, save_embeddings, get_file_mtime, clear_storage

# variabel global
EXCEL_PATH = Path("data/KitabAudioFernanda.xlsx")
STORAGE_DIR = Path("storage")
MAX_HISTORY = 6
TOP_K_RESULTS = 3

# ekstrak rentang harga dari query
def extract_price_range(query: str) -> Tuple[Optional[int], Optional[int]]:
    q = query.lower()
    pattern = r"(\d+[.,]?\d*)\s*(k|rb|ribu)?"
    nums = re.findall(pattern, q)

    cleaned = []
    for num, suffix in nums:
        num = num.replace(".", "").replace(",", "")
        val = int(num)
        if suffix in ["k", "rb", "ribu"]:
            val = val * 1000
        cleaned.append(val)

    if not cleaned:
        return None, None

    if "sampai" in q or "-" in q:
        low = min(cleaned)
        high = max(cleaned)
        return low, high

    if "maks" in q or "maksimal" in q or "<=" in q:
        return None, cleaned[0]

    target = cleaned[0]
    low = int(target * 0.8)
    high = int(target * 1.2)
    return low, high

# sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_default = st.session_state.get("GOOGLE_API_KEY", "")
    api_input = st.text_input("Google API Key", value=api_default, type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simpan key"):
            try:
                set_api_key(api_input)
                st.session_state["GOOGLE_API_KEY"] = api_input
                st.success("API key disimpan.")
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("🔄 Reset semua"):
            clear_storage("storage")
            st.session_state.clear()
            st.cache_data.clear()
            st.rerun()

# load data dan build docs
@st.cache_data
def load_data_and_docs() -> Tuple[pd.DataFrame, List[str]]:
    try:
        df = load_excel_all_sheets(EXCEL_PATH)
        docs = build_docs_from_df(df)
        return df, docs
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), []

df, docs = load_data_and_docs()

# cek data
def get_embeddings(docs: List[str]) -> np.ndarray:
    """Get or create embeddings for documents."""
    # kalau belum ada API key di session/env, hentikan di sini
    if "GOOGLE_API_KEY" not in st.session_state and not (os_key := (Path(".env").exists())):
        # os_key itu cuma placeholder, tetep kita stop di bawah karena rag_core ngecek lagi
        raise RuntimeError("API key belum diisi. Masukkan dulu di sidebar.")

    excel_mtime = get_file_mtime(EXCEL_PATH)
    embs, saved_docs, saved_mtime = load_embeddings(str(STORAGE_DIR))

    if (
        embs is not None
        and saved_docs is not None
        and saved_mtime == excel_mtime
        and len(saved_docs) == len(docs)
    ):
        return embs

    st.warning("Creating new embeddings...")
    all_vecs = [embed_text(d) for d in docs]
    embs = np.vstack(all_vecs)
    save_embeddings(embs, docs, excel_mtime, str(STORAGE_DIR))
    st.success("New embeddings saved.")
    return embs

# coba dapatkan embeddings
try:
    embs = get_embeddings(docs)
except RuntimeError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error processing embeddings: {e}")
    st.stop()

# session state untuk chat
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Halo 👋 aku chatbot khusus TWS dari data Fernanda Gunsan.",
            }
        ]

init_session_state()

# chat interface
st.title("Chatbot TWS")

user_input = st.chat_input("Ketik pertanyaan kamu di sini...")

if user_input:
    # simpan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})

    low, high = extract_price_range(user_input)

    current_docs = docs
    current_embs = embs
    filtered_info = ""

    if low is not None or high is not None:
        mask = df["harga_clean"].notna()
        if low is not None:
            mask = mask & (df["harga_clean"] >= low)
        if high is not None:
            mask = mask & (df["harga_clean"] <= high)

        sub_df = df[mask].reset_index()
        if len(sub_df) > 0:
            idxs = sub_df["index"].tolist()
            current_docs = [docs[i] for i in idxs]
            current_embs = embs[idxs, :]
            filtered_info = f"(difilter harga: {low or '-'} s/d {high or '-'}, ketemu {len(sub_df)} baris)\n"
        else:
            current_docs = docs
            current_embs = embs

    # retrieve
    results = retrieve(user_input, current_docs, current_embs, top_k=TOP_K_RESULTS)

    if len(results) == 0:
        answer = "Aku nggak nemu data yang cocok di Excel kamu untuk pertanyaan itu."
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        context_parts = []
        for ctx, score, _ in results:
            context_parts.append(ctx)
        context_joined = "\n\n---\n\n".join(context_parts)

        history_text = ""
        for m in st.session_state.messages[-MAX_HISTORY:]:
            history_text += f"{m['role']}: {m['content']}\n"

        final_context = f"""
        {filtered_info}
        Riwayat percakapan:
        {history_text}

        Konteks dari Excel:
        {context_joined}
        """.strip()

        try:
            answer = ask_gemini(final_context, user_input)
        except RuntimeError as e:
            # kalau tiba-tiba API hilang waktu chat
            answer = f"⚠️ {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})

# tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
