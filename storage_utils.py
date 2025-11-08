from pathlib import Path
import numpy as np
import pickle
import json
import os
from typing import Optional, Tuple, List

def save_embeddings(
    embeddings: np.ndarray,
    docs: List[str],
    excel_mtime: float,
    folder: str = "storage"
) -> None:
    """
    Simpan embeddings, docs, dan metadata secara atomik.
    - embeddings disimpan sebagai .npy
    - docs disimpan sebagai JSON
    - meta simpan mtime excel
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    # validasi sederhana
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if len(docs) != embeddings.shape[0]:
        raise ValueError(
            f"Jumlah docs ({len(docs)}) tidak cocok dengan jumlah embeddings ({embeddings.shape[0]})."
        )

    # simpan secara atomik dengan file .tmp lalu rename
    emb_tmp = folder_path / "embeddings.npy.tmp"
    emb_final = folder_path / "embeddings.npy"
    # penting: buka dulu, baru np.save, supaya numpy tidak menambah .npy lagi
    with open(emb_tmp, "wb") as f:
        np.save(f, embeddings)
    os.replace(emb_tmp, emb_final)

    # simpan
    docs_tmp = folder_path / "docs.json.tmp"
    docs_final = folder_path / "docs.json"
    with open(docs_tmp, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    os.replace(docs_tmp, docs_final)

    # hapus PKL lama kalau ada
    pkl_path = folder_path / "docs.pkl"
    if pkl_path.exists():
        try:
            pkl_path.unlink()
        except Exception:
            pass

    # simpan meta
    meta_tmp = folder_path / "meta.json.tmp"
    meta_final = folder_path / "meta.json"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump({"excel_mtime": excel_mtime}, f)
    os.replace(meta_tmp, meta_final)


def load_embeddings(folder: str = "storage") -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[float]]:
    """
    Load embeddings, docs, dan excel_mtime.
    Mengembalikan (None, None, None) kalau storage belum lengkap / rusak.
    """
    folder_path = Path(folder)
    emb_path = folder_path / "embeddings.npy"
    docs_json = folder_path / "docs.json"
    docs_pkl = folder_path / "docs.pkl"
    meta_path = folder_path / "meta.json"

    # cek kelengkapan
    if not emb_path.exists() or not meta_path.exists() or (not docs_json.exists() and not docs_pkl.exists()):
        return None, None, None

    # load embeddings
    try:
        with open(emb_path, "rb") as f:
            embeddings = np.load(f, allow_pickle=False)
    except Exception:
        return None, None, None

    # load docs
    docs: Optional[List[str]] = None
    if docs_json.exists():
        try:
            with open(docs_json, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception:
            return None, None, None
    else:
        # fallback legacy
        try:
            with open(docs_pkl, "rb") as f:
                docs = pickle.load(f)
        except Exception:
            return None, None, None

    # validasi panjang
    if docs is not None and embeddings is not None and len(docs) != embeddings.shape[0]:
        return None, None, None

    # load meta
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        excel_mtime = meta.get("excel_mtime")
    except Exception:
        return None, None, None

    return embeddings, docs, excel_mtime

# helper untuk mtime file
def get_file_mtime(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    return p.stat().st_mtime

# clear storage
def clear_storage(folder: str = "storage") -> None:
    """Hapus semua file di folder storage."""
    p = Path(folder)
    if not p.exists():
        return
    for f in p.iterdir():
        try:
            f.unlink()
        except Exception:
            pass
