import pandas as pd
import re
from pathlib import Path
from typing import Optional, List

# ubah nilai string harga jadi number
def to_number(x: Optional[str]) -> Optional[int]:
    """ubah 'Rp 121.000' / '121k' / '121000' jadi int 121000"""
    if pd.isna(x):
        return None
    s = str(x).lower().strip()
    digits = re.sub(r"[^0-9]", "", s)
    return int(digits) if digits else None

# load semua sheet dari excel
def load_excel_all_sheets(path: str | Path) -> pd.DataFrame:
    """baca semua sheet dari 1 excel dan gabung jadi 1 dataframe"""
    path = Path(path)
    try:
        all_sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

    frames = []
    for sheet_name, df in all_sheets.items():
        df["sheet_source"] = sheet_name
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    harga_cols = [c for c in full_df.columns if "harga" in str(c).lower()]

    # Menggunakan apply untuk efisiensi
    full_df["harga_clean"] = full_df[harga_cols].apply(lambda row: next((to_number(row[hc]) for hc in harga_cols if pd.notna(row[hc])), None), axis=1)

    return full_df

# buat teks per baris untuk diembed
def build_docs_from_df(df: pd.DataFrame) -> List[str]:
    """buat teks per baris untuk di-embed."""
    docs = []
    for _, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        if pd.notna(row.get("harga_clean")):
            harga_num = int(row["harga_clean"])
            parts.append(f"Harga dalam angka: {harga_num}")
            k_val = int(round(harga_num / 1000))
            parts.append(f"Rentang harga: sekitar {k_val}k")
        docs.append("\n".join(parts))
    return docs

# untuk testing
if __name__ == "__main__":
    df = load_excel_all_sheets("data/KitabAudioFernanda.xlsx")
    docs = build_docs_from_df(df)
    print(df.head())
    print("docs:", docs[0][:200])
