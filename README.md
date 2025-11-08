# Chatbot RAG TWS (Streamlit + Gemini)

Proyek ini adalah chatbot sederhana yang bisa menjawab pertanyaan seputar TWS berdasarkan data Excel (punya Bang Fernanda) menggunakan konsep RAG (Retrieval Augmented Generation).

## Fitur
- Data TWS dari Fernanda Gunsan
- Support bahasa Indonesia & bahasa asing (jawab mengikuti bahasa user)
- Save embedding ke folder `storage/` supaya tidak embed ulang
- Sidebar untuk input Google API Key

## Model AI
- **Embedding**: `gemini-embedding-exp-03-07`
- **Chat**: `gemini-2.5-flash`

## Syarat
- Python **3.11** (disarankan)
- API key **Google Gemini**

## Cara jalanin dengan Conda

```bash
# 1. clone repo
git clone https://github.com/reyaput/ChatBotTWS.git
cd ChatBotTWS

# 2. buat environment
conda create --name chatbotTWS python=3.11
conda activate chatbotTWS

# 3. install dependency
pip install -r requirements.txt

# 4. jalankan streamlit
streamlit run app.py
```

## Cara jalanin tanpa Conda

```bash
# 1. clone repo
git clone https://github.com/reyaput/ChatBotTWS.git
cd ChatBotTWS

# 2. buat virtual env (opsional tapi disarankan)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. install dependency
pip install -r requirements.txt

# 4. jalanin
streamlit run app.py
```
