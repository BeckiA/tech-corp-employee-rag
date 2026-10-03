# TechCorp Employee RAG App with Interactive Gradio UI

This project builds a full-featured Retrieval-Augmented Generation (RAG) system with a modern, interactive Gradio web interface. It processes PDF documents, splits text into chunks, computes vector embeddings using Google Generative AI, stores them in ChromaDB, and synthesizes concise factual answers using Gemini 3.5 Flash.

## ✨ Features & UI Highlights

- **Interactive Web Interface**: Built with Gradio featuring a glassmorphism dark theme.
- **Live Process Loading States**: Every stage of document ingestion and RAG query execution is visualized in real-time with step-by-step progress spinners and indicators:
  1. 🔍 **Step 1: Vector Search** (Querying ChromaDB for relevant chunks)
  2. 📄 **Step 2: Context Parsing** (Formatting source passages & metadata)
  3. 🧠 **Step 3: Gemini AI Synthesis** (Generating concise response)
- **Document Management**: Upload any custom PDF or use the included `TechCorp_Official_Employee_Handbook.pdf`.
- **Context Inspector**: Expandable panel showing exact source text passages retrieved from the PDF for transparency.
- **Hyperparameter Controls**: Adjust Top-K retrieved passages, temperature, chunk size, and chunk overlap on the fly.
- **Quick Sample Prompts**: One-click preset buttons for common employee handbook queries (remote work, PTO, health benefits, data privacy).

## 📁 Project Structure

```text
my-rag-project/
├── rag_app.py                            # Main application with Gradio UI & RAG pipeline
├── .env                                  # API key configuration
├── .gitignore
├── TechCorp_Official_Employee_Handbook.pdf # Default PDF knowledge base
├── chroma_db/                            # Local Chroma vector database
└── README.md
```

## 🚀 How to Run using Virtual Environment (`venv`)

### Option 1: Direct Execution (Quickest & Works Anywhere)

Run the script directly using the virtual environment's Python binary without activating:

- **PowerShell / Command Prompt:**
  ```powershell
  .\venv\Scripts\python.exe rag_app.py
  ```

- **Git Bash:**
  ```bash
  ./venv/Scripts/python.exe rag_app.py
  ```

---

### Option 2: Activate `venv` First, Then Run

Activate the environment depending on your terminal type:

#### 1️⃣ PowerShell
```powershell
.\venv\Scripts\Activate.ps1
python rag_app.py
```
*(If PowerShell blocks script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).*

#### 2️⃣ Command Prompt (cmd.exe)
```cmd
venv\Scripts\activate.bat
python rag_app.py
```

#### 3️⃣ Git Bash
```bash
source venv/Scripts/activate
python rag_app.py
```

---

## ⚙️ Initial Setup & Prerequisites

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)

### 2. Install Dependencies into `venv`
```bash
.\venv\Scripts\pip.exe install python-dotenv langchain langchain-community langchain-text-splitters langchain-google-genai pypdf chromadb gradio
```

### 3. Configure API Key
Create or edit `.env` in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Access the Web App
Once running, open your web browser at:
`http://127.0.0.1:7860`

## ⚙️ Environment Variables

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`: Required for Google Gemini embeddings and LLM inference.
