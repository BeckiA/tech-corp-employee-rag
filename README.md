# TechCorp Employee RAG App

This project builds a vector database from a PDF document using LangChain, Chroma, and Google Generative AI embeddings. The script loads a PDF, splits it into chunks, creates embeddings, and stores them in a local Chroma database for retrieval-augmented generation workflows.

## What this project does

The current implementation in `rag_app.py`:

- loads a PDF file from the project directory
- splits the document into text chunks
- creates embeddings with `GoogleGenerativeAIEmbeddings`
- stores the embedded chunks in `./chroma_db` using Chroma

## Project structure

```text
my-rag-project/
├── rag_app.py
├── .env
├── .gitignore
├── TechCorp_Official_Employee_Handbook.pdf
├── chroma_db/
└── README.md
```

## Prerequisites

- Python 3.10+
- A Google API key with access to Gemini embeddings
- A PDF file to index

## Setup

1. Open a terminal in the project folder.
2. Create and activate a virtual environment if desired.
3. Install the required packages:

```bash
pip install python-dotenv langchain langchain-community langchain-text-splitters langchain-google-genai pypdf chromadb
```

4. Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

5. Make sure the PDF file exists in the project folder. The script currently expects:

```text
TechCorp_Official_Employee_Handbook.pdf
```

If your file has a different name, update the path in `rag_app.py`.

## Run the app

```bash
python rag_app.py
```

This will process the PDF and build the Chroma vector store in the `./chroma_db` directory.

## Notes

- The current script builds the knowledge base but does not yet add a query/retrieval interface.
- The vector database is persisted locally in `./chroma_db` and can be reused for later retrieval steps.
- If you want to extend this into a full chatbot or question-answering app, the next step is to add a retriever and LLM prompt chain.

## Environment variables

The script uses:

- `GOOGLE_API_KEY` for Google Gemini access

## Troubleshooting

- If the script fails to import packages, install the dependencies listed above.
- If embeddings fail, verify that your Google API key is valid and has access to the required generative AI services.
- If the PDF is not found, confirm that the file exists in the project root and matches the name in the script.
