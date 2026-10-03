import os
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"langchain-community.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Direct use of automatic function calling.*",
    category=UserWarning,
)

from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is available for LangChain Google GenAI
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

DEFAULT_PDF = "TechCorp_Official_Employee_Handbook.pdf"
CHROMA_DIR = "./chroma_db"

# Global state holders for vector database and retriever
vector_db = None
retriever = None
current_pdf_name = DEFAULT_PDF
kb_stats = {
    "pages": 0,
    "chunks": 0,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "status": "Initializing...",
}

# Define prompt template
TEMPLATE = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate.from_template(TEMPLATE)


def get_llm(model_name="gemini-3.5-flash", temperature=0.0):
    """Instantiate ChatGoogleGenerativeAI with fallback options."""
    try:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    except Exception:
        # Fallback to gemini-1.5-flash if gemini-3.5-flash is not available
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)


def initialize_or_reload_db(pdf_path=None, chunk_size=500, chunk_overlap=50, force_reindex=False, progress=gr.Progress()):
    """
    Builds or loads Chroma vector store with detailed progress & loading state feedback.
    Returns status message HTML and info markdown.
    """
    global vector_db, retriever, current_pdf_name, kb_stats

    target_pdf = pdf_path if pdf_path and os.path.exists(pdf_path) else DEFAULT_PDF
    current_pdf_name = os.path.basename(target_pdf)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # If already indexed and no forced re-indexing or new PDF, load from local disk
    if os.path.exists(CHROMA_DIR) and not force_reindex and vector_db is not None:
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        status_html = """
        <div class="status-pill status-ready">
            <span class="status-dot green"></span>
            <strong>Knowledge Base Active</strong> &bull; Local ChromaDB Loaded
        </div>
        """
        stats_md = f"**Document:** `{current_pdf_name}`  \n**Indexed Chunks:** `{kb_stats['chunks']}` &bull; **Pages:** `{kb_stats['pages']}`"
        return status_html, stats_md

    # Start multi-step ingestion loading state
    progress(0.1, desc="📄 Step 1/4: Reading PDF document...")
    time.sleep(0.3)
    
    if not os.path.exists(target_pdf):
        error_html = f"""
        <div class="status-pill status-error">
            <span class="status-dot red"></span>
            <strong>Error:</strong> PDF file standard path <code>{target_pdf}</code> not found!
        </div>
        """
        return error_html, "Please upload a valid PDF document."

    loader = PyPDFLoader(target_pdf)
    documents = loader.load()
    total_pages = len(documents)

    progress(0.35, desc=f"✂️ Step 2/4: Chunking {total_pages} pages into text passages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)

    progress(0.65, desc=f"⚡ Step 3/4: Computing embeddings for {total_chunks} chunks using Gemini...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    progress(1.0, desc="✅ Step 4/4: Knowledge Base Ready!")
    
    kb_stats = {
        "pages": total_pages,
        "chunks": total_chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "status": "Ready",
    }

    status_html = f"""
    <div class="status-pill status-ready">
        <span class="status-dot green"></span>
        <strong>Knowledge Base Ready</strong> &bull; {total_chunks} Chunks Indexed
    </div>
    """
    stats_md = f"**Document:** `{current_pdf_name}`  \n**Pages:** `{total_pages}` &bull; **Chunks:** `{total_chunks}`  \n**Chunk Size:** `{chunk_size}` (Overlap: `{chunk_overlap}`)"
    
    return status_html, stats_md


def ask_rag_question_stream(user_question, history, top_k=2, temperature=0.0):
    """
    Generator function that streams every single process step as a live loading state to the user.
    """
    global vector_db, retriever

    if not user_question or not user_question.strip():
        yield history, "", render_process_box("idle"), "*No context query executed yet.*"
        return

    # Add user and assistant messages in dict format for Gradio 6
    current_history = history.copy() if history else []
    current_history.append({"role": "user", "content": user_question})
    current_history.append({"role": "assistant", "content": "⏳ *Searching vector database...*"})

    # --- STEP 1: RETRIEVAL LOADING STATE ---
    step1_box = render_process_box(
        state="step1",
        step1_text=f"Searching ChromaDB Vector Store for top-{int(top_k)} matching chunks...",
        step2_text="Extracting Context & Source Passages",
        step3_text="Generating AI Answer with Gemini 3.5 Flash"
    )
    context_loading_md = "🔄 **Retrieval in progress...** Querying embedding index."
    
    yield current_history, "", step1_box, context_loading_md
    time.sleep(0.4)  # Visual pause to make the state transition clear to the user

    # Execute vector store retrieval
    if retriever is None or vector_db is None:
        initialize_or_reload_db()
    
    custom_retriever = vector_db.as_retriever(search_kwargs={"k": int(top_k)})
    retrieved_docs = custom_retriever.invoke(user_question)

    # Format retrieved contexts
    formatted_context_list = []
    inspector_md_items = []
    
    for idx, doc in enumerate(retrieved_docs, 1):
        page_num = doc.metadata.get("page", "N/A")
        if isinstance(page_num, int):
            page_num += 1  # 1-indexed for display
        content_text = doc.page_content.strip()
        formatted_context_list.append(content_text)
        
        inspector_md_items.append(
            f"### 📄 Source Passage #{idx} (Page {page_num})\n"
            f"```text\n{content_text}\n```"
        )
    
    context_text = "\n\n".join(formatted_context_list)
    context_inspector_md = "\n\n---\n\n".join(inspector_md_items) if inspector_md_items else "*No matching passages found.*"

    # --- STEP 2: CONTEXT EXTRACTED LOADING STATE ---
    current_history[-1] = {"role": "assistant", "content": "⏳ *Context extracted. Preparing AI prompt...*"}
    step2_box = render_process_box(
        state="step2",
        step1_text=f"Found {len(retrieved_docs)} relevant context chunks",
        step2_text=f"Context extracted & formatted ({len(context_text)} characters)",
        step3_text="Sending query & context to Gemini AI model..."
    )
    
    yield current_history, "", step2_box, context_inspector_md
    time.sleep(0.4)

    # --- STEP 3: LLM SYNTHESIS LOADING STATE ---
    current_history[-1] = {"role": "assistant", "content": "🧠 *Gemini 3.5 Flash is synthesizing answer...*"}
    step3_box = render_process_box(
        state="step3",
        step1_text=f"Vector Search Complete ({len(retrieved_docs)} passages)",
        step2_text="Context Formatted & Ready",
        step3_text="Gemini 3.5 Flash is synthesizing final answer..."
    )
    
    yield current_history, "", step3_box, context_inspector_md
    time.sleep(0.2)

    # Invoke LLM model
    llm = get_llm(model_name="gemini-3.5-flash", temperature=float(temperature))
    prompt_value = PROMPT.format(context=context_text, question=user_question)
    
    try:
        response = llm.invoke(prompt_value)
        if isinstance(response.content, list):
            clean_answer = response.content[0].get("text", "")
        else:
            clean_answer = str(response.content)
    except Exception as e:
        clean_answer = f"⚠️ An error occurred while contacting Gemini AI: {str(e)}"

    # --- STEP 4: PROCESS COMPLETE STATE ---
    final_box = render_process_box(
        state="complete",
        step1_text=f"Vector Search: {len(retrieved_docs)} passages match",
        step2_text=f"Context Inspector updated",
        step3_text="Answer synthesized successfully"
    )

    # Update final chatbot message with assistant response
    current_history[-1] = {"role": "assistant", "content": clean_answer}

    yield current_history, "", final_box, context_inspector_md


def render_process_box(state="idle", step1_text="", step2_text="", step3_text=""):
    """Renders HTML for the live step-by-step process status indicator."""
    if state == "idle":
        return """
        <div class="process-container idle">
            <div class="process-header">
                <span class="pulse-dot grey"></span>
                <span class="process-title">RAG Pipeline Monitor &bull; Standby</span>
            </div>
            <div class="process-body">
                <div class="step-item"><span class="icon">🔍</span> <span>1. Vector Search (ChromaDB)</span></div>
                <div class="step-item"><span class="icon">📄</span> <span>2. Context Formatting</span></div>
                <div class="step-item"><span class="icon">🧠</span> <span>3. AI Answer Generation</span></div>
            </div>
        </div>
        """
    
    s1_class = "done" if state in ["step2", "step3", "complete"] else ("active" if state == "step1" else "pending")
    s2_class = "done" if state in ["step3", "complete"] else ("active" if state == "step2" else "pending")
    s3_class = "done" if state == "complete" else ("active" if state == "step3" else "pending")

    s1_icon = "✅" if state in ["step2", "step3", "complete"] else '<span class="spinner"></span>'
    s2_icon = "✅" if state in ["step3", "complete"] else ('<span class="spinner"></span>' if state == "step2" else "⏳")
    s3_icon = "✨" if state == "complete" else ('<span class="spinner"></span>' if state == "step3" else "⏳")

    return f"""
    <div class="process-container active">
        <div class="process-header">
            <span class="pulse-dot blue"></span>
            <span class="process-title">Live RAG Execution Progress</span>
        </div>
        <div class="process-body">
            <div class="step-item {s1_class}">
                <span class="icon">{s1_icon}</span>
                <div class="step-details">
                    <span class="step-name">Step 1: Context Retrieval</span>
                    <span class="step-desc">{step1_text}</span>
                </div>
            </div>
            <div class="step-item {s2_class}">
                <span class="icon">{s2_icon}</span>
                <div class="step-details">
                    <span class="step-name">Step 2: Context Parsing</span>
                    <span class="step-desc">{step2_text}</span>
                </div>
            </div>
            <div class="step-item {s3_class}">
                <span class="icon">{s3_icon}</span>
                <div class="step-details">
                    <span class="step-name">Step 3: Gemini Synthesis</span>
                    <span class="step-desc">{step3_text}</span>
                </div>
            </div>
        </div>
    </div>
    """


def handle_sample_click(question_text, history, top_k, temperature):
    """Helper event handler when a user clicks any quick sample question button."""
    for h, u, box, ctx in ask_rag_question_stream(question_text, history, top_k, temperature):
        yield h, u, box, ctx


# Custom CSS for modern aesthetic design
CUSTOM_CSS = """
/* Theme overrides and global font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(17, 24, 39, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
}

/* Header Banner */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}

.header-title-box h1 {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
}

.badge-group {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-top: 10px;
}

.model-pill {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.4);
    color: #c7d2fe;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* Status Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 24px;
    font-size: 13px;
    font-weight: 500;
}

.status-ready {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}

.status-error {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.status-dot.green { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
.status-dot.red { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }

/* Process Container Component */
.process-container {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.process-container.active {
    border-color: #6366f1;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
}

.process-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.process-title {
    font-size: 14px;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.2px;
}

.pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
}
.pulse-dot.blue { background-color: #38bdf8; box-shadow: 0 0 10px #38bdf8; animation: pulse 1.5s infinite; }
.pulse-dot.grey { background-color: #64748b; }

@keyframes pulse {
    0% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.15); }
    100% { opacity: 0.4; transform: scale(0.9); }
}

.process-body {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid transparent;
    font-size: 13px;
    transition: all 0.2s ease;
}

.step-item.pending { opacity: 0.5; }
.step-item.active {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(129, 140, 248, 0.3);
    opacity: 1;
}
.step-item.done {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.2);
    opacity: 1;
}

.step-details {
    display: flex;
    flex-direction: column;
}

.step-name {
    font-weight: 600;
    color: #f3f4f6;
}

.step-desc {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 2px;
}

/* Spinner Animation */
.spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    border-top-color: #818cf8;
    animation: spin 0.8s linear infinite;
    margin-top: 2px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Quick Question Buttons */
.sample-btn button {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    font-size: 12px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}

.sample-btn button:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: rgba(129, 140, 248, 0.5) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

/* Chatbot customization */
.gradio-container .message-row {
    margin-bottom: 12px !important;
}
"""

# Build Gradio Blocks Interface
def create_gradio_app():
    with gr.Blocks(title="TechCorp RAG Assistant") as app:
        
        # Header Section
        gr.HTML("""
        <div class="header-banner">
            <div class="header-title-box">
                <h1>📚 TechCorp AI Handbook & RAG Assistant</h1>
                <p class="header-subtitle">Interactive PDF Retrieval-Augmented Generation system with live process visibility</p>
            </div>
            <div class="badge-group">
                <span class="model-pill">🤖 Gemini 3.5 Flash</span>
                <span class="model-pill">⚡ ChromaDB Vector Search</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            
            # LEFT SIDEBAR: Document Control & Settings
            with gr.Column(scale=1):
                
                # Knowledge Base Status Box
                with gr.Group(elem_classes=["glass-card"]):
                    gr.Markdown("### 📂 Knowledge Base Management")
                    kb_status_output = gr.HTML(
                        """
                        <div class="status-pill status-ready">
                            <span class="status-dot green"></span>
                            <strong>Knowledge Base Loaded</strong> &bull; Ready
                        </div>
                        """
                    )
                    kb_stats_markdown = gr.Markdown(
                        f"**Document:** `{DEFAULT_PDF}`  \n**Status:** Vector Database ready."
                    )
                    
                    pdf_file_input = gr.File(
                        label="Upload Custom PDF Document",
                        file_types=[".pdf"],
                        type="filepath",
                        elem_id="pdf-uploader"
                    )
                    
                    reindex_btn = gr.Button(
                        "⚡ Index / Reload Knowledge Base",
                        variant="primary",
                        size="sm"
                    )

                # RAG Settings Accordion
                with gr.Accordion("⚙️ RAG Hyperparameters", open=False):
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        label="Top-K Retrieved Contexts",
                        info="Number of relevant PDF chunks to feed the LLM"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        label="Model Temperature",
                        info="0.0 for factual responses, higher for creative responses"
                    )
                    chunk_size_input = gr.Number(
                        value=500,
                        label="Chunk Size (characters)",
                        precision=0
                    )
                    chunk_overlap_input = gr.Number(
                        value=50,
                        label="Chunk Overlap (characters)",
                        precision=0
                    )

                # Suggested Questions Card
                with gr.Group(elem_classes=["glass-card"]):
                    gr.Markdown("### 💡 Quick Sample Questions")
                    gr.Markdown("<span style='font-size: 12px; color: #94a3b8;'>Click any prompt below to test live RAG execution:</span>")
                    
                    sample_q1 = gr.Button("🏢 What is the policy on remote work and flexible hours?", elem_classes=["sample-btn"])
                    sample_q2 = gr.Button("🏖️ How many PTO days and paid company holidays are provided?", elem_classes=["sample-btn"])
                    sample_q3 = gr.Button("🎁 What employee health, dental, and retirement benefits are offered?", elem_classes=["sample-btn"])
                    sample_q4 = gr.Button("🔒 What are the rules regarding data privacy and code of conduct?", elem_classes=["sample-btn"])

            # RIGHT PANEL: Live Execution Monitor & Chatbot
            with gr.Column(scale=2):
                
                # Live Step-by-step Execution Status Monitor
                process_monitor_html = gr.HTML(
                    value=render_process_box("idle"),
                    elem_id="process-monitor"
                )

                # Interactive Chatbot Window
                chatbot = gr.Chatbot(
                    label="Employee Handbook Assistant",
                    height=450,
                    avatar_images=(None, "https://api.iconify.design/lucide:bot.svg?color=%23818cf8")
                )

                # User Input Row
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask anything about TechCorp policies, benefits, or guidelines...",
                        label="Your Question",
                        scale=4,
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button("Send Question 🚀", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", scale=1)

                # Context Inspector Accordion
                with gr.Accordion("🔍 Context Inspector (Retrieved PDF Passages)", open=True):
                    context_inspector_markdown = gr.Markdown(
                        "*When you ask a question, the retrieved text passages from the Chroma vector database will be displayed here in real-time.*"
                    )

        # Event Wireups
        
        # Initial load callback
        app.load(
            fn=initialize_or_reload_db,
            inputs=[],
            outputs=[kb_status_output, kb_stats_markdown]
        )

        # Re-index Document Callback
        reindex_btn.click(
            fn=lambda pdf, c_size, c_overlap: initialize_or_reload_db(pdf, c_size, c_overlap, force_reindex=True),
            inputs=[pdf_file_input, chunk_size_input, chunk_overlap_input],
            outputs=[kb_status_output, kb_stats_markdown]
        )

        # Submit Question Callback (Streamed loading states)
        submit_event = submit_btn.click(
            fn=ask_rag_question_stream,
            inputs=[msg_input, chatbot, top_k_slider, temp_slider],
            outputs=[chatbot, msg_input, process_monitor_html, context_inspector_markdown]
        )
        
        msg_input.submit(
            fn=ask_rag_question_stream,
            inputs=[msg_input, chatbot, top_k_slider, temp_slider],
            outputs=[chatbot, msg_input, process_monitor_html, context_inspector_markdown]
        )

        # Clear Chat Callback
        clear_btn.click(
            fn=lambda: ([], render_process_box("idle"), "*Context cleared.*"),
            inputs=[],
            outputs=[chatbot, process_monitor_html, context_inspector_markdown]
        )

        # Quick Sample Question Callbacks
        for sample_btn, text in [
            (sample_q1, "What is the policy on remote work and flexible hours?"),
            (sample_q2, "How many PTO days and paid company holidays are provided?"),
            (sample_q3, "What employee health, dental, and retirement benefits are offered?"),
            (sample_q4, "What are the rules regarding data privacy and code of conduct?")
        ]:
            sample_btn.click(
                fn=handle_sample_click,
                inputs=[gr.State(text), chatbot, top_k_slider, temp_slider],
                outputs=[chatbot, msg_input, process_monitor_html, context_inspector_markdown]
            )

    return app


if __name__ == "__main__":
    print("\n--- Initializing TechCorp RAG Interactive UI ---")
    
    # Pre-build vector database on startup
    initialize_or_reload_db()
    
    # Launch Gradio app
    app = create_gradio_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS
    )