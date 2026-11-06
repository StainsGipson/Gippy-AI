import os
import warnings
import sqlite3
import uuid
import numpy as np
import io
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
import docx2txt

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")
load_dotenv()

app = FastAPI()

# --- DATABASE SETUP ---
DB_FILE = "gippy_history.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()
init_db()

# --- EMBEDDINGS & TOOLS ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
document_vectors = []
document_texts = []

@tool
def search_documents(question: str) -> str:
    """Search the uploaded documents for information."""
    if not document_vectors:
        return "No documents uploaded yet."
    query_vec = np.array(embeddings.embed_query(question))
    similarities = np.dot(document_vectors, query_vec) / (
        np.linalg.norm(document_vectors, axis=1) * np.linalg.norm(query_vec)
    )
    top_k = min(3, len(document_texts))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return "\n\n".join([document_texts[i] for i in top_indices])

@tool
def get_current_time(timezone: str) -> str:
    """Get exact current time."""
    try: 
        return datetime.now(ZoneInfo(timezone)).strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception as e: 
        return f"Error: {e}"

web_search = TavilySearch(max_results=2)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1) # Lower temp for strictness

system_prompt = """You are Gippy AI, an elite RAG assistant.
CRITICAL RULES:
1. **DOCUMENTS**: If the user asks ANYTHING about their uploaded files (PDF, DOCX), you MUST use the `search_documents` tool first. Do NOT ask the user to clarify. Search the document and give the answer based on what you find.
2. **CONTEXT**: If the user says "answers these" or "what is in the pdf", search the document immediately and summarize it.
3. **IDENTITY**: You are Gippy AI.
"""
agent = create_react_agent(llm, [get_current_time, search_documents, web_search], prompt=system_prompt)

# --- API ENDPOINTS ---
class ChatRequest(BaseModel):
    conv_id: str
    message: str

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global document_vectors, document_texts
    content = ""
    file_bytes = await file.read()
    try:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                if page.extract_text(): content += page.extract_text() + "\n"
        elif file.filename.endswith(".docx"):
            content = docx2txt.process(io.BytesIO(file_bytes))
        elif file.filename.endswith(".txt"):
            content = file_bytes.decode("utf-8")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(content)
        document_texts = chunks
        document_vectors = np.array(embeddings.embed_documents(chunks))
        return {"message": f"Successfully processed {file.filename}!"}
    except Exception as e:
        return {"message": f"Error processing file: {str(e)}"}

@app.get("/api/conversations")
def get_convs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]

@app.post("/api/new_chat")
def new_chat():
    conv_id = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)", (conv_id, "New Chat", datetime.now()))
    conn.commit()
    conn.close()
    return {"id": conv_id}

@app.get("/api/messages/{conv_id}")
def get_msgs(conv_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (request.conv_id, "user", request.message))
    c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (request.conv_id,))
    if c.fetchone()[0] == 1:
        c.execute("UPDATE conversations SET title = ? WHERE id = ?", (request.message[:30], request.conv_id))
    conn.commit()
    conn.close()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (request.conv_id,))
    history = c.fetchall()
    conn.close()

    chat_history = [(r, c) for r, c in history]

    def generate():
        full_response = ""
        try:
            for event in agent.stream({"messages": chat_history}, stream_mode="messages"):
                if event[0].content:
                    full_response += event[0].content
                    yield event[0].content
        except Exception as e:
            yield f"Error: {e}"
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (request.conv_id, "assistant", full_response))
        conn.commit()
        conn.close()

    return StreamingResponse(generate(), media_type="text/plain")

# --- PERFECT RESPONSIVE UI ---
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Gippy AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #09090b; --sidebar: #121215; --accent: #8b5cf6; --accent-glow: #a78bfa; --text: #f4f4f5; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
        
        /* Sidebar */
        .sidebar { width: 260px; background: var(--sidebar); border-right: 1px solid #27272a; display: flex; flex-direction: column; padding: 20px; flex-shrink: 0; transition: transform 0.3s; z-index: 100; }
        .sidebar h2 { font-weight: 600; background: linear-gradient(90deg, #8b5cf6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-size: 1.5rem; margin-bottom: 20px; }
        .new-chat-btn { background: var(--accent); color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: 600; margin-bottom: 20px; }
        .chat-list { flex: 1; overflow-y: auto; }
        .chat-item { padding: 10px; margin-bottom: 8px; border-radius: 6px; cursor: pointer; color: #a1a1aa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .chat-item:hover { background: #27272a; color: white; }
        
        /* Main Area */
        .main { flex: 1; display: flex; flex-direction: column; height: 100vh; width: 100%; }
        .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 12px 18px; border-radius: 12px; line-height: 1.6; font-size: 0.95rem; word-wrap: break-word; }
        .msg-user { align-self: flex-end; background: #27272a; border: 1px solid #3f3f46; }
        .msg-bot { align-self: flex-start; background: rgba(139, 92, 246, 0.05); border: 1px solid rgba(139, 92, 246, 0.2); }
        .msg-bot pre { background: #000; padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid #333; margin-top: 10px; }
        .msg-bot code { font-family: monospace; font-size: 0.9rem; }

        /* Input Area */
        .input-area { padding: 15px 20px; background: var(--bg); border-top: 1px solid #27272a; display: flex; flex-direction: column; gap: 10px; }
        .input-wrapper { display: flex; background: #18181b; border: 1px solid #3f3f46; border-radius: 12px; padding: 8px; align-items: center; }
        .input-wrapper:focus-within { border-color: var(--accent); }
        input[type="text"] { flex: 1; background: transparent; border: none; color: white; font-size: 1rem; outline: none; padding: 10px; }
        .send-btn { background: var(--accent); color: white; border: none; width: 40px; height: 40px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
        
        /* Upload Button */
        .upload-label { display: inline-flex; align-items: center; gap: 5px; background: #27272a; color: #a1a1aa; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; border: 1px solid #3f3f46; width: fit-content; }
        .upload-label:hover { background: #3f3f46; color: white; }
        #file-upload { display: none; }
        
        /* Mobile Toggle */
        .menu-toggle { display: none; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer; }
        
        /* Mobile Breakpoint */
        @media (max-width: 768px) {
            .sidebar { position: fixed; top: 0; bottom: 0; left: 0; transform: translateX(-100%); }
            .sidebar.open { transform: translateX(0); }
            .menu-toggle { display: block; position: absolute; top: 15px; left: 15px; z-index: 50; }
            .main { padding-top: 0; }
            .chat-container { padding-top: 60px; padding-bottom: 10px; }
            .message { max-width: 90%; }
            .input-area { padding: 10px; }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>

    <div class="sidebar" id="sidebar">
        <h2>✨ Gippy AI</h2>
        <button class="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
        <div class="chat-list" id="chat-list"></div>
    </div>

    <div class="main">
        <button class="menu-toggle" onclick="toggleSidebar()">☰</button>
        <div class="chat-container" id="chat-container">
            <div class="message msg-bot">Hello! I am Gippy AI. Upload a PDF or DOCX below, and ask me anything about it!</div>
        </div>
        <div class="input-area">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <label for="file-upload" class="upload-label">📎 Upload Document</label>
                <input type="file" id="file-upload" accept=".pdf,.docx,.txt" onchange="uploadFile()">
                <span id="upload-status" style="font-size: 0.8rem; color: #10b981;"></span>
            </div>
            <div class="input-wrapper">
                <input type="text" id="user-input" placeholder="Ask Gippy anything..." onkeypress="handleKeyPress(event)">
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </div>

    <script>
        let currentConvId = null;

        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('open');
        }

        async function loadConversations() {
            const res = await fetch('/api/conversations');
            const convs = await res.json();
            const list = document.getElementById('chat-list');
            list.innerHTML = '';
            convs.forEach(conv => {
                const div = document.createElement('div');
                div.className = 'chat-item';
                div.innerText = conv.title;
                div.onclick = () => {
                    loadChat(conv.id);
                    if (window.innerWidth <= 768) toggleSidebar(); 
                };
                list.appendChild(div);
            });
        }

        async function startNewChat() {
            const res = await fetch('/api/new_chat', { method: 'POST' });
            const data = await res.json();
            currentConvId = data.id;
            document.getElementById('chat-container').innerHTML = '';
            if (window.innerWidth <= 768) toggleSidebar(); 
            loadConversations();
        }

        async function loadChat(id) {
            currentConvId = id;
            const res = await fetch(`/api/messages/${id}`);
            const msgs = await res.json();
            const container = document.getElementById('chat-container');
            container.innerHTML = '';
            msgs.forEach(msg => addMessage(msg.role, msg.content));
        }

        async function uploadFile() {
            const fileInput = document.getElementById('file-upload');
            const file = fileInput.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append("file", file);
            document.getElementById('upload-status').innerText = "Embedding document...";
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('upload-status').innerText = data.message;
            setTimeout(() => { document.getElementById('upload-status').innerText = ""; }, 4000);
        }

        function addMessage(role, text) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${role === 'user' ? 'msg-user' : 'msg-bot'}`;
            div.innerHTML = marked.parse(text); 
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }

        function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text || !currentConvId) return;
            input.value = '';
            addMessage('user', text);
            const botDiv = addMessage('assistant', '');
            botDiv.style.opacity = '0.7';

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conv_id: currentConvId, message: text })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let fullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                fullText += chunk;
                botDiv.innerHTML = marked.parse(fullText);
                document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;
            }
            botDiv.style.opacity = '1';
            loadConversations(); 
        }

        async function init() {
            await loadConversations();
            const res = await fetch('/api/conversations');
            const convs = await res.json();
            if (convs.length > 0) { loadChat(convs[0].id); } else { startNewChat(); }
        }
        init();
    </script>
</body>
</html>
    """