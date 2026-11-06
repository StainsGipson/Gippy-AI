import os
import warnings
import sqlite3
import uuid
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.embeddings import FastEmbedEmbeddings
import docx2txt
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import sqlite3

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

# --- AI SETUP ---
combined_text = ""
for filename in os.listdir("."):
    if filename.lower().endswith((".pdf", ".docx", ".txt")):
        try:
            if filename.endswith(".pdf"):
                reader = PdfReader(filename)
                for page in reader.pages:
                    if page.extract_text(): combined_text += f"\n--- FILE: {filename} ---\n" + page.extract_text()
            elif filename.endswith(".docx"):
                combined_text += f"\n--- FILE: {filename} ---\n" + docx2txt.process(filename)
            elif filename.endswith(".txt"):
                with open(filename, "r", encoding="utf-8") as f:
                    combined_text += f"\n--- FILE: {filename} ---\n" + f.read()
        except: pass

docs = [Document(page_content=combined_text)] if combined_text else [Document(page_content="No docs.")]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectors = np.array(embeddings_model.embed_documents([d.page_content for d in splits]))
# Groq does not provide embeddings directly; you need to use a compatible API
# Simplest approach: use a free embedding API like Gemini or Jinavectors = np.array(embeddings_model.embed_documents([d.page_content for d in splits]))

class DocSearchTool:
    def __init__(self, vectors, docs, emb_model):
        self.vectors = vectors
        self.docs = docs
        self._emb = emb_model
    def search(self, query: str, k: int = 3) -> str:
        query_vec = np.array(self._emb.embed_query(query))
        similarities = np.dot(self.vectors, query_vec) / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return "\n\n".join([self.docs[i].page_content for i in top_k_indices])

local_doc_store = DocSearchTool(vectors, splits, embeddings_model)

@tool
def search_document(question: str) -> str:
    """Search uploaded files."""
    return local_doc_store.search(question)

@tool
def get_current_time(timezone: str) -> str:
    """Get exact current time. You must provide a valid IANA timezone name (e.g., 'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles')."""
    try: 
        return datetime.now(ZoneInfo(timezone)).strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception as e: 
        return f"Error: {e}"

web_search = TavilySearch(max_results=2)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

# STRICT PROMPT TO STOP CHEATING AND FIX FORMATTING
system_prompt = """You are Gippy AI, a highly intelligent, fast, and elegant assistant.
CRITICAL RULES:
1. **IDENTITY**: Always introduce yourself as Gippy AI. Never say ChatGPT.
2. **NO LINKS & NO CONTEXT**: Do not output raw URLs or links. Do NOT output a "Context" section at the end of your reply. Just give the final answer directly.
3. **TIMEZONES (ZERO CHEATING)**: If the user asks for the time in multiple cities (e.g., USA), you MUST call the `get_current_time` tool separately for EACH city. Do NOT calculate the math yourself. Do NOT output a markdown table. Use a clean bulleted list.
4. **TOOLS**: Use web search for current events. Use document search for uploaded files.
"""
agent = create_react_agent(llm, [get_current_time, search_document, web_search], prompt=system_prompt)

# --- API ENDPOINTS ---
class ChatRequest(BaseModel):
    conv_id: str
    message: str

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

# --- FRONTEND (FIXED UI) ---
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gippy AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #09090b;
            --sidebar: #121215;
            --accent: #8b5cf6;
            --accent-glow: #a78bfa;
            --text: #f4f4f5;
        }
        body {
            margin: 0; font-family: 'Inter', sans-serif;
            background: var(--bg); color: var(--text);
            display: flex; height: 100vh; overflow: hidden;
        }
        .sidebar {
            width: 260px; background: var(--sidebar);
            border-right: 1px solid #27272a;
            display: flex; flex-direction: column; padding: 20px;
            box-shadow: 2px 0 15px rgba(0,0,0,0.5);
        }
        .sidebar h2 {
            margin: 0 0 20px 0; font-weight: 600; letter-spacing: 1px;
            background: linear-gradient(90deg, #8b5cf6, #06b6d4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-align: center; font-size: 1.5rem;
        }
        .new-chat-btn {
            background: var(--accent); color: white; border: none;
            padding: 12px; border-radius: 8px; cursor: pointer;
            font-weight: 600; transition: 0.3s; margin-bottom: 20px;
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
        }
        .new-chat-btn:hover { background: var(--accent-glow); }
        .chat-list { flex: 1; overflow-y: auto; }
        .chat-item {
            padding: 10px; margin-bottom: 8px; border-radius: 6px;
            cursor: pointer; color: #a1a1aa; transition: 0.2s;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            border: 1px solid transparent;
        }
        .chat-item:hover { background: #27272a; color: white; border-color: #3f3f46; }
        
        .main {
            flex: 1; display: flex; flex-direction: column; position: relative;
        }
        .chat-container {
            flex: 1; overflow-y: auto; padding: 40px;
            display: flex; flex-direction: column; gap: 20px;
            scroll-behavior: smooth;
        }
        
        /* MESSAGE STYLING FIXED */
        .message {
            max-width: 75%; padding: 12px 18px; border-radius: 12px;
            line-height: 1.6; animation: fadeIn 0.3s ease-out;
            font-size: 0.95rem;
            word-wrap: break-word;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .msg-user {
            align-self: flex-end; background: #27272a; border: 1px solid #3f3f46;
        }
        .msg-bot {
            align-self: flex-start; background: rgba(139, 92, 246, 0.05);
            border: 1px solid rgba(139, 92, 246, 0.2);
        }
        .msg-bot pre {
            background: #000; padding: 15px; border-radius: 8px; overflow-x: auto;
            border: 1px solid #333;
        }
        .msg-bot code { font-family: monospace; font-size: 0.9rem; }
        
        .input-area {
            padding: 20px 40px; background: var(--bg);
            border-top: 1px solid #27272a;
            max-width: 800px; margin: 0 auto; width: 100%;
        }
        .input-wrapper {
            display: flex; background: #18181b; border: 1px solid #3f3f46;
            border-radius: 12px; padding: 10px; transition: 0.3s;
        }
        .input-wrapper:focus-within {
            border-color: var(--accent); box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
        }
        input {
            flex: 1; background: transparent; border: none; color: white;
            font-size: 1rem; outline: none; padding: 10px;
        }
        .send-btn {
            background: var(--accent); color: white; border: none;
            width: 45px; height: 45px; border-radius: 8px;
            cursor: pointer; display: flex; align-items: center; justify-content: center;
            transition: 0.2s; font-size: 1.2rem;
        }
        .send-btn:hover { background: var(--accent-glow); }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>

    <div class="sidebar">
        <h2>✨ Gippy AI</h2>
        <button class="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
        <div class="chat-list" id="chat-list"></div>
    </div>

    <div class="main">
        <div class="chat-container" id="chat-container"></div>
        <div class="input-area">
            <div class="input-wrapper">
                <input type="text" id="user-input" placeholder="Ask Gippy anything..." onkeypress="handleKeyPress(event)">
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </div>

    <script>
        let currentConvId = null;

        async function loadConversations() {
            const res = await fetch('/api/conversations');
            const convs = await res.json();
            const list = document.getElementById('chat-list');
            list.innerHTML = '';
            convs.forEach(conv => {
                const div = document.createElement('div');
                div.className = 'chat-item';
                div.innerText = conv.title;
                div.onclick = () => loadChat(conv.id);
                list.appendChild(div);
            });
        }

        async function startNewChat() {
            const res = await fetch('/api/new_chat', { method: 'POST' });
            const data = await res.json();
            currentConvId = data.id;
            document.getElementById('chat-container').innerHTML = '';
            loadConversations();
        }

        async function loadChat(id) {
            currentConvId = id;
            const res = await fetch(`/api/messages/${id}`);
            const msgs = await res.json();
            const container = document.getElementById('chat-container');
            container.innerHTML = '';
            msgs.forEach(msg => addMessage(msg.role, msg.content, false));
        }

        function addMessage(role, text, animate = true) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${role === 'user' ? 'msg-user' : 'msg-bot'}`;
            if(!animate) div.style.animation = 'none';
            // Use marked.js to render markdown (fixes code blocks)
            div.innerHTML = marked.parse(text); 
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

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
            if (convs.length > 0) {
                loadChat(convs[0].id);
            } else {
                startNewChat();
            }
        }
        init();
    </script>
</body>
</html>
    """