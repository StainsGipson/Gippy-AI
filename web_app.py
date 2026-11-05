import os
import warnings
import sqlite3
import uuid
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pypdf import PdfReader
import docx2txt
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import streamlit as st

warnings.filterwarnings("ignore")
load_dotenv()

# --- 1. DATABASE SETUP (SQLite) ---
DB_FILE = "gippy_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create tables for Conversations and Messages
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  conversation_id TEXT, role TEXT, content TEXT, 
                  FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
    conn.commit()
    conn.close()

init_db()

def create_new_conversation():
    conv_id = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)", 
              (conv_id, "New Chat", datetime.now()))
    conn.commit()
    conn.close()
    return conv_id

def get_conversations():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def update_conversation_title(conv_id, title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    conn.commit()
    conn.close()

def save_message(conv_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", 
              (conv_id, role, content))
    conn.commit()
    conn.close()

def get_messages(conv_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]

# --- 2. CUSTOM BEAUTIFUL UI CSS ---
st.set_page_config(page_title="Gippy AI", page_icon="✨", layout="wide")

st.markdown("""
<style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f13 0%, #1a1a24 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #15151e;
        border-right: 1px solid #2a2a35;
    }
    
    /* Header */
    .header-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #7F7FD5, #86A8E7, #91EAE4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Input Box */
    .stChatInput {
        border-radius: 12px;
        border: 1px solid #3a3a4a;
        background-color: #1a1a24;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD DOCUMENTS & SETUP AGENT ---
@st.cache_resource
def setup_agent():
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

    docs = [Document(page_content=combined_text)] if combined_text else [Document(page_content="No documents loaded.")]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = np.array(embeddings_model.embed_documents([d.page_content for d in splits]))

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
        """Search the uploaded files."""
        return local_doc_store.search(question)

    @tool
    def get_current_time(timezone: str = "America/New_York") -> str:
        """Get exact current time."""
        try:
            return datetime.now(ZoneInfo(timezone)).strftime("%A, %B %d, %Y at %I:%M %p %Z")
        except: return "Error"

    web_search = TavilySearch(max_results=2)
    
    # Fast Model
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

    system_prompt = """You are Gippy AI, a highly intelligent, classic, and elegant AI assistant.
    CRITICAL RULES:
    1. **IDENTITY**: Always introduce yourself as Gippy AI. Never say ChatGPT.
    2. **TOOLS**: Use web search for current events. Use document search for uploaded files.
    3. **MEMORY**: Remember the user's name and previous context.
    """
    return create_react_agent(llm, [get_current_time, search_document, web_search], prompt=system_prompt)

with st.spinner("Initializing Gippy AI..."):
    agent = setup_agent()

# --- 4. SIDEBAR (CHAT MANAGEMENT) ---
with st.sidebar:
    st.markdown('<h2 style="color:#86A8E7;">✨ Gippy AI</h2>', unsafe_allow_html=True)
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = create_new_conversation()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🗂️ Past Conversations")
    
    conversations = get_conversations()
    for conv_id, title in conversations:
        # Show a button for each past chat
        if st.button(f"💬 {title[:25]}...", key=conv_id, use_container_width=True):
            st.session_state.current_chat_id = conv_id
            st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
st.markdown('<h1 class="header-title">Gippy AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#888;">Your elegant and intelligent assistant.</p>', unsafe_allow_html=True)

# Handle Session State for Chats
if "current_chat_id" not in st.session_state:
    conversations = get_conversations()
    if not conversations:
        st.session_state.current_chat_id = create_new_conversation()
    else:
        st.session_state.current_chat_id = conversations[0][0]

# Load messages for the current chat
messages = get_messages(st.session_state.current_chat_id)

# Display Messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask Gippy anything..."):
    
    # 1. If this is a new chat, set the title based on first message
    if len(messages) == 0:
        update_conversation_title(st.session_state.current_chat_id, prompt)

    # 2. Save User Message
    save_message(st.session_state.current_chat_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 3. Generate AI Response
    with st.chat_message("assistant"):
        def generate_response():
            chat_history = [(msg["role"], msg["content"]) for msg in messages]
            chat_history.append(("user", prompt))
            
            try:
                for event in agent.stream({"messages": chat_history}, stream_mode="messages"):
                    if event[0].content:
                        yield event[0].content
            except Exception as e:
                yield f"Error: {e}"

        response = st.write_stream(generate_response())
        
        # 4. Save AI Message
        save_message(st.session_state.current_chat_id, "assistant", response)