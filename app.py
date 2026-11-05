import os
import warnings
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

warnings.filterwarnings("ignore")
load_dotenv()

print("Scanning folder for documents...")

def load_all_documents(folder_path="."):
    combined_text = ""
    found_files = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    if page.extract_text():
                        combined_text += f"\n--- FILE: {filename} ---\n" + page.extract_text()
                found_files.append(filename)
            except: pass
                
        elif filename.lower().endswith(".docx"):
            try:
                # We keep the raw text, NO paragraph numbers
                combined_text += f"\n--- FILE: {filename} ---\n" + docx2txt.process(file_path)
                found_files.append(filename)
            except: pass
                
        elif filename.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_text += f"\n--- FILE: {filename} ---\n" + f.read()
                found_files.append(filename)
            except: pass
                
    if found_files:
        print(f"✅ Loaded: {', '.join(found_files)}")
        
    return combined_text

document_text = load_all_documents()

# Create chunks
docs = [Document(page_content=document_text)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = np.array(embeddings_model.embed_documents([d.page_content for d in splits]))

class DocSearchTool:
    def __init__(self, vectors, docs, emb_model):
        self.vectors = vectors
        self.docs = docs
        self._emb = emb_model

    def search(self, query: str, k: int = 4) -> str:
        query_vec = np.array(self._emb.embed_query(query))
        similarities = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return "\n\n".join([self.docs[i].page_content for i in top_k_indices])

local_doc_store = DocSearchTool(vectors, splits, embeddings_model)

@tool
def search_document(question: str) -> str:
    """Use this tool when the user asks a question about the uploaded files/documents."""
    return local_doc_store.search(question)

@tool
def get_current_time(timezone: str = "America/New_York") -> str:
    """Get the exact current date and time for a specific timezone."""
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception as e:
        return f"Error getting time: {e}"

web_search = TavilySearch(max_results=3)

# FAST MODEL: Switched to Llama-3.3-70b for instant, smart answers
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# THE SMART PROMPT


system_prompt = """You are a highly intelligent and fast AI assistant.

CRITICAL RULES:
1. **FILE SOURCES**: The context contains text tagged with `--- FILE: filename ---`. 
2. **TARGETED ANSWERS**: If the user asks about a specific file, ONLY read the text under that file's tag.
3. **INTERPRETATION**: If the user asks for "q2", use your intelligence to find the 2nd question/topic in that file.
4. **NO HALLUCINATION**: If the answer isn't in the file, say so.
5. **REAL-TIME FACTS**: If the user asks about **current world leaders, presidents, prime ministers, or recent news**, you MUST use `tavily_search_results_json` to get the latest information. DO NOT answer from your internal memory. The current date is November 5, 2026.
6. **MEMORY**: Always remember the user's name and context.
"""

agent = create_react_agent(llm, [get_current_time, search_document, web_search], prompt=system_prompt)

if __name__ == "__main__":
    print("🤖 Fast & Smart Agent is ready! Type 'quit' to exit.")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break
        
        chat_history.append(("user", user_input))
        
        try:
            response = agent.invoke({"messages": chat_history})
            bot_reply = response['messages'][-1].content
            chat_history.append(("assistant", bot_reply))
            print(f"Bot: {bot_reply}")
        except Exception as e:
            print(f"Error: {e}")
            chat_history.pop()