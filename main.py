"""
Minimal FastAPI RAG Backend - No LangChain Dependencies
Uses only: FastAPI, sentence-transformers, FAISS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import numpy as np
from pathlib import Path

# Minimal dependencies
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI(
    title="Sugarcane Variety RAG API",
    description="Chatbot API for Sugarcane Variety Information",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: str

class HealthCheck(BaseModel):
    status: str
    message: str

# Global variables
embeddings_model = None
index = None
documents = []
document_sources = []

def load_documents(folder_path: str):
    """Load all text files from folder"""
    docs = []
    sources = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return docs, sources
    
    for file_path in Path(folder_path).rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into chunks
                chunks = split_text(content, chunk_size=1000, overlap=200)
                docs.extend(chunks)
                sources.extend([str(file_path)] * len(chunks))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return docs, sources

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Simple text splitter"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def initialize_rag():
    """Initialize RAG system"""
    global embeddings_model, index, documents, document_sources
    
    print("🚀 Initializing RAG System...")
    
    # 1. Load embedding model
    print("🧠 Loading embedding model...")
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 2. Load documents
    print("📄 Loading documents...")
    documents, document_sources = load_documents("documents")
    
    if len(documents) == 0:
        print("⚠️  No documents found in documents/ folder")
        return False
    
    print(f"✅ Loaded {len(documents)} text chunks")
    
    # 3. Create embeddings
    print("💾 Creating embeddings...")
    embeddings = embeddings_model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # 4. Create FAISS index
    print("🔍 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, "faiss_index.bin")
    print("✅ RAG System initialized successfully!")
    
    return True

def search_documents(query: str, k: int = 3):
    """Search for relevant documents"""
    global embeddings_model, index, documents, document_sources
    
    if index is None or embeddings_model is None:
        return [], []
    
    # Encode query
    query_embedding = embeddings_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    # Get results
    results = []
    sources = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append(documents[idx])
            sources.append(document_sources[idx])
    
    return results, list(set(sources))

def generate_response(query: str):
    """Generate response from retrieved documents"""
    results, sources = search_documents(query, k=3)
    
    if len(results) == 0:
        return "No relevant information found. Please add documents to the documents/ folder.", []
    
    # Combine results
    context = "\n\n".join(results)
    
    # Simple response formatting
    response = f"""Based on the sugarcane variety information available:

{context[:1500]}

For more specific information about any variety mentioned, please ask a more targeted question."""
    
    return response, sources

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    success = initialize_rag()
    if not success:
        print("⚠️  RAG initialization incomplete")

@app.get("/", response_model=HealthCheck)
async def root():
    return {"status": "ok", "message": "Sugarcane Variety RAG API is running"}

@app.get("/health", response_model=HealthCheck)
async def health_check():
    if index is None:
        return {"status": "warning", "message": "RAG system not fully initialized"}
    return {"status": "healthy", "message": "All systems operational"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response, sources = generate_response(request.message)
        conversation_id = request.conversation_id or f"conv_{os.urandom(8).hex()}"
        
        return ChatResponse(
            response=response,
            sources=sources,
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/varieties")
async def get_varieties():
    varieties = [
        {
            "name": "CP77-400",
            "origin": "Louisiana, USA",
            "key_features": ["High sugar content", "Good ratooning", "Medium maturity"]
        },
        {
            "name": "CPF-237",
            "origin": "India (SBI)",
            "key_features": ["Very high sugar", "Drought tolerant", "Early maturity"]
        },
        {
            "name": "CPF-250",
            "origin": "India (SBI)",
            "key_features": ["Excellent disease resistance", "Stable performance"]
        },
        {
            "name": "CPF-251",
            "origin": "India (SBI)",
            "key_features": ["Early maturity", "Good sugar content"]
        },
        {
            "name": "CPF-253",
            "origin": "India (SBI)",
            "key_features": ["Highest sugar", "Superior disease resistance"]
        }
    ]
    return {"varieties": varieties}

@app.post("/query")
async def query_specific(variety: str, question: str):
    combined_query = f"Tell me about {variety} variety: {question}"
    
    try:
        response, sources = generate_response(combined_query)
        return {
            "variety": variety,
            "question": question,
            "answer": response,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)