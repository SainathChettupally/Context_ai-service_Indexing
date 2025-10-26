import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from dotenv import load_dotenv
from pydantic import BaseModel
import openai

# --- Initialization ---

# Load environment variables from the root .env file
load_dotenv(dotenv_path='../.env')

# Initialize FastAPI app
app = FastAPI(
    title="AI Service",
    description="Handles vector search and indexing with Pinecone.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Client Initialization ---

# Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION = os.environ.get("PINECONE_REGION")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None

# OpenAI-compatible client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 768))

openai_client = None
if OPENAI_API_KEY and OPENAI_BASE_URL and EMBEDDING_MODEL_NAME:
    openai_client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    print(f"✅ OpenAI client initialized for model {EMBEDDING_MODEL_NAME} via {OPENAI_BASE_URL}")
else:
    print("⚠️ WARNING: OpenAI client environment variables are missing.")


# --- Pydantic Models ---
class IndexRequest(BaseModel):
    fileId: str
    content: str
    knowledgeBlockId: str

class SearchRequest(BaseModel):
    query: str
    knowledgeBlockId: str
    top_k: int = 5

class DeleteRequest(BaseModel):
    fileId: str

# --- Helper Functions ---
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not isinstance(text, str): return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# --- FastAPI Events ---
@app.on_event("startup")
def startup_event():
    if not pc:
        print("Pinecone client not initialized. Skipping index creation.")
        return

    print("Checking for Pinecone index...")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating it now...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec={
                    "pod": {
                        "environment": PINECONE_REGION
                    }
                }
            )
            print(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "AI Service is running"}

@app.post("/index-file")
async def index_file(request: IndexRequest):
    if not pc or not openai_client:
        raise HTTPException(status_code=503, detail="AI clients not initialized.")

    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        text_chunks = chunk_text(request.content)
        if not text_chunks:
            return {"message": "No text content to index."}

        embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text_chunks
        )
        embeddings = [item.embedding for item in embedding_response.data]

        vectors_to_upsert = [
            {
                "id": f"{request.fileId}-{i}",
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "fileId": request.fileId,
                    "knowledgeBlockId": request.knowledgeBlockId
                }
            } for i, (chunk, vector) in enumerate(zip(text_chunks, embeddings))
        ]
        
        index.upsert(vectors=vectors_to_upsert)
        return {"message": f"Successfully indexed {len(vectors_to_upsert)} chunks."}

    except Exception as e:
        print(f"Error during indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-context")
async def search_context(request: SearchRequest):
    if not pc or not openai_client:
        raise HTTPException(status_code=503, detail="AI clients not initialized.")

    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        print(f"--- RAG: New Search Request ---")
        print(f"[RAG] Query: '{request.query}'")
        print(f"[RAG] Knowledge Block ID: '{request.knowledgeBlockId}'")
        print(f"[RAG] Top K: {request.top_k}")

        # Create embedding for the query
        query_embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=request.query
        )
        query_vector = query_embedding_response.data[0].embedding
        print(f"[RAG] Successfully created query vector.")

        # Query Pinecone
        query_result = index.query(
            vector=query_vector,
            top_k=request.top_k,
            include_metadata=True,
            filter={"knowledgeBlockId": request.knowledgeBlockId}
        )
        print(f"[RAG] Pinecone response received. Found {len(query_result['matches'])} matches.")
        
        # Log the full details of each match
        for match in query_result['matches']:
            print(f"  - Match ID: {match['id']}, Score: {match['score']:.4f}")
            print(f"    Text: {match['metadata']['text'][:100]}...") # Log first 100 chars

        context = " ".join([match['metadata']['text'] for match in query_result['matches']])
        print(f"[RAG] Final context string prepared.")
        print("---------------------------------")
        return {"context": context}

    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-file")
async def delete_file(request: DeleteRequest):
    if not pc:
        raise HTTPException(status_code=503, detail="Pinecone client not initialized.")

    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(filter={"fileId": request.fileId})
        return {"message": f"Delete command issued for fileId: {request.fileId}"}

    except Exception as e:
        print(f"Error during deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Main execution block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)