import os
import re
import sys
import glob
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 0. Setup and Initialization ---

# Load environment variables
load_dotenv()

# Check for API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("FATAL: GROQ_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)

# FastAPI App initialization
app = FastAPI(title="YouTube RAG API")

# Cache for vector stores (key: video_id, value: FAISS index)
VECTOR_STORE_CACHE: Dict[str, FAISS] = {}

# --- 1. RAG Component Setup (Initialized once at startup) ---

# Choose a stable, fast Groq Model
GROQ_MODEL = "llama-3.1-8b-instant" 

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.3
)

# Initialize Embeddings
# Must be initialized outside the request loop for efficiency
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Prompt Template
PROMPT_TEMPLATE = PromptTemplate(
    template="""
        You are a helpful assistant. Answer the user's question only from the provided context.
        If the context is insufficient and you cannot find the answer, you must respond with: 
        "I could not find the answer in the provided video transcript."

        Context: 
        ---
        {context}
        ---
        
        Question: {question}
    """,
    input_variables=['context', 'question']
)

# --- 2. Data Models ---

class IndexRequest(BaseModel):
    video_id: str

class QueryRequest(BaseModel):
    video_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_chunks: int

# --- 3. Core RAG Utility Functions ---

def clean_transcript(raw_text: str) -> str:
    """Robust VTT cleaning: removes timestamps, cues, and VTT header markup."""
    # Remove metadata headers
    text = re.sub(r"(WEBVTT|Kind:|Language:|NOTE).*\n", "", raw_text)
    # Remove timestamps (00:00:00.000 --> 00:00:00.000)
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*\n?", "", text)
    # Remove cue IDs (simple numbers on their own lines)
    text = re.sub(r"^\d+$\n?", "", text, flags=re.MULTILINE)
    # Remove HTML-like tags (<c.colorE5E5E5>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def download_and_process_transcript(video_id: str) -> FAISS:
    """Downloads transcript, cleans it, and creates a FAISS vector store."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Create transcript directory
    save_dir = "transcripts"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the output template. 
    # yt-dlp will append the language code and extension (e.g., .en.vtt)
    # so we use a base filename inside the directory.
    output_template = os.path.join(save_dir, video_id)

    # 1. Download Transcript
    ydl_opts = {
        "skip_download": True, 
        "writesubtitles": True,
        "writeautomaticsub": True,
        # Check multiple English variants to be safe
        "subtitleslangs": ["en", "en-US", "en-orig"], 
        "subtitlesformat": "vtt", # Force VTT format
        "outtmpl": output_template, # e.g., transcripts/VIDEO_ID
        "quiet": True,
        "no_warnings": True,
    }
    
    print(f"Attempting download for {video_id}...")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"yt-dlp warning (might be non-fatal): {e}")

    # 2. Find the downloaded file using pattern matching (Glob)
    # This fixes the issue where the file might be named .en.vtt or .en-US.vtt
    search_pattern = f"{output_template}*.vtt"
    found_files = glob.glob(search_pattern)

    if not found_files:
        print(f"Files found in dir: {os.listdir(save_dir)}")
        raise HTTPException(status_code=404, detail=f"No transcript downloaded. YouTube might not have English captions for {video_id}.")

    vtt_filename = found_files[0] # Use the first matching file
    print(f"Found transcript file: {vtt_filename}")

    # 3. Load, Clean, and Chunk
    try:
        with open(vtt_filename, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        cleaned_text = clean_transcript(raw_content)
        
        if not cleaned_text:
            raise ValueError("Transcript file was empty after cleaning.")

        chunks = splitter.create_documents([cleaned_text])
        
        if not chunks:
            raise ValueError("No text chunks created from transcript.")

        # 4. Create Vector Store
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process transcript: {e}")


def format_docs_for_prompt(docs):
    """Combines retrieved documents into a single string for the prompt context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# --- 4. FastAPI Endpoints ---

@app.get("/")
def read_root():
    return {"message": "YouTube RAG API is running."}

@app.post("/index", status_code=200)
def index_video(request: IndexRequest):
    """
    Processes a video ID: downloads the transcript, embeds it,
    and caches the FAISS vector store.
    """
    video_id = request.video_id

    # Refresh index if requested, or check cache
    if video_id in VECTOR_STORE_CACHE:
        print(f"Video {video_id} found in cache.")
        return {"status": "success", "message": f"Video {video_id} already indexed (cached)."}

    print(f"Indexing video: {video_id}...")
    try:
        vector_store = download_and_process_transcript(video_id)
        VECTOR_STORE_CACHE[video_id] = vector_store
        print(f"Successfully indexed video: {video_id}.")
        return {"status": "success", "message": f"Video {video_id} indexed successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    """
    Performs RAG query using the cached vector store.
    """
    video_id = request.video_id
    question = request.question

    if video_id not in VECTOR_STORE_CACHE:
        raise HTTPException(status_code=404, detail="Video must be indexed before querying.")

    vector_store = VECTOR_STORE_CACHE[video_id]
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    rag_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs_for_prompt), 
            question=RunnablePassthrough() 
        )
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    try:
        answer = rag_chain.invoke(question)
        retrieved_docs = retriever.invoke(question)

        return QueryResponse(
            answer=answer,
            source_chunks=len(retrieved_docs)
        )
    except Exception as e:
        print(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed.")

# To run: uvicorn backend_api:app --reload