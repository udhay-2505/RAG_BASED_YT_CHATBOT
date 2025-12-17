import os
import re
import sys
import yt_dlp
from dotenv import load_dotenv

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP AND DEPENDENCY INSTALLATION ---

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

print("Installing required packages (if not already installed)...")
# Note: In a notebook, use !pip install. In a script, you might skip this or use poetry/requirements.txt
# For a clean notebook execution, keeping the installs here.
# You will need a GROQ_API_KEY in your .env file for the LLM to work.
#!{sys.executable} -m pip install -q youtube-transcript-api langchain langchain-core langchain-community langchain-groq faiss-cpu yt-dlp langchain-text-splitters sentence-transformers

print("✅ Setup complete.")

# --- 2. CONFIGURATION ---

# The video ID you were using
video_id = "Ks-_Mh1QhMc"
url = f"https://www.youtube.com/watch?v={video_id}"
vtt_filename = f"{video_id}.en.vtt"

# --- 3. DATA INGESTION: DOWNLOAD TRANSCRIPT ---

print(f"\n--- Downloading transcript for video ID: {video_id} ---")
ydl_opts = {
    "skip_download": True, # Don't download the video itself
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en"],
    "outtmpl": video_id, # This sets the base name for the output file
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Check if English subtitles exist before attempting download
        info = ydl.extract_info(url, download=False)
        subs = info.get("subtitles") or info.get("automatic_captions")

        if subs and "en" in subs:
            # Download the subtitle file (will be named Ks-_Mh1QhMc.en.vtt)
            ydl.download([url])
            print(f"✅ Transcript file '{vtt_filename}' downloaded!")
        else:
            print("❌ No English subtitles found for this video. Cannot proceed.")
            sys.exit(1)

except Exception as e:
    print(f"An error occurred during yt-dlp download: {e}")
    sys.exit(1)


# --- 4. DATA PREPROCESSING: CLEANING AND SPLITTING ---

def clean_transcript(raw_text: str) -> str:
    """
    Robust function to clean raw VTT/SRT text by removing headers, timestamps, 
    and markup tags, leaving only the spoken dialogue.
    """
    # Remove all VTT/SRT time codes (e.g., 00:00:01.000 --> 00:00:04.000)
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*\n?", "", raw_text)
    # Remove cue numbers (lines consisting only of digits)
    text = re.sub(r"^\d+$\n?", "", text, flags=re.MULTILINE)
    # Remove header lines (WEBVTT, Kind, Language, NOTE)
    text = re.sub(r"(WEBVTT|Kind|Language|NOTE).*?\n", "", text)
    # Remove any remaining tags (like <c>, </i>, or <01:23:45.678>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace (multiple spaces/newlines to a single space)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean_vtt(vtt_path):
    """Loads the VTT file, extracts raw content, and cleans it."""
    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        return clean_transcript(raw_content)
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {vtt_path}")
        return ""

# Load and clean the transcript
text = load_and_clean_vtt(vtt_filename)
if not text:
    print("Failed to load or clean transcript. Exiting.")
    sys.exit(1)

print("✅ Transcript cleaned successfully!")

# Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text])
print(f"Generated {len(chunks)} text chunks.")

# --- 5. INDEXING: EMBEDDINGS AND VECTOR STORE ---

print("--- Creating vector store (Embedding and FAISS) ---")
# Use a high-quality, fast embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ Vector store created!")

# --- 6. RAG CHAIN CONSTRUCTION ---

# 6.1. Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 6.2. LLM (Ensure GROQ_API_KEY is set in your .env file)
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ ERROR: GROQ_API_KEY environment variable is not set. Please check your .env file.")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    # FIX: Use a currently stable model like Llama 3 8B
    model_name="openai/gpt-oss-20b", 
    temperature=0.3
)

# 6.3. Prompt Template
prompt = PromptTemplate(
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

# 6.4. LCEL Chain Components
# Function to format documents for the prompt (already cleaned)
def format_docs_for_prompt(docs):
    """Combines retrieved documents into a single string for the prompt context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# The RAG Chain structure:
# 1. Input question
# 2. Parallel run: Retrieve context AND pass question through
# 3. Format context using format_docs_for_prompt
# 4. Invoke Prompt
# 5. Invoke LLM
# 6. Parse output

main_chain = (
    RunnableParallel(
        # The retriever fetches and formats the documents using the question input
        context=retriever | RunnableLambda(format_docs_for_prompt), 
        question=RunnablePassthrough() # Passes the original question through
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG Chain constructed successfully!")

# --- 7. GENERATION (TESTING) ---

question_1 = "what is the title of the video, what is it all about?"
print(f"\n--- Testing Query 1 ---\nQuestion: {question_1}")
answer_1 = main_chain.invoke(question_1)
print(f"Answer:\n{answer_1}")

question_2 = "is the topic discussed in th video? If yes what was it"
print(f"\n--- Testing Query 2 ---\nQuestion: {question_2}")
answer_2 = main_chain.invoke(question_2)
print(f"Answer:\n{answer_2}")