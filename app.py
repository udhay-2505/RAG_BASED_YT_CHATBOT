import streamlit as st
import requests
import re
import os
from dotenv import load_dotenv

# --- 0. Configuration and Setup ---

load_dotenv()

# Use the backend URL where the FastAPI app is running
# Default Uvicorn port is 8000
API_URL = "http://localhost:8000" 

# --- 1. Streamlit UI Functions ---

def extract_video_id(url_or_id: str) -> str:
    """Extracts the YouTube video ID from a URL or returns the ID if already clean."""
    if "youtu.be" in url_or_id or "youtube.com" in url_or_id:
        match = re.search(r"(?<=v=)[\w-]+|(?<=youtu\.be\/)[\w-]+", url_or_id)
        return match.group(0) if match else ""
    return url_or_id

def index_video(video_id: str):
    """Calls the FastAPI endpoint to download and index the video transcript."""
    st.session_state.indexing = True
    st.session_state.indexed_id = None
    
    st.info(f"Indexing video ID: `{video_id}`. This may take a moment...")
    
    try:
        response = requests.post(f"{API_URL}/index", json={"video_id": video_id}, timeout=300)
        
        if response.status_code == 200:
            st.session_state.indexed_id = video_id
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.success(f"Video indexed! Ready to chat about video ID: `{video_id}`")
        else:
            error_detail = response.json().get("detail", "Unknown error during indexing.")
            st.error(f"Indexing failed (Status {response.status_code}): {error_detail}")
            st.session_state.indexed_id = None

    except requests.exceptions.Timeout:
        st.error("Indexing request timed out. The video might be very long or the backend is slow.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the FastAPI backend at {API_URL}. Ensure the API is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        st.session_state.indexing = False
        st.rerun()


def send_query(video_id: str, question: str):
    """Calls the FastAPI endpoint for RAG query."""
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking... (Powered by Groq)"):
        try:
            payload = {"video_id": video_id, "question": question}
            response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            elif response.status_code == 404:
                st.session_state.messages.append({"role": "assistant", "content": f"ERROR: Video ID `{video_id}` is not currently indexed. Please click 'Index Video' first."})
            else:
                error_detail = response.json().get("detail", "Unknown API error.")
                st.session_state.messages.append({"role": "assistant", "content": f"ERROR: Query failed (Status {response.status_code}): {error_detail}"})

        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({"role": "assistant", "content": f"ERROR: Cannot connect to the FastAPI backend at {API_URL}. Is the API running?"})
        except requests.exceptions.Timeout:
            st.session_state.messages.append({"role": "assistant", "content": "ERROR: Query request timed out. The model took too long to respond."})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"ERROR: An unexpected error occurred: {e}"})

# --- 2. Main Streamlit App Layout ---

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("YouTube Video RAG Chatbot 🤖")
st.caption("Backend: FastAPI + LangChain/Groq | Frontend: Streamlit")

# Initialize Session State
if "indexed_id" not in st.session_state:
    st.session_state.indexed_id = None
if "indexing" not in st.session_state:
    st.session_state.indexing = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar: Video Input and Indexing ---
with st.sidebar:
    st.header("1. Video Setup")
    
    # Text input for YouTube URL or ID
    url_input = st.text_input(
        "Enter YouTube URL or ID",
        key="url_input",
        placeholder="e.g., Ks-_Mh1QhMc or full URL"
    )
    
    # Extract ID
    raw_video_id = extract_video_id(url_input)
    
    st.markdown("---")
    
    # Index Button
    if st.button(
        "Index Video", 
        disabled=not raw_video_id or st.session_state.indexing, 
        use_container_width=True,
        type="primary"
    ):
        index_video(raw_video_id)
        
    st.markdown("---")
    
    # Display Current Status
    st.header("2. Status")
    if st.session_state.indexed_id:
        st.success(f"Indexed ID: `{st.session_state.indexed_id}`")
    elif st.session_state.indexing:
        st.warning("Indexing in progress...")
    else:
        st.warning("No video indexed yet.")

# --- Main Chat Interface ---

if st.session_state.indexed_id:
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the video transcript..."):
        # This will trigger the send_query function, which updates st.session_state.messages
        # The app will rerun and display the new messages automatically
        send_query(st.session_state.indexed_id, prompt)

else:
    st.info("Please enter a YouTube video URL/ID in the sidebar and click 'Index Video' to start the RAG process.")

st.sidebar.markdown("""
---
**Note:** Ensure your FastAPI backend is running before launching this Streamlit app.
""")