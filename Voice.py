from typing import List, Dict, Tuple
import os
import tempfile
from datetime import datetime
import uuid
import inspect
import asyncio

import streamlit as st

# Safe dotenv import (won't crash if module missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    def load_dotenv(*a, **k):
        return

# Optional / external imports that may fail on some environments
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams
except Exception:
    QdrantClient = None
    models = None
    Distance = None
    VectorParams = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    RecursiveCharacterTextSplitter = None
    PyPDFLoader = None

try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Agents module (your local file). Provide safe stubs if missing.
try:
    from agents import Agent, Runner
except Exception:
    Agent = None
    Runner = None

# Qdrant Collection name
COLLECTION_NAME = "voice-rag-agent"

# -----------------------
# Helpers and fallbacks
# -----------------------

def run_runner_sync(runner_obj, *args, **kwargs):
    """
    Attempt to call a synchronous runner API:
      - If Runner.run_sync exists, use it.
      - If Runner.run exists and returns coroutine, run it via asyncio.
      - If Runner.run exists and is already sync, call it.
    This makes the code robust against different Runner implementations.
    """
    if runner_obj is None:
        raise RuntimeError("Runner object is not available.")

    # If the runner has run_sync, use it
    if hasattr(Runner, "run_sync"):
        return Runner.run_sync(*args, **kwargs)

    # If Runner.run is a coroutine function (async def)
    if hasattr(Runner, "run") and inspect.iscoroutinefunction(Runner.run):
        # call and run in new event loop
        coro = Runner.run(*args, **kwargs)
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # If already in an event loop (unlikely in Streamlit), create new task
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    # If Runner.run is a normal callable
    if hasattr(Runner, "run"):
        return Runner.run(*args, **kwargs)

    raise RuntimeError("No runnable Runner method available.")


# -----------------------
# Session state init
# -----------------------
def init_session_state() -> None:
    defaults = {
        "qdrant_url": "",
        "qdrant_api_key": "",
        "openai_api_key": "",
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": "coral",
        "processed_documents": [],
        "setup_complete": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------
# Sidebar
# -----------------------
def setup_sidebar() -> None:
    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL", st.session_state.qdrant_url, type="password"
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key", st.session_state.qdrant_api_key, type="password"
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", st.session_state.openai_api_key, type="password"
        )

        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")

        voices = [
            "alloy", "ash", "ballad", "coral", "echo", "fable",
            "onyx", "nova", "sage", "shimmer", "verse"
        ]

        try:
            idx = voices.index(st.session_state.selected_voice)
        except Exception:
            idx = 0

        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            voices,
            index=idx,
        )


# -----------------------
# Qdrant setup
# -----------------------
def setup_qdrant():
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is not installed. Add qdrant-client to requirements.")

    if not st.session_state.qdrant_url or not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant credentials missing")

    client = QdrantClient(
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key
    )

    if TextEmbedding is None:
        raise RuntimeError("fastembed is not installed. Add fastembed to requirements.")

    embedding_model = TextEmbedding()
    dim = len(list(embedding_model.embed(["test"]))[0])

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    except Exception as e:
        # On many setups the create_collection will raise "already exists" or similar; ignore that
        if "already exists" not in str(e).lower():
            raise e

    return client, embedding_model


# -----------------------
# PDF processing
# -----------------------
def process_pdf(file) -> List:
    if PyPDFLoader is None or RecursiveCharacterTextSplitter is None:
        st.error("PDF processing dependencies are not installed. Install langchain-community and langchain-text-splitters.")
        return []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()

            for d in docs:
                d.metadata.update({
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    except Exception as e:
        st.error(f"PDF Error: {e}")
        return []


# -----------------------
# Store embeddings
# -----------------------
def store_embeddings(client, embedding_model, documents):
    if client is None or models is None:
        raise RuntimeError("Qdrant client or models missing.")

    for doc in documents:
        vector = list(embedding_model.embed([doc.page_content]))[0]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={**doc.metadata, "content": doc.page_content}
                )
            ]
        )


# -----------------------
# Agents setup
# -----------------------
def setup_agents(api_key: str):
    # Minimal stub Agent fallback
    if Agent is None:
        class _Agent:
            def __init__(self, name, instructions, model):
                self.name = name
                self.instructions = instructions
                self.model = model
        _Agent = _Agent
        processor = _Agent("Processor (stub)", "Answer concisely.", "gpt-stub")
        tts = _Agent("TTS (stub)", "Make speechable.", "gpt-stub")
        return processor, tts

    os.environ["OPENAI_API_KEY"] = api_key

    processor = Agent(
        name="Documentation Processor",
        instructions="""
        Answer the question clearly and concisely.
        Reference the source PDF when needed.
        Keep output readable and speakable.
        """,
        model="gpt-4o"
    )

    tts = Agent(
        name="TTS",
        instructions="Rewrite the text to be spoken clearly and naturally.",
        model="gpt-4o"
    )

    return processor, tts


# -----------------------
# Process query
# -----------------------
def process_query(query: str) -> Dict:
    try:
        client = st.session_state.client
        embed = st.session_state.embedding_model

        if client is None or embed is None:
            return {"status": "error", "error": "Embeddings or vector DB not initialized."}

        q_embed = list(embed.embed([query]))[0]

        search = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_embed.tolist(),
            limit=3,
            with_payload=True
        )

        # qdrant-client may return different shapes, handle robustly
        results = getattr(search, "points", None) or search

        if not results:
            return {"status": "error", "error": "No relevant documents found."}

        context = ""
        for r in results:
            payload = getattr(r, "payload", r.get("payload", {}))
            file_name = payload.get("file_name", "Unknown")
            content = payload.get("content", "")
            context += f"\nFrom file {file_name}:\n{content}\n"

        context += f"\n\nUser Question: {query}"

        # Agents
        processor = st.session_state.processor_agent
        tts_agent = st.session_state.tts_agent

        # Generate text using runner helper
        try:
            proc_result = run_runner_sync(processor, context)
            processed_text = getattr(proc_result, "final_output", str(proc_result))
        except Exception as e:
            # If Runner not available, fallback to returning context summary
            processed_text = f"(Agent not available) Context summary:\n{context[:2000]}"

        try:
            tts_result = run_runner_sync(tts_agent, processed_text)
            processed_spoken = getattr(tts_result, "final_output", str(tts_result))
        except Exception:
            processed_spoken = processed_text  # fallback

        # Generate MP3 via OpenAI SDK (if available)
        if OpenAI is None:
            # Save processed_text as a simple text file to download instead
            audio_path = None
            temp_txt = os.path.join(tempfile.gettempdir(), f"response_{uuid.uuid4()}.txt")
            with open(temp_txt, "w", encoding="utf-8") as f:
                f.write(processed_text)
            return {
                "status": "success",
                "text": processed_text,
                "audio": None,
                "text_file": temp_txt,
                "sources": list({(r.payload.get("file_name") if hasattr(r, "payload") else r.get("payload", {}).get("file_name", "Unknown")) for r in results})
            }

        # Create OpenAI client
        client_openai = OpenAI(api_key=st.session_state.openai_api_key)

        # The OpenAI SDK response shapes can vary; handle both bytes and objects with .content
        try:
            speech_resp = client_openai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=st.session_state.selected_voice,
                input=processed_text,
                response_format="mp3"
            )
        except Exception as e:
            return {"status": "error", "error": f"OpenAI TTS error: {e}"}

        audio_bytes = None
        if isinstance(speech_resp, (bytes, bytearray)):
            audio_bytes = bytes(speech_resp)
        else:
            # try common attributes
            audio_bytes = getattr(speech_resp, "content", None) or getattr(speech_resp, "data", None) or None
            if isinstance(audio_bytes, str):
                audio_bytes = audio_bytes.encode("utf-8")

        if not audio_bytes:
            return {"status": "error", "error": "Failed to obtain audio bytes from OpenAI response."}

        audio_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        return {
            "status": "success",
            "text": processed_text,
            "audio": audio_path,
            "sources": list({(r.payload.get("file_name") if hasattr(r, "payload") else r.get("payload", {}).get("file_name", "Unknown")) for r in results})
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# -----------------------
# Main app
# -----------------------
def main():
    st.set_page_config(page_title="Voice RAG Agent", layout="wide")
    init_session_state()
    setup_sidebar()

    st.title("🎙️ Voice RAG Agent (Streamlit-Ready)")
    st.caption("Upload PDFs ➜ Ask questions ➜ Get text + audio answers.")

    file = st.file_uploader("Upload PDF files", type=["pdf"])

    if file:
        if file.name not in st.session_state.processed_documents:
            try:
                if st.session_state.client is None:
                    st.session_state.client, st.session_state.embedding_model = setup_qdrant()

                docs = process_pdf(file)
                if docs:
                    store_embeddings(
                        st.session_state.client,
                        st.session_state.embedding_model,
                        docs
                    )
                    st.session_state.processed_documents.append(file.name)
                    st.success(f"Processed: {file.name}")
                    st.session_state.setup_complete = True
            except Exception as e:
                st.error(f"Error during processing: {e}")

    if st.session_state.processed_documents:
        st.sidebar.subheader("Processed PDFs")
        for name in st.session_state.processed_documents:
            st.sidebar.write("📄 " + name)

    ask = st.text_input(
        "Ask something:",
        placeholder="Example: Explain how authentication works."
    )

    if ask and st.session_state.setup_complete:
        if st.session_state.processor_agent is None:
            st.session_state.processor_agent, st.session_state.tts_agent = setup_agents(
                st.session_state.openai_api_key
            )

        with st.spinner("Generating answer..."):
            result = process_query(ask)

        if result.get("status") == "success":
            st.markdown("### 📝 Answer")
            st.write(result.get("text", "(no text returned)"))

            if result.get("audio"):
                st.markdown("### 🔊 Audio")
                st.audio(result["audio"])
                with open(result["audio"], "rb") as f:
                    st.download_button(
                        "Download MP3",
                        f.read(),
                        file_name="response.mp3",
                        mime="audio/mp3"
                    )
            else:
                # If we returned a text file (because TTS not available), allow download
                if result.get("text_file"):
                    with open(result["text_file"], "rb") as f:
                        st.download_button(
                            "Download text response",
                            f.read(),
                            file_name="response.txt",
                            mime="text/plain"
                        )

            st.markdown("### 📚 Sources")
            for s in result.get("sources", []):
                st.write("- " + str(s))
        else:
            st.error(result.get("error", "Unknown error"))

    elif not st.session_state.setup_complete:
        st.info("Upload a PDF to begin.")


if __name__ == "__main__":
    main()
