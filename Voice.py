from typing import List, Dict, Tuple
import os
import tempfile
from datetime import datetime
import uuid

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import TextEmbedding
from openai import OpenAI

from agents import Agent, Runner

# Load environment variables
load_dotenv()

# Qdrant Collection
COLLECTION_NAME = "voice-rag-agent"


# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# SIDEBAR CONFIG
# -----------------------------------------------------------
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

        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            voices,
            index=voices.index(st.session_state.selected_voice),
        )


# -----------------------------------------------------------
# QDRANT SETUP
# -----------------------------------------------------------
def setup_qdrant() -> Tuple[QdrantClient, TextEmbedding]:
    if not st.session_state.qdrant_url or not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant credentials missing")

    client = QdrantClient(
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key
    )

    embedding_model = TextEmbedding()
    dim = len(list(embedding_model.embed(["test"]))[0])

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    return client, embedding_model


# -----------------------------------------------------------
# PDF PROCESSING
# -----------------------------------------------------------
def process_pdf(file) -> List:
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


# -----------------------------------------------------------
# STORE EMBEDDINGS
# -----------------------------------------------------------
def store_embeddings(client, embedding_model, documents):
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


# -----------------------------------------------------------
# AGENTS
# -----------------------------------------------------------
def setup_agents(api_key: str):
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


# -----------------------------------------------------------
# PROCESS QUERY (SYNCHRONOUS)
# -----------------------------------------------------------
def process_query(query: str) -> Dict:
    try:
        client = st.session_state.client
        embed = st.session_state.embedding_model

        # Embedding query
        q_embed = list(embed.embed([query]))[0]

        search = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_embed.tolist(),
            limit=3,
            with_payload=True
        )

        results = search.points

        if not results:
            return {"status": "error", "error": "No relevant documents found."}

        # Build context
        context = ""
        for r in results:
            payload = r.payload
            context += f"\nFrom file {payload['file_name']}:\n{payload['content']}\n"

        context += f"\n\nUser Question: {query}"

        # Agents
        processor = st.session_state.processor_agent
        tts_agent = st.session_state.tts_agent

        # Generate text
        processed_text = Runner.run_sync(processor, context).final_output

        # Voice formatting
        processed_spoken = Runner.run_sync(tts_agent, processed_text).final_output

        # Generate MP3
        client_openai = OpenAI(api_key=st.session_state.openai_api_key)

        speech = client_openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=st.session_state.selected_voice,
            input=processed_text,
            response_format="mp3"
        )

        audio_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
        with open(audio_path, "wb") as f:
            f.write(speech)

        return {
            "status": "success",
            "text": processed_text,
            "audio": audio_path,
            "sources": list({r.payload["file_name"] for r in results})
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# -----------------------------------------------------------
# MAIN STREAMLIT APP
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Voice RAG Agent", layout="wide")
    init_session_state()
    setup_sidebar()

    st.title("🎙️ Voice RAG Agent (Streamlit-Ready)")
    st.caption("Upload PDFs ➜ Ask questions ➜ Get text + audio answers.")

    # PDF Upload
    file = st.file_uploader("Upload PDF files", type=["pdf"])

    if file:
        if file.name not in st.session_state.processed_documents:
            if not st.session_state.client:
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

    if st.session_state.processed_documents:
        st.sidebar.subheader("Processed PDFs")
        for name in st.session_state.processed_documents:
            st.sidebar.write("📄 " + name)

    # Question Interface
    ask = st.text_input(
        "Ask something:",
        placeholder="Example: Explain how authentication works."
    )

    if ask and st.session_state.setup_complete:
        if not st.session_state.processor_agent:
            st.session_state.processor_agent, st.session_state.tts_agent = setup_agents(
                st.session_state.openai_api_key
            )

        with st.spinner("Generating answer..."):
            result = process_query(ask)

        if result["status"] == "success":
            st.markdown("### 📝 Answer")
            st.write(result["text"])

            st.markdown("### 🔊 Audio")
            st.audio(result["audio"])

            with open(result["audio"], "rb") as f:
                st.download_button(
                    "Download MP3",
                    f.read(),
                    file_name="response.mp3",
                    mime="audio/mp3"
                )

            st.markdown("### 📚 Sources")
            for s in result["sources"]:
                st.write("- " + s)
        else:
            st.error(result["error"])

    elif not st.session_state.setup_complete:
        st.info("Upload a PDF to begin.")


if __name__ == "__main__":
    main()
