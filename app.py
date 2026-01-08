import streamlit as st
from rag.query_rag import (
    load_chunks_and_embeddings,
    embed_chunks,
    answer_question
)

st.set_page_config(
    page_title="Portfolio Chatbot (Streamlit)",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Portfolio Assistant")
st.caption("Ask about education, experience, projects, research, or skills")

# -------------------------
# Load RAG assets ONCE
# -------------------------
@st.cache_resource
def load_rag():
    chunks = load_chunks_and_embeddings()
    embeddings = embed_chunks(chunks)
    return chunks, embeddings

chunks, embeddings = load_rag()

# -------------------------
# Session state
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------
# Render history
# -------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Input
# -------------------------
question = st.chat_input("Ask a question…")

if question:
    # User message
    st.session_state.chat.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(question, chunks, embeddings)
            st.markdown(answer)

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer
    })
