# rag/llm_phi2_langchain.py

import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_LLM_INSTANCE = None

class Phi2LLM:
    def __init__(self, model_name="google/gemma-2-2b-it:nebius"):
        hf_token = st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.error("Missing HF_TOKEN! Please add it to Streamlit secrets.")
            hf_token = os.environ.get("HF_TOKEN", "no_token_found")

        os.environ["OPENAI_API_KEY"] = hf_token

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            max_tokens=500,
            openai_api_base="https://router.huggingface.co/v1",
            openai_api_key=hf_token,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are a professional portfolio assistant representing Neeraj Jawahirani.

Use ONLY the information provided in the context below.
Do NOT use outside knowledge or make up information.
If the question asks for the most recent / latest / current item, answer using ONLY that single item.

Instructions:
- Use ONLY the information provided in the context.
- Rewrite the answer in natural, professional language.
- Do NOT copy the context verbatim unless necessary.
- When possible, combine details into a single fluent sentence or short paragraph.
- If multiple entries are relevant, summarize them clearly.
- Only say "I don’t have that information." if no entry is relevant.

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def generate_answer(self, context: str, question: str) -> str:
        try:
            messages = self.prompt.format_messages(
                context=context,
                question=question
            )
            response = self.llm(messages)
            return response.content.strip()

        except Exception as e:
            return f"Error connecting to the AI service: {str(e)}"


def get_llm():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = Phi2LLM()
    return _LLM_INSTANCE
