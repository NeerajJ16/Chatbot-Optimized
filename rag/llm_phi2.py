import os
import streamlit as st
from openai import OpenAI

_LLM_INSTANCE = None

class Phi2LLM:
    def __init__(self, model_name="openai/gpt-oss-20b:groq"):
        # We use st.secrets for safety on Streamlit Cloud
        # Ensure HF_TOKEN is added to your Streamlit App Secrets
        api_key = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        
        # Pointing to the Hugging Face OpenAI-compatible router
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def generate_answer(self, context, question, max_tokens=500):
        # Using your exact professional prompt format
        prompt = f"""You are a professional portfolio assistant representing Neeraj Jawahirani.

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
        try:
            # We use the Chat Completion interface which the Router requires
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error connecting to the AI service: {str(e)}"


def get_llm():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        # We keep the name Phi2LLM but it now runs GPT-OSS-20B via API
        _LLM_INSTANCE = Phi2LLM()
    return _LLM_INSTANCE
