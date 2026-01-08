import os
import streamlit as st
from openai import OpenAI

_LLM_INSTANCE = None

class Phi2LLM:
    def __init__(self, model_name="google/gemma-2-2b-it:nebius"):
        # 1. Pull the token from Streamlit Secrets
        hf_token = st.secrets.get("HF_TOKEN")
        
        # 2. Check if it's missing and provide a clear warning
        if not hf_token:
            st.error("Missing HF_TOKEN! Please add it to Streamlit Cloud -> Settings -> Secrets.")
            # Fallback for local testing if you forgot the secrets file
            hf_token = os.environ.get("HF_TOKEN", "no_token_found")

        # 3. SET the environment variable manually to satisfy the OpenAI client
        os.environ["OPENAI_API_KEY"] = hf_token 
        
        # 4. Initialize the client
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token, # We pass it here AND as an env var above
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
