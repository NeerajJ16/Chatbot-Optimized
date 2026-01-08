# rag/llm_phi2.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_LLM_INSTANCE = None

class Phi2LLM:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"Loading Phi-2 model ({model_name})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_answer(self, context, question, max_tokens=256):
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # ✅ Extract ONLY new tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()


def get_llm():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = Phi2LLM()
    return _LLM_INSTANCE
