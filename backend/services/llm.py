from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class LLMService:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def generate_answer(self, context: str, question: str) -> str:
        prompt = f"""<s>[INST] You are a helpful study assistant. Based ONLY on the context below, give a short and precise answer. If the context doesn't contain the answer, say "Not found in documents."

Context: {context}

Question: {question} [/INST]"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.3)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        prompt = f"""<s>[INST] Generate exactly {num_questions} multiple choice quiz questions from the text below.
        Each question must have 4 options (A, B, C, D) and indicate the correct answer.
        Return ONLY valid JSON as a list like this:
            [
                {{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}},
                    ...
            ]

Text: {context} [/INST]"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=1500, temperature=0.3)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

    def summarize(self, context: str) -> str:
        prompt = f"""<s>[INST] Summarize the following text in a short and clear way. Focus on the key points.

Text: {context} [/INST]"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=300, temperature=0.3)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()