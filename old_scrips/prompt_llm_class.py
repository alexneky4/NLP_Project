from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class PromptedLLMModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def predict(self, anchor, text_a, text_b):
        prompt = f"""
                    Compare the following stories in terms of:
                    1. Abstract theme
                    2. Course of action
                    3. Outcome
                    
                    Reference story:
                    {anchor}
                    
                    Story A:
                    {text_a}
                    
                    Story B:
                    {text_b}
                    
                    Which story is more narratively similar?
                    Answer with only A or B.
                """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.strip().upper()

        return response.startswith("A")
