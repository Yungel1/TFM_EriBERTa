from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM, pipeline
import torch


def run_ner_gemma_pipeline():
    pipe = pipeline("text-generation", model="google/gemma-3-1b-it")
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}, ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"}, ]
            },
        ],
    ]
    output = pipe(messages, max_new_tokens=50)
    print(output)

if __name__ == "__main__":
    run_ner_gemma_pipeline()