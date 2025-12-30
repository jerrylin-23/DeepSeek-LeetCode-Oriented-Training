"""
Merge LoRA adapter with base model and prepare for GGUF conversion.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-base"
ADAPTER_PATH = "./deepseek-leetcode-final"
OUTPUT_PATH = "./merged_model"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",  # Use CPU to avoid GPU memory issues
    trust_remote_code=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Done! Merged model saved to:", OUTPUT_PATH)
print("\nNext step: Convert to GGUF with:")
print(f"  python llama_cpp/convert_hf_to_gguf.py {OUTPUT_PATH} --outfile deepseek-leetcode.gguf --outtype f16")
