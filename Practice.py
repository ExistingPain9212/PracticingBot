from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Define local temp path
local_dir = "/tmp/roaster_model"

# Download model from Hugging Face if not already present
if not os.path.exists(local_dir):
    print("🔻 Downloading model to temporary folder...")
    snapshot_download(repo_id="distilgpt2", local_dir=local_dir)
    print("✅ Download complete!")

# Load model and tokenizer
print("🔄 Loading model into memory...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir)
print("✅ Model loaded!")

# Main interaction loop
while True:
    prompt = input("\n👾 Enter your prompt (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # number of tokens to generate
            do_sample=True,     # randomness
            top_k=50,           # Top-k sampling
            top_p=0.95          # nucleus sampling
        )

    # Decode and print the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🤖 AI Reply: {response}")
