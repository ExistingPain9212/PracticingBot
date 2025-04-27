from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load GPT-2 Medium
model_name = "gpt2-medium"  # This is the larger, better GPT-2 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Important! GPT-2 doesn't have a padding token, so we set it manually
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Simple chatbot loop
conversation_history = ""

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Update conversation history
    conversation_history += f"User: {user_input}\n"

    # Encode
    inputs = tokenizer(conversation_history, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # Generate a reply
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,        # Enable randomness
            top_k=50,              # Top-K sampling
            top_p=0.95,            # Nucleus sampling
            temperature=0.7,       # Slight randomness to improve creativity
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Only show new AI-generated part
    ai_response = generated_text[len(conversation_history):]
    print(f"\n🤖 AI: {ai_response.strip()}")

    # Update history
    conversation_history += f"AI: {ai_response.strip()}\n"
