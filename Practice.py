from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load TinyLlama 1.1B Chat model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"🔄 Downloading and loading model: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Ensure padding tokens are handled
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Create a simple text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)

# Simple chatbot loop
conversation_history = ""

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    conversation_history += f"User: {user_input}\nAI:"

    response = generator(
        conversation_history,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = response[0]["generated_text"]

    # Extract only the AI's new message
    ai_reply = generated_text[len(conversation_history):].strip()
    print(f"\n🤖 AI: {ai_reply}")

    # Update conversation history
    conversation_history += f" {ai_reply}\n"
