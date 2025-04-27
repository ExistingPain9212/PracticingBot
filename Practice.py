from transformers import GPTJForCausalLM, AutoTokenizer
import torch

# Load GPT-J model and tokenizer from Hugging Face
model_name = "EleutherAI/gpt-j-6B"  # You can also use "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)

# Define a context variable
conversation_history = ""

# Main interaction loop
while True:
    user_input = input("\n👾 Enter your prompt (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Add the user input to conversation history
    conversation_history += f"User: {user_input}\n"

    # Tokenize the conversation history
    inputs = tokenizer(conversation_history, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    # Generate AI output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,     # Randomness for diverse outputs
            top_k=50,           # Top-k sampling
            top_p=0.95          # Nucleus sampling
        )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_reply = response[len(conversation_history):]  # Strip user history from response
    print(f"\n🤖 AI Reply: {ai_reply}")

    # Update conversation history with AI response
    conversation_history += f"AI: {ai_reply}\n"
