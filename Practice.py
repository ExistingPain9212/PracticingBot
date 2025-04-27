from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load the Phi-2 model
model_name = "microsoft/phi-2"

print(f"🔄 Downloading and loading model: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Padding handling
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Create a text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Conversation loop
system_prompt = "You are a helpful, friendly, and intelligent AI assistant.\n"
conversation_history = system_prompt

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    prompt = conversation_history + f"User: {user_input}\nAI:"
    
    response = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = response[0]["generated_text"]
    ai_reply = generated_text[len(prompt):].strip()

    # Clean cutoff if it starts rambling
    ai_reply = ai_reply.split("User:")[0].strip()

    print(f"\n🤖 AI: {ai_reply}")

    # Update conversation history
    conversation_history += f"User: {user_input}\nAI: {ai_reply}\n"
