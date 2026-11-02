import torch
from transformers import AutoTokenizer
from model.transformer import GPTModel

def test_generation():
    print("Initializing tokenizer and downloading GPT-2 weights...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPTModel.from_pretrained("gpt2")
    model.eval()

    prompt = "Artificial Intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\nPrompt: '{prompt}'")
    print(f"Token IDs: {input_ids.tolist()}")

    # Generate 15 tokens autoregressively using KV cache
    generated = input_ids
    kv_cache = None

    print("\nGenerating tokens autoregressively with KV-cache...")
    with torch.no_grad():
        for step in range(15):
            # If kv_cache exists, only feed the latest single token!
            current_input = generated[:, -1:] if kv_cache is not None else generated
            logits, kv_cache = model(current_input, kv_cache=kv_cache)
            
            # Greedy argmax sampling
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token_id], dim=-1)

            # Decode single token
            token_str = tokenizer.decode(next_token_id[0])
            print(f"Step {step+1:02d}: token={next_token_id.item()} -> '{token_str}'")

    final_text = tokenizer.decode(generated[0])
    print("\n=== FINAL GENERATED TEXT ===")
    print(final_text)
    print("============================\n")

if __name__ == "__main__":
    test_generation()

