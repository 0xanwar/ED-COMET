from transformers import BartForConditionalGeneration, BartTokenizer

# Load model and tokenizer
model = BartForConditionalGeneration.from_pretrained("./checkpoints/bart_hf")
tokenizer = BartTokenizer.from_pretrained("./checkpoints/bart_hf")

print(f"✓ Model vocab size: {model.config.vocab_size}")
print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
print(f"✓ Match: {model.config.vocab_size == len(tokenizer)}")

# Test tokenization
text = "The capital of Egypt is Cairo"
tokens = tokenizer.encode(text)
print(f"\n✓ Test tokenization: {tokens[:10]}")
