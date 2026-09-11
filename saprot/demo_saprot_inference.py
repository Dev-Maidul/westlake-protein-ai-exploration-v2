import torch
from transformers import EsmTokenizer, EsmForMaskedLM

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "westlake-repl/SaProt_35M_AF2"
print(f"Loading tokenizer and model from {model_path} ...")
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path).to(device)
model.eval()

# Lowercase letters = 3Di structure tokens, uppercase = amino acid tokens, '#' = masked/low-confidence
seq = "M#EvVpQpL#VyQdYaKv"
print(f"Input sequence: {seq}")

inputs = tokenizer(seq, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

print("Logits shape:", outputs.logits.shape)

input_ids = inputs["input_ids"][0]
input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print("\nTokenized input (each token = amino acid + 3Di structure char):")
print(input_tokens)

print("\nModel's top-1 predicted token at every position:")
for pos, tok in enumerate(input_tokens):
    top_id = outputs.logits[0, pos].argmax().item()
    top_token = tokenizer.convert_ids_to_tokens([top_id])[0]
    print(f"  pos {pos:2d} | input = {tok:6s} | predicted = {top_token}")
