from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

model = SentenceTransformer(
    MODEL_ID,
    device="cpu",
)

query = "Why does my Wi-Fi keep disconnecting?"
formatted_query = QUERY_INSTRUCTION + query

inputs = model.tokenizer(
    formatted_query,
    return_tensors="pt",
)

print("Token IDs:", inputs["input_ids"][0].tolist())
print("Token count:", len(inputs["input_ids"][0]))