from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

# Load BGE
model = SentenceTransformer(
    MODEL_ID,
    device="cpu",
)

query = "Why does my Wi-Fi keep disconnecting?"
formatted_query = QUERY_INSTRUCTION + query

# Generate Python reference embedding
embedding = model.encode(
    formatted_query,
    normalize_embeddings=True,
)

print("Embedding shape:", embedding.shape)
print("L2 norm:", (embedding @ embedding) ** 0.5)
print("First 10 values:", embedding[:10].tolist())