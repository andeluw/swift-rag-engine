from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-300m"

model = SentenceTransformer(MODEL_ID)

query = "Why does my Wi-Fi keep disconnecting?"

documents = [
    "Your wireless connection may drop because of weak signal or router interference.",
    "Restart the router and move closer to improve Wi-Fi stability.",
    "Bananas are a good source of potassium.",
    "SwiftUI uses a declarative approach to build user interfaces.",
]

# Embed query and documents
query_embedding = model.encode_query(query)
document_embeddings = model.encode_document(documents)

# Compare
similarities = model.similarity(
    query_embedding,
    document_embeddings
)[0]

results = list(zip(documents, similarities.tolist()))
results.sort(key=lambda result: result[1], reverse=True)

print("EmbeddingGemma Hugging Face")
print("Embedding shape:", query_embedding.shape)
print("Query:", query)
print()

for document, similarity in results:
    print(similarity, document)