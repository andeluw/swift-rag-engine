from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

def main():
    model = SentenceTransformer(MODEL_ID)

    query = "Why does my Wi-Fi keep disconnecting?"

    documents = [
        "Wireless interference can cause an unstable Wi-Fi connection.",
        "The router includes a two-year hardware warranty.",
        "Restarting the router can resolve temporary connectivity problems.",
    ]

    query_embedding = model.encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True
    )
    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True
    )

    print(f"Model: {MODEL_ID}")
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Document embeddings shape: {document_embeddings.shape}")

    # similarities = document_embeddings @ query_embedding # matrix multiplication
    similarities = model.similarity(query_embedding, document_embeddings)[0]

    print("\nSimilarity scores:")
    for document, score in zip(documents, similarities):
        print(f"{float(score):.4f} - {document}")

    # ranking = similarities.argsort()[::-1]
    ranking = similarities.argsort(descending=True)

    print("\nRanking:")
    for rank, index in enumerate(ranking, start=1):
        index = int(index)

        print(
            f"{rank}. "
            f"{float(similarities[index]):.4f} "
            f"{documents[index]}"
        )

if __name__ == "__main__":
    main()