from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-300m"

def main():
    model = SentenceTransformer(MODEL_ID)
    print(model)

    query = "Why does my Wi-Fi keep disconnecting?"

    documents = [
        "Wireless interference can cause an unstable Wi-Fi connection.",
        "The router includes a two-year hardware warranty.",
        "Restarting the router can resolve temporary connectivity problems.",
    ]

    query_embedding = model.encode_query(query)
    document_embeddings = model.encode_document(documents)

    print(f"Model: {MODEL_ID}")
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Document embeddings shape: {document_embeddings.shape}")

    similarities = model.similarity(query_embedding, document_embeddings)

    print("\nSimilarity scores:")
    for document, score in zip(documents, similarities[0]):
        print(f"{float(score):.4f} - {document}")

    ranking = similarities.argsort(descending=True)[0]

    print("\nRanking:")
    for rank, index in enumerate(ranking, start=1):
        index = int(index)

        print(
            f"{rank}. "
            f"{float(similarities[0][index]):.4f} "
            f"{documents[index]}"
        )

if __name__ == "__main__":
    main()