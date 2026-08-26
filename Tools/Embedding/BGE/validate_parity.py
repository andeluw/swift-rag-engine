import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


def main():
    query = "Why does my Wi-Fi keep disconnecting?"
    text = QUERY_INSTRUCTION + query

    # SentenceTransformer pipeline
    sentence_model = SentenceTransformer(
        MODEL_ID,
        device="cpu",
    )

    reference_embedding = sentence_model.encode(
        text,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    print(
        "SentenceTransformer embedding shape:",
        reference_embedding.shape,
    )

    # Manual pipeline using AutoTokenizer and AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    transformer = AutoModel.from_pretrained(MODEL_ID)
    transformer.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = transformer(**inputs)

    # CLS pooling
    manual_embedding = outputs.last_hidden_state[:, 0]

    # L2 normalization
    manual_embedding = F.normalize(
        manual_embedding,
        p=2,
        dim=1,
    )

    # Remove batch dimension:
    # [1, 384] -> [384]
    manual_embedding = manual_embedding[0]

    print(
        "Manual embedding shape:",
        manual_embedding.shape,
    )

    # Compare the embeddings
    cosine_similarity = F.cosine_similarity(
        reference_embedding.unsqueeze(0),
        manual_embedding.unsqueeze(0),
    )

    max_absolute_difference = (
        reference_embedding - manual_embedding
    ).abs().max()

    print(
        "\nCosine similarity:",
        float(cosine_similarity[0]),
    )

    print(
        "Max absolute difference:",
        float(max_absolute_difference),
    )


if __name__ == "__main__":
    main()