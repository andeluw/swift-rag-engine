import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer


MODEL_ID = "google/embeddinggemma-300m"


class EmbeddingGemmaWrapper(nn.Module):
    def __init__(self, sentence_model: SentenceTransformer):
        super().__init__()

        # SentenceTransformer module 0:
        # Gemma3TextModel Transformer backbone
        self.transformer = sentence_model[0].auto_model

        # SentenceTransformer modules 2 and 3:
        # Dense 768 -> 3072 -> 768
        self.dense_1 = sentence_model[2].linear
        self.dense_2 = sentence_model[3].linear

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state

        # Mean pooling -> Ignore padding tokens using attention_mask
        expanded_mask = attention_mask.unsqueeze(-1).to(
            token_embeddings.dtype
        )

        summed_embeddings = (
            token_embeddings * expanded_mask
        ).sum(dim=1)

        token_count = expanded_mask.sum(dim=1).clamp(
            min=1e-9
        )

        embedding = summed_embeddings / token_count

        # Projection (768 -> 3072)
        embedding = self.dense_1(embedding)

        # Projection (3072 -> 768)
        embedding = self.dense_2(embedding)

        # L2 normalization
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return embedding


def main():
    sentence_model = SentenceTransformer(
        MODEL_ID,
        device="cpu",
    )

    sentence_model.eval()

    query = "Why does my Wi-Fi keep disconnecting?"

    # SentenceTransformer pipeline
    reference_embedding = sentence_model.encode(
        query,
        prompt_name="Retrieval-query",
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    print(
        "Reference embedding shape:",
        reference_embedding.shape,
    )

    # Prepare the exact retrieval-query input
    query_prompt = sentence_model.prompts[
        "Retrieval-query"
    ]

    formatted_query = query_prompt + query

    print("\nQuery prompt:")
    print(repr(query_prompt))

    print("\nFormatted query:")
    print(formatted_query)

    tokenizer = sentence_model.tokenizer

    inputs = tokenizer(
        formatted_query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=sentence_model.max_seq_length,
    )

    print(
        "\ninput_ids shape:",
        inputs["input_ids"].shape,
    )

    print(
        "attention_mask shape:",
        inputs["attention_mask"].shape,
    )

    # Custom PyTorch wrapper
    wrapper = EmbeddingGemmaWrapper(
        sentence_model
    )

    wrapper.eval()

    with torch.no_grad():
        wrapper_embedding = wrapper(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]

    print(
        "\nWrapper embedding shape:",
        wrapper_embedding.shape,
    )

    # Compare the embeddings
    cosine_similarity = F.cosine_similarity(
        reference_embedding.unsqueeze(0),
        wrapper_embedding.unsqueeze(0),
    )[0]

    max_absolute_difference = (
        reference_embedding - wrapper_embedding
    ).abs().max()

    print(
        "\nCosine similarity:",
        float(cosine_similarity),
    )

    print(
        "Max absolute difference:",
        float(max_absolute_difference),
    )


if __name__ == "__main__":
    main()