from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

MAX_SEQUENCE_LENGTH = 128

OUTPUT_PATH = Path(
    "BuildArtifacts/CoreML/BGE.mlpackage"
)


class BGECoreML(nn.Module):
    def __init__(self, sentence_model: SentenceTransformer):
        super().__init__()

        # SentenceTransformer module 0:
        # BERT Transformer backbone
        self.transformer = sentence_model[0].auto_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # BERT token type IDs -> Single sequence, so all zeros
        token_type_ids = torch.zeros_like(input_ids)

        # Transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        token_embeddings = outputs[0]

        # CLS pooling -> Take first token embedding
        embedding = token_embeddings[:, 0]

        # L2 normalization
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding


def main():
    # Load BGE
    print(f"Loading model: {MODEL_ID}")

    sentence_model = SentenceTransformer(
        MODEL_ID,
        device="cpu",
        model_kwargs={
            "attn_implementation": "eager",
        }
    )

    sentence_model.eval()

    model = BGECoreML(sentence_model)
    model.eval()

    # Create representative input for tracing
    query = "Why does my Wi-Fi keep disconnecting?"
    formatted_query = QUERY_INSTRUCTION + query

    inputs = sentence_model.tokenizer(
        formatted_query,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    input_ids = inputs["input_ids"].to(torch.int32)
    attention_mask = inputs["attention_mask"].to(torch.int32)

    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")

    # Run SentenceTransformer reference
    reference_embedding = sentence_model.encode(
        formatted_query,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    print(f"reference embedding shape: {reference_embedding.shape}")

    # Run PyTorch wrapper
    with torch.no_grad():
        embedding = model(
            input_ids,
            attention_mask,
        )[0]

    print(f"embedding shape: {embedding.shape}")
    print(f"L2 norm: {torch.linalg.vector_norm(embedding)}")

    # Compare wrapper with SentenceTransformer
    cosine_similarity = F.cosine_similarity(
        reference_embedding.unsqueeze(0),
        embedding.unsqueeze(0),
    )[0]

    max_absolute_difference = (
        reference_embedding - embedding
    ).abs().max()

    print(f"cosine similarity: {float(cosine_similarity)}")
    print(f"max absolute difference: {float(max_absolute_difference)}")

    # Trace PyTorch model
    print("\nTracing PyTorch model...")

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (
                input_ids,
                attention_mask,
            ),
        )

    print("Tracing complete.")

    # Convert to Core ML
    print("\nConverting to Core ML...")

    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, MAX_SEQUENCE_LENGTH),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, MAX_SEQUENCE_LENGTH),
                dtype=np.int32,
            )
        ],
        outputs=[
            ct.TensorType(
                name="embedding",
                dtype=np.float32,
            )
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS26,
    )

    print("Core ML conversion complete.")

    # Run Core ML
    prediction = coreml_model.predict(
        {
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.int32),
        }
    )

    coreml_embedding = np.asarray(
        prediction["embedding"]
    ).reshape(-1)

    reference_embedding = (
        reference_embedding
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # Compare Core ML with SentenceTransformer
    cosine_similarity = np.dot(
        reference_embedding,
        coreml_embedding,
    ) / (
        np.linalg.norm(reference_embedding)
        * np.linalg.norm(coreml_embedding)
    )

    max_absolute_difference = np.max(
        np.abs(
            reference_embedding - coreml_embedding
        )
    )

    print(f"Core ML embedding shape: {coreml_embedding.shape}")
    print(f"Core ML L2 norm: {np.linalg.norm(coreml_embedding)}")
    print(f"Core ML cosine similarity: {float(cosine_similarity)}")
    print(f"Core ML max absolute difference: {float(max_absolute_difference)}")

    # Save Core ML model
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coreml_model.save(str(OUTPUT_PATH))

    print(f"Core ML model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()