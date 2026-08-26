from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-300m"

MAX_SEQUENCE_LENGTH = 128

OUTPUT_PATH = Path(
  "BuildArtifacts/CoreML/EmbeddingGemma.mlpackage"
)

class EmbeddingGemmaCoreML(nn.Module):
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
            use_cache=False,
            return_dict=False,
        )

        token_embeddings = outputs[0]

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
            dim=-1,
        )

        return embedding

def main():
    # Load Google's EmbeddingGemma
    print(f"Loading model: {MODEL_ID}")

    sentence_model = SentenceTransformer(
        MODEL_ID, 
        device="cpu",
        model_kwargs={
            "attn_implementation": "eager",
        }
    )

    sentence_model.eval()

    model = EmbeddingGemmaCoreML(sentence_model)
    model.eval()

    # Create representative input for tracing
    query = "Why does my Wi-Fi keep disconnecting?"
    query_prompt = sentence_model.prompts["Retrieval-query"]
    formatted_query = query_prompt + query

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

    # Run PyTorch
    with torch.no_grad():
        embedding = model(input_ids, attention_mask)

    print(f"embedding shape: {embedding.shape}")
    print(f"L2 norm: {torch.linalg.vector_norm(embedding, dim=1)}")

    # Tracing PyTorch model
    # with torch.no_grad():
    #     traced_model = torch.jit.trace(
    #         model,
    #         (input_ids, attention_mask),
    #     )

    # print("Tracing complete.")

    # Export PyTorch model
    print("\nExporting PyTorch model...")

    with torch.no_grad():
        exported_model = torch.export.export(
            model,
            (
                input_ids,
                attention_mask,
            ),
        )

    print("Export complete.")

    # Convert to Core ML
    print("\nConverting to Core ML...")

    coreml_model = ct.convert(
        exported_model,
        convert_to="mlprogram",
        # inputs=[
        #     ct.TensorType(
        #         name="input_ids",
        #         shape=(1, MAX_SEQUENCE_LENGTH),
        #         dtype=np.int32,
        #     ),
        #     ct.TensorType(
        #         name="attention_mask",
        #         shape=(1, MAX_SEQUENCE_LENGTH),
        #         dtype=np.int32,
        #     )
        # ],
        # outputs=[
        #     ct.TensorType(name="embedding")
        # ],
        minimum_deployment_target=ct.target.iOS26
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    coreml_model.save(str(OUTPUT_PATH))
    print(f"Core ML model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()