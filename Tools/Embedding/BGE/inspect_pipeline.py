import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

MODEL_ID = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    query = "Why does my Wi-Fi keep disconnecting?"
    text = QUERY_INSTRUCTION + query

    # Tokenization to convert text into tensors that the Transformer understand
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    print("input_ids shape:", inputs["input_ids"].shape)
    print("attention_mask shape:", inputs["attention_mask"].shape)

    print("\ninput_ids: ")
    print(inputs["input_ids"])

    # Run Transformer inference
    with torch.no_grad():
        output = model(**inputs)

    print("\nlast_hidden_state shape:", output.last_hidden_state.shape)

    # CLS pooling to get a single vector representation of the input text
    embedding = output.last_hidden_state[:, 0]

    print("\nCLS embedding shape:", embedding.shape)

    # L2 normalization to ensure the embedding has unit length
    embedding = F.normalize(embedding, p=2, dim=1)

    print("\nNormalized embedding shape:", embedding.shape)

    print("\nL2 norm:", torch.linalg.vector_norm(embedding, dim=1))

if __name__ == "__main__":
    main()