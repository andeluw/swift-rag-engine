# SwiftRAGEngine

A local Retrieval-Augmented Generation (RAG) prototype for macOS, written in Swift.

Documents are chunked, embedded, retrieved, and used as context for Apple's on-device Foundation Models. Once the embedding model assets are downloaded, document indexing, retrieval, and generation run locally on the Mac.

The repository includes a small SwiftUI chat app over four bundled sample documents.

## Requirements

- macOS 26.5+ (deployment target)
- Xcode 26+
- Apple silicon Mac with Apple Intelligence enabled
- Swift package dependencies resolved by Xcode:
  - [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM)
  - [swift-transformers](https://github.com/huggingface/swift-transformers)
  - [swift-huggingface](https://github.com/huggingface/swift-huggingface)
  - [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

Apple Foundation Models are used for answer generation, so the app requires Foundation Models availability on the device.

## Run

Clone the repository and open the Xcode project:

```sh
git clone https://github.com/andeluw/swift-rag-engine.git
cd swift-rag-engine
open SwiftRAGEngine.xcodeproj
```

Build and run the app.

On first launch, the default `EmbeddingGemmaProvider` downloads and loads the EmbeddingGemma model into the user's Application Support directory. The app then indexes four bundled sample documents:

- Wi-Fi
- SwiftUI
- Nutrition
- Database

After indexing, the chat is ready for questions.

Try:

> Why does my Wi-Fi keep disconnecting?

You can also ask something the bundled documents do not cover:

> What is the capital of France?

If retrieval does not find sufficiently relevant context, the engine returns an insufficient-information response without calling the language model.

## How It Works

```text
.txt files
    │
    ▼
TextLoader
    │
    ▼
TextChunker
    │
    ▼
EmbeddingProvider
    │
    ▼
[EmbeddedChunk]
    │
    │
question ──▶ EmbeddingProvider
    │
    ▼
VectorSearch
cosine similarity + Top-K
    │
    ▼
RetrievalPolicy
minimum-score gate
    │
    ├── insufficient ──▶ fixed response
    │
    ▼
ContextBuilder
    │
    ▼
RAGGenerator
Apple Foundation Models
    │
    ▼
Grounded Answer
```

| Stage | Type | Notes |
| --- | --- | --- |
| Load | `TextLoader` | Loads `.txt` documents through the `DocumentLoader` abstraction |
| Chunk | `TextChunker` | Paragraph-first chunking with sentence and word fallback |
| Embed | `EmbeddingProvider` | EmbeddingGemma by default, with BGE as an alternative |
| Search | `VectorSearch` | Brute-force cosine similarity over indexed chunks |
| Gate | `RetrievalPolicy` | Rejects retrieval results below the configured minimum score |
| Context | `ContextBuilder` | Builds generation context from retrieved chunks |
| Generate | `RAGGenerator` | Uses Apple's on-device Foundation Models |

`RAGEngine` wires the pipeline together, while `RAGViewModel` drives the SwiftUI chat and exposes retrieved chunks as answer sources.

## Retrieval Gating

Top-K vector search always returns the closest results, even when none of them are actually relevant.

SwiftRAGEngine adds a retrieval gate before generation. If the highest cosine similarity score is below `RetrievalPolicy.minimumScore`, the engine returns:

> The provided documents don't contain enough information to answer this question.

The language model is not called in this case.

The sample app currently uses a minimum score of `0.30`. This value is corpus- and model-dependent and should be evaluated rather than treated as a universal threshold.

## Configuration

The main pipeline components are constructor-injected.

The sample app configures the engine in `RAGViewModel.prepare()`:

```swift
var engine = RAGEngine(
    embedder: try await EmbeddingGemmaProvider(),
    tokenCounter: try await EmbeddingGemmaTokenCounter(),
    retrievalPolicy: RetrievalPolicy(minimumScore: 0.30),
    targetTokens: 100,
    topK: 3
)
```

This makes individual parts of the pipeline replaceable:

- To use different documents, replace the sample-loading flow in `RAGViewModel`.
- To support another document format, implement `DocumentLoader`.
- To use another embedding model, implement `EmbeddingProvider` and a matching `TokenCounter`.

## Embedding Models

### EmbeddingGemma

EmbeddingGemma is the default embedding provider.

- 768-dimensional embeddings
- Separate retrieval-query and retrieval-document tasks
- Downloaded and loaded at runtime through CoreML-LLM
- No manual model conversion required for the default app flow

### BGE-small-en-v1.5

The repository also contains an experimental BGE-small-en-v1.5 Core ML integration.

- 384-dimensional embeddings
- 128-token sequence length
- Requires local Core ML conversion before use

Create the model package:

```sh
python3 -m venv .venv
source .venv/bin/activate

pip install coremltools torch sentence-transformers

python Tools/Embedding/BGE/convert_coreml.py
```

The conversion script writes the Core ML model to:

```text
BuildArtifacts/CoreML/BGE.mlpackage
```

Copy the package into:

```text
SwiftRAGEngine/Resources/MLModels/
```

and add it to the Xcode target so Core ML generates the `BGE` model class.

Then swap the embedding provider and token counter:

```swift
let embedder: any EmbeddingProvider =
    try await BGEEmbeddingProvider()

let tokenCounter: any TokenCounter =
    try await BGETokenCounter()
```

Validation scripts under `Tools/Embedding/` can be used to compare converted-model outputs with their Python reference implementations.

## Limitations

- `.txt` documents only
- In-memory index rebuilt on every launch
- Brute-force vector search
- macOS only
