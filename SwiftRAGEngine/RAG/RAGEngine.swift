//
//  RAGEngine.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct RAGEngine {
    private let embedder: any EmbeddingProvider
    private let chunker: TextChunker
    private let generator: RAGGenerator
    private let retrievalPolicy: RetrievalPolicy
    private let topK: Int

    private(set) var chunks: [DocumentChunk] = []
    private(set) var embeddedChunks: [EmbeddedChunk] = []

    init(
        embedder: any EmbeddingProvider,
        tokenCounter: any TokenCounter,
        retrievalPolicy: RetrievalPolicy,
        generator: RAGGenerator = RAGGenerator(),
        targetTokens: Int = 100,
        topK: Int = 3
    ) {
        self.embedder = embedder
        self.generator = generator
        self.retrievalPolicy = retrievalPolicy
        self.topK = topK

        self.chunker = TextChunker(
            targetTokens: targetTokens,
            maxTokens: tokenCounter.maxTokens,
            tokenCount: { tokenCounter.count($0) }
        )
    }

    mutating func index(
        documents: [LoadedDocument]
    ) async throws {
        var newChunks: [DocumentChunk] = []

        for document in documents {
            newChunks.append(
                contentsOf: chunker.chunk(
                    text: document.text,
                    source: document.source
                )
            )
        }

        var newEmbeddedChunks: [EmbeddedChunk] = []
        newEmbeddedChunks.reserveCapacity(newChunks.count)

        for chunk in newChunks {
            let embedding = try await embedder.embedDocument(chunk.text)

            newEmbeddedChunks.append(
                EmbeddedChunk(chunk: chunk, embedding: embedding)
            )
        }
        
        chunks = newChunks
        embeddedChunks = newEmbeddedChunks
    }

    func ask(
        _ query: String
    ) async throws -> RAGResult {
        guard !embeddedChunks.isEmpty else {
            throw RAGEngineError.notIndexed
        }

        // Embed query
        let queryEmbedding = try await embedder.embedQuery(query)

        // Retrieve Top-K
        let results = VectorSearch.search(
            queryEmbedding: queryEmbedding,
            chunks: embeddedChunks,
            topK: topK
        )

        // Check retrieval relevance
        guard retrievalPolicy.hasSufficientContext(results) else {
            return RAGResult(
                answer:
                    "The provided documents don't contain enough information to answer this question.",
                hasSufficientContext: false,
                retrievedResults: results
            )
        }

        // Build context
        let context = ContextBuilder.build(from: results)

        // Generate answer
        let generatedAnswer = try await generator.generate(
            question: query,
            context: context
        )

        return RAGResult(
            answer: generatedAnswer.answer,
            hasSufficientContext: true,
            retrievedResults: results
        )
    }
}

enum RAGEngineError: Error {
    case notIndexed
}
