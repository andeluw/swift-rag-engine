//
//  RAGExample.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 29/08/26.
//

import Foundation

enum RAGExample {
    static func run() async throws {
        //        let query = "Why does my Wi-Fi keep disconnecting?"
        //        let query = "How does SwiftUI update the interface?"
        //        let query = "What nutrients are found in bananas?"
        //        let query = "How can database indexes improve performance?"
        let query = "What is capital city of France?"

        // Load documents
        let documents = try loadSampleDocuments()

        // EmbeddingGemma
        let embedder: any EmbeddingProvider = try await EmbeddingGemmaProvider()
        let tokenCounter: any TokenCounter =
            try await EmbeddingGemmaTokenCounter()

        // BGE
        //        let embedder: any EmbeddingProvider = try await BGEEmbeddingProvider()
        //        let tokenCounter: any TokenCounter = try await BGETokenCounter()

        // Chunk document
        let chunker = TextChunker(
            targetTokens: 100,
            maxTokens: tokenCounter.maxTokens,
            tokenCount: { tokenCounter.count($0) }
        )

        var chunks: [DocumentChunk] = []

        for document in documents {
            chunks.append(
                contentsOf: chunker.chunk(
                    text: document.text,
                    source: document.source
                )
            )
        }

        // Embed chunks
        var embeddedChunks: [EmbeddedChunk] = []

        for chunk in chunks {
            let embedding = try await embedder.embedDocument(chunk.text)

            embeddedChunks.append(
                EmbeddedChunk(chunk: chunk, embedding: embedding)
            )
        }

        // Embed query
        let queryEmbedding = try await embedder.embedQuery(query)

        // Retrieve Top-K
        let results = VectorSearch.search(
            queryEmbedding: queryEmbedding,
            chunks: embeddedChunks,
            topK: 3
        )

        // Build context
        let context = ContextBuilder.build(from: results)

        // Generate answer
        let generator = RAGGenerator()

        let response = try await generator.generate(
            question: query,
            context: context
        )

        // Output
        printResults(
            query: query,
            chunks: chunks,
            embeddedChunks: embeddedChunks,
            results: results,
            response: response,
            tokenCount: { tokenCounter.count($0) }
        )

    }

    private static func loadSampleDocuments() throws -> [LoadedDocument] {
        let loader = TextLoader()

        let sampleNames = [
            "wifi",
            "swiftui",
            "nutrition",
            "database",
        ]

        return try sampleNames.map { name in
            let url =
                Bundle.main.url(
                    forResource: name,
                    withExtension: "txt",
                    subdirectory: "Samples"
                )
                ?? Bundle.main.url(
                    forResource: name,
                    withExtension: "txt"
                )

            guard let url else {
                throw DocumentLoaderError.fileNotFound
            }

            return try loader.load(from: url)
        }
    }

    private static func printResults(
        query: String,
        chunks: [DocumentChunk],
        embeddedChunks: [EmbeddedChunk],
        results: [SearchResult],
        response: RAGResponse,
        tokenCount: (String) -> Int
    ) {
        print("Chunks")
        print()

        for (index, chunk) in chunks.enumerated() {
            print("Chunk \(index)")
            print("Tokens:", tokenCount(chunk.text))
            print(chunk.text)
            print("---")
        }

        for (index, item) in embeddedChunks.enumerated() {
            print("Embedded Chunk \(index)")
            print("Dimension:", item.embedding.count)
            print("Source:", item.chunk.source)
            print("---")
        }

        print()
        print("Query:", query)
        print()

        for (index, result) in results.enumerated() {
            print("Result \(index + 1)")
            print("Score:", result.score)
            print(result.chunk.text)
            print("---")
        }

        print()
        print("Generated Answer")
        print(response.displayAnswer)
        print("Sufficient Context:", response.hasSufficientContext)

    }
}
