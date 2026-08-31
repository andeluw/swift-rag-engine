//
//  RAGExample.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 29/08/26.
//

import Foundation

enum RAGExample {
    static func run() async throws {
        let query = "Why does my Wi-Fi keep disconnecting?"
        //        let query = "How does SwiftUI update the interface?"
        //        let query = "What nutrients are found in bananas?"
        //        let query = "How can database indexes improve performance?"

        // Negative retrieval test
        //                let query = "What is capital city of France?"

        // Load documents
        let documents = try loadSampleDocuments()

        // EmbeddingGemma
        let embedder: any EmbeddingProvider = try await EmbeddingGemmaProvider()
        let tokenCounter: any TokenCounter =
            try await EmbeddingGemmaTokenCounter()

        // BGE
        //        let embedder: any EmbeddingProvider = try await BGEEmbeddingProvider()
        //        let tokenCounter: any TokenCounter = try await BGETokenCounter()

        let retrievalPolicy = RetrievalPolicy(minimumScore: 0.30)

        var engine = RAGEngine(
            embedder: embedder,
            tokenCounter: tokenCounter,
            retrievalPolicy: retrievalPolicy,
            targetTokens: 100,
            topK: 3
        )

        // Index documents
        try await engine.index(documents: documents)

        // Ask
        let result = try await engine.ask(query)

        // Output
        printResults(
            query: query,
            chunks: engine.chunks,
            embeddedChunks: engine.embeddedChunks,
            result: result,
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
        result: RAGResult,
        tokenCount: (String) -> Int
    ) {
        print("Chunks")
        print()

        for (index, chunk) in chunks.enumerated() {
            print("Chunk \(index)")
            print("Source:", chunk.source)
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

        for (index, searchResult) in result.retrievedResults.enumerated() {
            print("Result \(index + 1)")
            print("Score:", searchResult.score)
            print("Source:", searchResult.chunk.source)
            print(searchResult.chunk.text)
            print("---")
        }

        print()
        print("Generated Answer")
        print(result.answer)
        print("Sufficient Context:", result.hasSufficientContext)

    }
}
