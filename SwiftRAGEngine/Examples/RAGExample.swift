//
//  RAGExample.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 29/08/26.
//

import Foundation

enum RAGExample {
    static func run() async throws {
        let documentText = """
            Wi-Fi connections can become unstable when a device is far from the router. Thick walls, floors, and other physical obstacles can weaken the wireless signal. Interference from nearby routers and electronic devices may also cause frequent disconnections. Moving closer to the router or changing the wireless channel can improve connection stability.

            SwiftUI is Apple's declarative framework for building user interfaces. Developers describe how the interface should look based on application state, and SwiftUI automatically updates the displayed views when that state changes. It supports reusable views, navigation, animations, and integration with other Apple platform frameworks.

            Bananas contain potassium, carbohydrates, fiber, and several vitamins. They are commonly eaten as a snack or included in smoothies and desserts. Potassium contributes to normal muscle and nerve function, while carbohydrates provide energy for daily activities.

            A relational database stores information in tables consisting of rows and columns. SQL can be used to query, insert, update, and delete records. Database indexes can improve query performance by allowing the database engine to locate relevant records without scanning every row in a table.
            """

        let query = "Why does my Wi-Fi keep disconnecting?"
        //        let query = "How does SwiftUI update the interface?"
        //        let query = "What nutrients are found in bananas?"
        //        let query = "How can database indexes improve performance?"

        //        let query = "What is capital city of France?"

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

        let chunks = chunker.chunk(
            text: documentText,
            source: "sample.txt"
        )

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
        print(response.answer)
        print("Sufficient Context:", response.hasSufficientContext)

    }
}
