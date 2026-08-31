//
//  RAGDemo.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 29/08/26.
//

import Foundation

enum RAGDemo {
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

        let queryInstruction =
            "Represent this sentence for searching relevant passages: "

        let tokenizer = try await BGETokenizer()
        let model = try BGEEmbeddingModel()

        var embeddedChunks: [EmbeddedChunk] = []

        let chunker = TextChunker(
            targetTokens: 100,
            maxTokens: 128,
            tokenCount: tokenizer.tokenCount
        )

        let chunks = chunker.chunk(
            text: documentText,
            source: "sample.txt"
        )

        for chunk in chunks {
            let input = tokenizer.encode(chunk.text)

            let embedding = try model.embed(
                inputIDs:
                    input.inputIDs,
                attentionMask: input.attentionMask
            )

            embeddedChunks.append(
                EmbeddedChunk(chunk: chunk, embedding: embedding)
            )
        }

        let queryInput = tokenizer.encode(queryInstruction + query)

        let queryEmbedding = try model.embed(
            inputIDs: queryInput.inputIDs,
            attentionMask: queryInput.attentionMask
        )

        let results = VectorSearch.search(
            queryEmbedding: queryEmbedding,
            chunks: embeddedChunks,
            topK: 3
        )

        let context = ContextBuilder.build(from: results)

        let generator = RAGGenerator()

        let answer = try await generator.generate(question: query, context: context)

        /// Results

        for (index, chunk) in chunks.enumerated() {
            print("Chunk \(index)")
            print("Tokens:", tokenizer.tokenCount(chunk.text))
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
        print(answer)
    }
}
