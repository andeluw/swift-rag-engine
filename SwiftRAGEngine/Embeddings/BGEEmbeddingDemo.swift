//
//  BGEEmbeddingDemo.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 27/08/26.
//

import Foundation

enum BGEEmbeddingDemo {
    static func run() async throws {
        let tokenizer = try await BGETokenizer()
        let model = try BGEEmbeddingModel()

        let queryInstruction =
            "Represent this sentence for searching relevant passages: "
        let query = "Why does my Wi-Fi keep disconnecting?"

        let documents = [
            "Your wireless connection may drop because of weak signal or router interference.",
            "Restart the router and move closer to improve Wi-Fi stability.",
            "Bananas are a good source of potassium.",
            "SwiftUI uses a declarative approach to build user interfaces.",
        ]

        // Embed query
        let queryInput = tokenizer.encode(
            queryInstruction + query
        )

        let queryEmbedding = try model.embed(
            inputIDs: queryInput.inputIDs,
            attentionMask: queryInput.attentionMask
        )

        var results: [(document: String, similarity: Float)] = []

        // Compare documents
        for document in documents {
            let documentInput = tokenizer.encode(document)

            let documentEmbedding = try model.embed(
                inputIDs: documentInput.inputIDs,
                attentionMask: documentInput.attentionMask
            )

            let similarity = cosineSimilarity(
                queryEmbedding,
                documentEmbedding
            )

            results.append(
                (document: document, similarity: similarity)
            )
        }

        results.sort {
            $0.similarity > $1.similarity
        }

        print("Query:", query)
        print()

        for result in results {
            print(result.similarity, result.document)
        }
    }

    private static func cosineSimilarity(
        _ a: [Float],
        _ b: [Float]
    ) -> Float {
        let dotProduct = zip(a, b).reduce(Float(0)) {
            $0 + ($1.0 * $1.1)
        }

        let normA = sqrt(
            a.reduce(Float(0)) {
                $0 + ($1 * $1)
            }
        )

        let normB = sqrt(
            b.reduce(Float(0)) {
                $0 + ($1 * $1)
            }
        )

        return dotProduct / (normA * normB)
    }
}
