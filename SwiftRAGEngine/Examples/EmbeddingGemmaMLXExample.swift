//
//  EmbeddingGemmaMLXExample.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 27/08/26.
//

import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

enum EmbeddingGemmaMLXExample {
    static func run() async throws {
        let configuration = ModelConfiguration(
            id: "mlx-community/embeddinggemma-300m-8bit"
        )

        // Load EmbeddingGemma
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration
        )

        let query = "Why does my Wi-Fi keep disconnecting?"

        let documents = [
            "Your wireless connection may drop because of weak signal or router interference.",
            "Restart the router and move closer to improve Wi-Fi stability.",
            "Bananas are a good source of potassium.",
            "SwiftUI uses a declarative approach to build user interfaces.",
        ]

        // Embed query
        let queryEmbedding = await embed(
            "task: search result | query: " + query,
            using: container
        )

        var results: [(document: String, similarity: Float)] = []

        // Embed and compare documents
        for document in documents {
            let documentEmbedding = await embed(
                "title: none | text: " + document,
                using: container
            )

            let similarity = CosineSimilarity.calculate(
                queryEmbedding,
                documentEmbedding
            )

            results.append(
                (
                    document: document,
                    similarity: similarity
                )
            )
        }

        results.sort {
            $0.similarity > $1.similarity
        }

        print("EmbeddingGemma MLX")
        print("Embedding shape:", queryEmbedding.count)
        print("Query:", query)
        print()

        for result in results {
            print(
                result.similarity,
                result.document
            )
        }
    }

    private static func embed(
        _ text: String,
        using container: EmbedderModelContainer
    ) async -> [Float] {
        return await container.perform { context -> [Float] in
            let tokenIDs = context.tokenizer.encode(
                text: text
            )

            let input = MLXArray(tokenIDs)
                .expandedDimensions(axis: 0)

            let output = context.model(
                input,
                positionIds: nil,
                tokenTypeIds: nil,
                attentionMask: nil
            )

            guard let embedding = output.pooledOutput else {
                return []
            }

            eval(embedding)

            return embedding[0]
                .asArray(Float.self)
        }
    }
}
