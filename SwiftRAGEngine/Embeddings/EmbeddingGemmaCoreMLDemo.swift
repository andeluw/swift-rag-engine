//
//  EmbeddingGemmaCoreMLDemo.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 27/08/26.
//

import CoreMLLLM
import Foundation

enum EmbeddingGemmaCoreMLDemo {
    static func run() async throws {
        let modelsDirectory = FileManager.default
            .urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            )[0]

        // Download and load EmbeddingGemma
        let model = try await EmbeddingGemma.downloadAndLoad(
            modelsDir: modelsDirectory
        )

        let query = "Why does my Wi-Fi keep disconnecting?"

        let documents = [
            "Your wireless connection may drop because of weak signal or router interference.",
            "Restart the router and move closer to improve Wi-Fi stability.",
            "Bananas are a good source of potassium.",
            "SwiftUI uses a declarative approach to build user interfaces.",
        ]

        // Embed query
        let queryEmbedding = try model.encode(
            text: query,
            task: .retrievalQuery,
            dim: 768
        )

        var results: [(document: String, similarity: Float)] = []

        // Embed and compare documents
        for document in documents {
            let documentEmbedding = try model.encode(
                text: document,
                task: .retrievalDocument,
                dim: 768
            )

            let similarity = CosineSimilarity.calculate(
                queryEmbedding,
                documentEmbedding
            )

            results.append((document: document, similarity: similarity))
        }
        
        results.sort {
            $0.similarity > $1.similarity
        }
        
        print("EmbeddingGemma Core ML")
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
}
