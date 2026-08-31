//
//  EmbeddingGemmaProvider.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import CoreMLLLM
import Foundation

struct EmbeddingGemmaProvider: EmbeddingProvider {
    private let model: EmbeddingGemma

    init() async throws {
        let modelsDirectory = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]

        model = try await EmbeddingGemma.downloadAndLoad(
            modelsDir: modelsDirectory
        )
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try model.encode(
            text: text,
            task: .retrievalQuery,
            dim: 768
        )
    }

    func embedDocument(_ text: String) async throws -> [Float] {
        try model.encode(
            text: text,
            task: .retrievalDocument,
            dim: 768
        )
    }
}
