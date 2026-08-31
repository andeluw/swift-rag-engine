//
//  BGEEmbeddingProvider.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct BGEEmbeddingProvider: EmbeddingProvider {
    private let tokenizer: BGETokenizer
    private let model: BGEEmbeddingModel

    init() async throws {
        tokenizer = try await BGETokenizer()
        model = try BGEEmbeddingModel()
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        let instruction =
            "Represent this sentence for searching relevant passages: "

        let input = tokenizer.encode(instruction + text)

        return try model.embed(
            inputIDs: input.inputIDs,
            attentionMask: input.attentionMask
        )
    }

    func embedDocument(_ text: String) async throws -> [Float] {
        let input = tokenizer.encode(text)

        return try model.embed(
            inputIDs: input.inputIDs,
            attentionMask: input.attentionMask
        )
    }
}
