//
//  RAGViewModel.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import Foundation
import Observation

@MainActor
@Observable
final class RAGViewModel {
    var input = ""
    
    private(set) var messages: [ChatMessage] = []
    private(set) var isIndexing = false
    private(set) var isGenerating = false
    private(set) var errorMessage: String?
    private(set) var documentCount = 0

    private var engine: RAGEngine?

    var isReady: Bool {
        engine != nil && !isIndexing
    }

    func dismissError() {
        errorMessage = nil
    }

    func prepare() async {
        guard engine == nil, !isIndexing else {
            return
        }

        isIndexing = true
        errorMessage = nil

        defer {
            isIndexing = false
        }

        do {
            let documents = try loadSampleDocuments()

            let embedder: any EmbeddingProvider =
                try await EmbeddingGemmaProvider()

            let tokenCounter: any TokenCounter =
                try await EmbeddingGemmaTokenCounter()

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

            self.engine = engine
            documentCount = documents.count
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func send() async {
        let question = input.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !question.isEmpty, !isGenerating, let engine else {
            return
        }

        input = ""
        errorMessage = nil

        messages.append(
            ChatMessage(
                role: .user,
                text: question,
                sources: []
            )
        )

        isGenerating = true

        defer {
            isGenerating = false
        }

        do {
            let result = try await engine.ask(question)

            let sources: [ChatSource]

            if result.hasSufficientContext {
                sources = result.retrievedResults.map { searchResult in
                    ChatSource(
                        id: searchResult.chunk.id,
                        name: searchResult.chunk.source,
                        score: searchResult.score,
                        text: searchResult.chunk.text
                    )
                }
            } else {
                sources = []
            }

            messages.append(
                ChatMessage(
                    role: .assistant,
                    text: result.answer,
                    sources: sources
                )
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func loadSampleDocuments() throws -> [LoadedDocument] {
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
}
