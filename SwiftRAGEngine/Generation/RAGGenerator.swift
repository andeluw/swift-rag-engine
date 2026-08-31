//
//  RAGGenerator.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import Foundation
import FoundationModels

struct RAGGenerator {
    private let model: SystemLanguageModel
    private let session: LanguageModelSession

    init() {
        let model = SystemLanguageModel.default
        
        self.model = model
        self.session = LanguageModelSession(
            instructions: """
                Answer the user's question using only the provided context.

                If the context does not contain enough information to answer,
                say that the provided context is insufficient.

                Do not use outside knowledge.
                """
        )
    }

    func generate(
        question: String,
        context: String
    ) async throws -> String {
        switch model.availability {
        case .available:
            print("Foundation Model available")

        case .unavailable(.appleIntelligenceNotEnabled):
            throw RAGGeneratorError.appleIntelligenceNotEnabled

        case .unavailable(.deviceNotEligible):
            throw RAGGeneratorError.deviceNotEligible

        case .unavailable(.modelNotReady):
            throw RAGGeneratorError.modelNotReady

        case .unavailable:
            throw RAGGeneratorError.modelUnavailable
        }

        let prompt = """
            Context:

            \(context)

            Question:
            \(question)
            """

        let response = try await session.respond(
            to: prompt
        )

        return response.content
    }
}

enum RAGGeneratorError: Error {
    case appleIntelligenceNotEnabled
    case deviceNotEligible
    case modelNotReady
    case modelUnavailable
}
