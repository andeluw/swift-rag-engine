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
    
    init() {
        model = SystemLanguageModel.default
    }
    
    func generate(
        question: String,
        context: String
    ) async throws -> GeneratedAnswer {
        try checkAvailability()
        
        let session = LanguageModelSession(
            model: model,
            instructions: """
                Answer the user's question using only the provided context.

                - Answer clearly and concisely.
                - Do not use outside knowledge.
                - Do not include information that is not supported by the context.

                If the provided context does not support an answer, state that the context is insufficient.
                """
        )
        
        let prompt = """
            Context:
            
            \(context)
            
            Question:
            \(question)
            """
        
        let response = try await session.respond(
            to: prompt,
            generating: GeneratedAnswer.self
        )
        
        return response.content
    }
    
    private func checkAvailability() throws {
        switch model.availability {
        case .available:
            break
            
        case .unavailable(.appleIntelligenceNotEnabled):
            throw RAGGeneratorError.appleIntelligenceNotEnabled
            
        case .unavailable(.deviceNotEligible):
            throw RAGGeneratorError.deviceNotEligible
            
        case .unavailable(.modelNotReady):
            throw RAGGeneratorError.modelNotReady
            
        case .unavailable:
            throw RAGGeneratorError.modelUnavailable
        }
    }
}

@Generable
struct GeneratedAnswer {
    let answer: String
}

enum RAGGeneratorError: Error {
    case appleIntelligenceNotEnabled
    case deviceNotEligible
    case modelNotReady
    case modelUnavailable
}
