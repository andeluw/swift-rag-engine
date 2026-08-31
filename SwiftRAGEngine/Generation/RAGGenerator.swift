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
            model: model,
            instructions: """
                Answer the user's question using only the provided context.

                If the context contains enough information:
                - Answer clearly and concisely.
                - Set hasSufficientContext to true.

                If the context does not contain enough information:
                - Set hasSufficientContext to false.
                - Explain briefly that the provided context is insufficient.

                Do not use outside knowledge.
                """
        )
    }
    
    func generate(
        question: String,
        context: String
    ) async throws -> RAGResponse {
        try checkAvailability()
        
        let prompt = """
            Context:
            
            \(context)
            
            Question:
            \(question)
            """
        
        let response = try await session.respond(
            to: prompt,
            generating: RAGResponse.self
        )
        
        return response.content
    }
    
    private func checkAvailability() throws {
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
    }
}

@Generable
struct RAGResponse {
    let answer: String
    let hasSufficientContext: Bool
    
    var displayAnswer: String {
        if hasSufficientContext {
            return answer
        }
        
        return "The provided documents don't contain enough information to answer this question."
    }
}

enum RAGGeneratorError: Error {
    case appleIntelligenceNotEnabled
    case deviceNotEligible
    case modelNotReady
    case modelUnavailable
}
