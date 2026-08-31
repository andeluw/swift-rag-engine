//
//  BGETokenCounter.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct BGETokenCounter: TokenCounter {
    private let tokenizer: BGETokenizer
    
    var maxTokens: Int {
        tokenizer.maxTokens
    }
    
    init() async throws {
        tokenizer = try await BGETokenizer()
    }
    
    func count(_ text: String) -> Int {
        tokenizer.tokenCount(text)
    }
}
