//
//  RAGResult.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct RAGResult {
    let answer: String
    let hasSufficientContext: Bool
    let retrievedResults: [SearchResult]
}
