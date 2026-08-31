//
//  RetrievalPolicy.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct RetrievalPolicy {
    let minimumScore: Float
    
    func hasSufficientContext(
        _ results: [SearchResult]
    ) -> Bool {
        guard let bestResult = results.first else {
            return false
        }
        
        return bestResult.score >= minimumScore
    }
}
