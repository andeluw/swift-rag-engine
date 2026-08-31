//
//  TokenCounter.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

protocol TokenCounter {
    var maxTokens: Int { get }
    
    func count(_ text: String) -> Int
}
