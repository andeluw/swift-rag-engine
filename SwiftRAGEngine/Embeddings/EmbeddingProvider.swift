//
//  EmbeddingProvider.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

protocol EmbeddingProvider { 
    func embedQuery(_ text: String) async throws -> [Float]
    func embedDocument(_ text: String) async throws -> [Float]
}
