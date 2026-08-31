//
//  ExampleRunner.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

enum ExampleRunner {
    static func run() async {
//        do {
//            try await BGEEmbeddingExample.run()
//        } catch {
//            print("BGE demo error: \(error)")
//        }
//
//        do {
//            try await EmbeddingGemmaMLXExample.run()
//        } catch {
//            print("EmbeddingGemma MLX demo error: \(error)")
//        }
//
//        do {
//            try await EmbeddingGemmaCoreMLExample.run()
//        } catch {
//            print("EmbeddingGemma CoreML demo error: \(error)")
//        }
        
        do {
            try await RAGExample.run()
        } catch {
            print("RAG Example error: \(error)")
        }
    }
}
