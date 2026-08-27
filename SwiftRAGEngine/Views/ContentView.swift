//
//  ContentView.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 24/08/26.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        Text("Swift RAG Engine")
            .task {
                do {
                    try await BGEEmbeddingDemo.run()
                } catch {
                    print("BGE demo error: \(error)")
                }

                do {
                    try await EmbeddingGemmaMLXDemo.run()
                } catch {
                    print("EmbeddingGemma MLX demo error: \(error)")
                }

                do {
                    try await EmbeddingGemmaCoreMLDemo.run()
                } catch {
                    print("EmbeddingGemma CoreML demo error: \(error)")
                }
            }
    }
}

#Preview {
    ContentView()
}
