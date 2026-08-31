//
//  ContextBuilder.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct ContextBuilder {
    static func build(
        from results: [SearchResult]
    ) -> String {
        results.enumerated()
            .map { index, result in
            """
            [Source \(index + 1)]
            \(result.chunk.text)
            """
            }
            .joined(separator: "\n\n---\n\n")
    }
}
