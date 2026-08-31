//
//  VectorSearch.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

struct VectorSearch {
    static func search(
        queryEmbedding: [Float],
        chunks: [EmbeddedChunk],
        topK: Int
    ) -> [SearchResult] {
        let results = chunks.map { item in
            let score = CosineSimilarity.calculate(
                queryEmbedding,
                item.embedding
            )

            return SearchResult(
                chunk: item.chunk,
                score: score
            )
        }

        return Array(
            results
                .sorted { $0.score > $1.score }
                .prefix(topK)
        )
    }
}
