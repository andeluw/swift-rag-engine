//
//  SImilarity.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 27/08/26.
//

import Foundation

enum CosineSimilarity {
    static func calculate(
        _ a: [Float],
        _ b: [Float]
    ) -> Float {
        let dotProduct = zip(a, b).reduce(Float(0)) {
            $0 + ($1.0 * $1.1)
        }

        let normA = sqrt(
            a.reduce(Float(0)) {
                $0 + ($1 * $1)
            }
        )

        let normB = sqrt(
            b.reduce(Float(0)) {
                $0 + ($1 * $1)
            }
        )

        return dotProduct / (normA * normB)
    }
}
