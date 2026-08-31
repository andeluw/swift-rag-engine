//
//  BGETokenizer.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 26/08/26.
//

import Foundation
import Tokenizers

final class BGETokenizer {
    private let tokenizer: Tokenizer

    private let maxSequenceLength = 128
    private let padTokenID = 0

    init() async throws {
        guard
            let tokenizerJSON = Bundle.main.url(
                forResource: "tokenizer",
                withExtension: "json",
            )
        else {
            throw BGETokenizerError.tokenizerNotFound
        }

        let tokenizerFolder = tokenizerJSON.deletingLastPathComponent()

        tokenizer = try await AutoTokenizer.from(
            modelFolder: tokenizerFolder
        )
    }
    
    func tokenCount(_ text: String) -> Int {
        tokenizer.encode(text: text).count
    }

    func encode(_ text: String) -> BGETokenizedInput {
        var tokenIDs = tokenizer.encode(text: text)

        // Truncate to maximum sequence length
        if tokenIDs.count > maxSequenceLength {
            tokenIDs = Array(tokenIDs.prefix(maxSequenceLength))
        }

        // Attention mask -> 1 for real tokens
        var attentionMask = Array(
            repeating: Int32(1),
            count: tokenIDs.count
        )

        var inputIDs = tokenIDs.map {
            Int32($0)
        }

        // Padding -> 0 until sequence length is 128
        let paddingCount = maxSequenceLength - inputIDs.count

        inputIDs.append(
            contentsOf: Array(
                repeating: Int32(padTokenID),
                count: paddingCount
            )
        )

        attentionMask.append(
            contentsOf: Array(
                repeating: Int32(0),
                count: paddingCount
            )
        )

        return BGETokenizedInput(
            inputIDs: inputIDs,
            attentionMask: attentionMask
        )
    }
}

struct BGETokenizedInput {
    let inputIDs: [Int32]
    let attentionMask: [Int32]
}

enum BGETokenizerError: Error {
    case tokenizerNotFound
}
