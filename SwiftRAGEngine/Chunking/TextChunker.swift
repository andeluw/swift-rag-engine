//
//  TextChunker.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 28/08/26.
//

import Foundation
import NaturalLanguage

struct TextChunker {
    let targetTokens: Int
    let maxTokens: Int
    let tokenCount: (String) -> Int

    init(
        targetTokens: Int = 100,
        maxTokens: Int = 128,
        tokenCount: @escaping (String) -> Int
    ) {
        self.targetTokens = targetTokens
        self.maxTokens = maxTokens
        self.tokenCount = tokenCount
    }

    func chunk(
        text: String,
        source: String
    ) -> [DocumentChunk] {
        let paragraphs = splitParagraphs(text)

        var chunks: [DocumentChunk] = []
        var currentText = ""

        for paragraph in paragraphs {
            // Split oversized paragraphs further
            if tokenCount(paragraph) > maxTokens {
                flush(&currentText, into: &chunks, source: source)
                
                let parts = splitLargeParagraph(paragraph)

                for part in parts {
                    chunks.append(
                        DocumentChunk(
                            id: UUID(),
                            text: part,
                            source: source
                        )
                    )
                }

                continue
            }

            let combined =
                currentText.isEmpty
                ? paragraph : "\(currentText)\n\n\(paragraph)"

            // Keep adjacent paragraphs together when possible
            if tokenCount(combined) <= targetTokens {
                currentText = combined
            } else {
                flush(&currentText, into: &chunks, source: source)

                currentText = paragraph
            }
        }
        
        flush(
            &currentText,
            into: &chunks,
            source: source
        )

        return chunks
    }

    private func splitParagraphs(_ text: String) -> [String] {
        text.components(separatedBy: "\n\n")
            .map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            .filter {
                !$0.isEmpty
            }
    }

    private func splitLargeParagraph(_ paragraph: String) -> [String] {
        let sentences = splitSentences(paragraph)

        var parts: [String] = []
        var currentText = ""

        for sentence in sentences {
            // Handle a single sentence that is still too large
            if tokenCount(sentence) > maxTokens {
                if !currentText.isEmpty {
                    parts.append(currentText)
                    currentText = ""
                }

                parts.append(contentsOf: splitLargeSentence(sentence))
                continue
            }

            let combined =
                currentText.isEmpty ? sentence : "\(currentText) \(sentence)"

            if tokenCount(combined) <= maxTokens {
                currentText = combined
            } else {
                if !currentText.isEmpty {
                    parts.append(currentText)
                }

                currentText = sentence
            }
        }
        
        if !currentText.isEmpty {
            parts.append(currentText)
        }

        return parts
    }

    private func splitSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []

        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) {
            range,
            _ in
            let sentence = String(text[range])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if !sentence.isEmpty {
                sentences.append(sentence)
            }

            return true
        }

        return sentences
    }

    private func splitLargeSentence(_ sentence: String) -> [String] {
        let words = sentence.split(whereSeparator: \.isWhitespace)

        var parts: [String] = []
        var currentText = ""

        for word in words {
            let word = String(word)

            let combined = currentText.isEmpty ? word : "\(currentText) \(word)"

            if tokenCount(combined) <= maxTokens {
                currentText = combined
            } else {
                if !currentText.isEmpty {
                    parts.append(currentText)
                }

                currentText = word
            }
        }

        if !currentText.isEmpty {
            parts.append(currentText)
        }

        return parts
    }

    private func flush(
        _ text: inout String,
        into chunks: inout [DocumentChunk],
        source: String
    ) {
        guard !text.isEmpty else {
            return
        }

        chunks.append(
            DocumentChunk(
                id: UUID(),
                text: text,
                source: source
            )
        )

        text = ""
    }
}
