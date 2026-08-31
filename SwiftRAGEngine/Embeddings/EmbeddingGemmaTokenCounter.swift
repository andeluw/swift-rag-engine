//
//  EmbeddingGemmaTokenCounter.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import CoreMLLLM
import Foundation
import Tokenizers

struct EmbeddingGemmaTokenCounter: TokenCounter {
    let maxTokens: Int
    
    private let tokenizer: Tokenizer
    private let documentPrefix: String
    
    init() async throws {
        let modelsDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        
        let bundleURL = try await Gemma3BundleDownloader.download(.embeddingGemma300m, into: modelsDirectory)
        
        let tokenizerDirectory = bundleURL.appendingPathComponent("hf_model")
        
        tokenizer = try await AutoTokenizer.from(
            modelFolder: tokenizerDirectory
        )
        
        let configURL = bundleURL
            .appendingPathComponent("model_config.json")
        
        
        let data = try Data(contentsOf: configURL)
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        maxTokens = json?["max_seq_len"] as? Int ?? 512
        
        let taskPrefixes = json?["task_prefixes"] as? [String: String]
        
        documentPrefix = taskPrefixes?["retrieval_document"] ?? ""
    }
    
    func count(_ text: String) -> Int {
        tokenizer.encode(text: text).count
    }
    
    private static func loadMaxTokens(
        from bundleURL: URL
    ) throws -> Int {
        let configURL = bundleURL
            .appendingPathComponent("model_config.json")
        
        
        let data = try Data(contentsOf: configURL)
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        return json?["max_seq_len"] as? Int ?? 512
    }
}
