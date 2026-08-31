//
//  TextLoader.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import Foundation

struct TextLoader: DocumentLoader {
    func load(from url: URL) throws -> LoadedDocument {
        guard url.pathExtension.lowercased() == "txt" else {
            throw DocumentLoaderError.unsupportedFormat
        }
        
        do {
            let text = try String(
                contentsOf: url,
                encoding: .utf8
            )
            
            return LoadedDocument(
                text: text,
                source: url.lastPathComponent
            )
        } catch {
            throw DocumentLoaderError.failedToRead
        }
    }
}
