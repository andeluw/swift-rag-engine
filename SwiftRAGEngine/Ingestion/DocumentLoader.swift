//
//  DocumentLoader.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import Foundation

protocol DocumentLoader {
    func load(from url: URL) throws -> LoadedDocument
}

enum DocumentLoaderError: Error {
    case fileNotFound
    case unsupportedFormat
    case failedToRead
}
