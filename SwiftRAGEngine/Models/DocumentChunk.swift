//
//  DocumentChunk.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 28/08/26.
//

import Foundation

struct DocumentChunk: Identifiable {
    let id: UUID
    let text: String
    let source: String
}
