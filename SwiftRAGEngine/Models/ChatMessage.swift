//
//  ChatMessage.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import Foundation

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let text: String
    let sources: [ChatSource]
    
    enum Role: Equatable {
        case user
        case assistant
    }
}

struct ChatSource: Identifiable {
    let id: UUID
    let name: String
    let score: Float
    let text: String
}
