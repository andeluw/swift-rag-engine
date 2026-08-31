//
//  ContentView.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 24/08/26.
//

import SwiftUI

struct ContentView: View {
    @State private var viewModel = RAGViewModel()

    var body: some View {
        ChatView(viewModel: viewModel)
    }
}

#Preview {
    ContentView()
}
