//
//  ContentView.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 24/08/26.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        Text("Swift RAG Engine")
            .task {
                await ExampleRunner.run()
            }
    }
}

#Preview {
    ContentView()
}
