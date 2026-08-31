import SwiftUI

struct ChatView: View {
    @Bindable var viewModel: RAGViewModel

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if viewModel.isReady, let error = viewModel.errorMessage {
                    errorBanner(error)
                        .padding(.vertical, 8)
                }

                messagesView
                    .frame(maxHeight: .infinity)
            }
            .navigationTitle("Swift RAG")
            .safeAreaInset(edge: .bottom, spacing: 0) {
                inputView
            }
            .task {
                await viewModel.prepare()
            }
        }
        .frame(minWidth: 480, minHeight: 600)
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")

            Text(message)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                viewModel.dismissError()
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
        }
        .font(.callout)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.red.opacity(0.12), in: .rect(cornerRadius: 10))
        .foregroundStyle(.red)
        .padding(.horizontal)
    }

    @ViewBuilder
    private var messagesView: some View {
        if viewModel.isIndexing {
            VStack(spacing: 12) {
                ProgressView()

                Text("Indexing documents...")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if !viewModel.isReady, let error = viewModel.errorMessage {
            ContentUnavailableView {
                Label(
                    "Couldn't Index Documents",
                    systemImage: "exclamationmark.triangle"
                )
            } description: {
                Text(error)
            } actions: {
                Button("Try Again") {
                    Task {
                        await viewModel.prepare()
                    }
                }
            }
        } else if viewModel.messages.isEmpty {
            ContentUnavailableView(
                "Ask Your Documents",
                systemImage: "doc.text.magnifyingglass",
                description: Text(
                    "Ask a question about the indexed documents."
                )
            )
        } else {
            scrollingMessages
        }
    }

    private var scrollingMessages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }

                    if viewModel.isGenerating {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)

                            Text("Generating...")
                                .foregroundStyle(.secondary)

                            Spacer()
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            .onChange(of: viewModel.messages.count) {
                guard let lastMessage = viewModel.messages.last else {
                    return
                }

                withAnimation {
                    proxy.scrollTo(
                        lastMessage.id,
                        anchor: .bottom
                    )
                }
            }
        }
    }

    private var inputView: some View {
        HStack(spacing: 8) {
            TextField(
                "Ask about your documents...",
                text: $viewModel.input
            )
            .textFieldStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.quaternary, in: .capsule)
            .overlay {
                Capsule()
                    .strokeBorder(.separator)
            }
            .onSubmit {
                Task {
                    await viewModel.send()
                }
            }

            Button {
                Task {
                    await viewModel.send()
                }
            } label: {
                Image(systemName: "arrow.up")
                    .imageScale(.large)
            }
            .buttonStyle(.borderedProminent)
            .buttonBorderShape(.circle)
            .controlSize(.extraLarge)
            .disabled(!canSend)
        }
        .padding(.horizontal)
        .padding(.top, 8)
        .padding(.bottom, 20)
    }

    private var canSend: Bool {
        viewModel.isReady
            && !viewModel.isGenerating
            && !viewModel.input
                .trimmingCharacters(
                    in: .whitespacesAndNewlines
                )
                .isEmpty
    }
}
