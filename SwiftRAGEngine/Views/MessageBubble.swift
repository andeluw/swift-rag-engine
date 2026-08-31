//
//  MessageBubble.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 31/08/26.
//

import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    @State private var isShowingSources = false

    var body: some View {
        HStack(alignment: .top) {
            if isUser {
                Spacer(minLength: 48)
            }

            VStack(
                alignment: .leading,
                spacing: 12
            ) {
                Text(message.text)
                    .textSelection(.enabled)

                if !message.sources.isEmpty {
                    sourcesView
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background {
                bubbleBackground
            }
            .frame(
                maxWidth: 520,
                alignment: isUser ? .trailing : .leading
            )

            if !isUser {
                Spacer(minLength: 48)
            }
        }
    }

    private var isUser: Bool {
        switch message.role {
        case .user:
            true

        case .assistant:
            false
        }
    }

    @ViewBuilder
    private var bubbleBackground: some View {
        if isUser {
            RoundedRectangle(
                cornerRadius: 18,
                style: .continuous
            )
            .fill(Color.accentColor.opacity(0.16))
        } else {
            RoundedRectangle(
                cornerRadius: 18,
                style: .continuous
            )
            .fill(.quaternary)
        }
    }

    private var sourcesView: some View {
        DisclosureGroup(
            isExpanded: $isShowingSources
        ) {
            VStack(
                alignment: .leading,
                spacing: 12
            ) {
                ForEach(
                    Array(message.sources.enumerated()),
                    id: \.element.id
                ) { index, source in
                    sourceView(source)

                    if index < message.sources.count - 1 {
                        Divider()
                    }
                }
            }
            .padding(.top, 8)
        } label: {
            Label(
                "Sources",
                systemImage: "doc.text"
            )
        }
        .font(.caption)
    }

    private func sourceView(
        _ source: ChatSource
    ) -> some View {
        VStack(
            alignment: .leading,
            spacing: 6
        ) {
            LabeledContent {
                Text(
                    source.score,
                    format: .number.precision(
                        .fractionLength(3)
                    )
                )
                .foregroundStyle(.secondary)
            } label: {
                Label(
                    source.name,
                    systemImage: "doc"
                )
                .fontWeight(.medium)
            }

            Text(source.text)
                .foregroundStyle(.secondary)
                .lineLimit(3)
        }
    }
}

#Preview("Assistant") {
    MessageBubble(
        message: ChatMessage(
            role: .assistant,
            text: """
                Wi-Fi can become unstable because of interference, \
                weak signal strength, or power-saving features.
                """,
            sources: [
                ChatSource(
                    id: UUID(),
                    name: "wifi.txt",
                    score: 0.612,
                    text: """
                        Wi-Fi connections can become unstable for \
                        several reasons. A device may be too far \
                        from the router.
                        """
                ),
                ChatSource(
                    id: UUID(),
                    name: "wifi.txt",
                    score: 0.583,
                    text: """
                        A weak signal does not always mean the \
                        internet connection itself is slow.
                        """
                ),
            ]
        )
    )
    .padding()
}

#Preview("User") {
    MessageBubble(
        message: ChatMessage(
            role: .user,
            text: "Why does my Wi-Fi keep disconnecting?",
            sources: []
        )
    )
    .padding()
}

#Preview("Dark") {
    MessageBubble(
        message: ChatMessage(
            role: .assistant,
            text: "Wi-Fi interference can cause an unstable connection.",
            sources: []
        )
    )
    .padding()
    .preferredColorScheme(.dark)
}
