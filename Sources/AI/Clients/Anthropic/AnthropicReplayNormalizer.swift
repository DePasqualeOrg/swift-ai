// Copyright © Anthony DePasquale

import Foundation

enum AnthropicReplayNormalizer {
  struct Plan {
    let systemTexts: [String]
    let messages: [AnthropicClient.MessageParam]
  }

  static func normalize(_ messages: [Message], thinkingEnabled: Bool) async throws -> Plan {
    let normalizedMessages = normalizedMessages(messages, thinkingEnabled: thinkingEnabled)
    var systemTexts: [String] = []
    var messageParams: [AnthropicClient.MessageParam] = []

    for message in normalizedMessages {
      if message.role == .system || message.role == .developer {
        systemTexts.append(contentsOf: AnthropicClient.systemInstructionTexts(for: message))
        continue
      }

      let contentBlocks = try await AnthropicClient.anthropicContentBlocks(for: message)
      guard !contentBlocks.isEmpty else { continue }
      messageParams.append(AnthropicClient.MessageParam(
        role: AnthropicClient.mapRole(message.role),
        text: nil,
        contentBlocks: contentBlocks,
        attachments: nil,
      ))
    }

    return Plan(systemTexts: systemTexts, messages: messageParams)
  }

  private static func normalizedMessages(_ messages: [Message], thinkingEnabled: Bool) -> [Message] {
    let sanitizedMessages = sanitizeToolHistory(in: messages)
    guard thinkingEnabled else { return sanitizedMessages }
    return collapseThinkingHistory(in: sanitizedMessages)
  }

  private static func sanitizeToolHistory(in messages: [Message]) -> [Message] {
    var normalizedMessages: [Message] = []
    var pendingToolCalls: [ToolCall] = []

    func flushPendingToolCalls() {
      guard let syntheticResultMessage = ToolReplaySupport.syntheticToolResultMessage(for: pendingToolCalls) else { return }
      normalizedMessages.append(syntheticResultMessage)
      pendingToolCalls.removeAll()
    }

    for message in messages {
      switch message.role {
        case .assistant:
          flushPendingToolCalls()
          normalizedMessages.append(message)
          pendingToolCalls = message.toolCalls

        case .tool:
          var remainingPendingIDs = Set(pendingToolCalls.map(\.id))
          var matchedToolResults: [ToolResult] = []
          var invalidContent: [Message.Content] = []

          for item in message.content {
            if case let .toolResult(toolResult) = item,
               remainingPendingIDs.contains(toolResult.id)
            {
              matchedToolResults.append(toolResult)
              remainingPendingIDs.remove(toolResult.id)
            } else {
              invalidContent.append(item)
            }
          }

          if !matchedToolResults.isEmpty {
            let matchedToolResultIDs = Set(matchedToolResults.map(\.id))
            normalizedMessages.append(Message(
              role: .tool,
              content: matchedToolResults.map(Message.Content.toolResult),
            ))
            pendingToolCalls.removeAll { matchedToolResultIDs.contains($0.id) }
          }

          if !invalidContent.isEmpty {
            flushPendingToolCalls()
            let collapsedMessage = Message(role: .tool, content: invalidContent).collapsingToolResults()
            if !collapsedMessage.content.isEmpty {
              normalizedMessages.append(collapsedMessage)
            }
          }

        case .system, .developer, .user:
          flushPendingToolCalls()
          normalizedMessages.append(message)
      }
    }

    flushPendingToolCalls()
    return normalizedMessages
  }

  private static func collapseThinkingHistory(in messages: [Message]) -> [Message] {
    var normalizedMessages: [Message] = []
    var skipUntilIndex = 0

    for (index, message) in messages.enumerated() {
      if index < skipUntilIndex {
        continue
      }

      if message.role == .assistant, message.hasToolCalls, !message.hasNativeAnthropicThinkingBlocks {
        normalizedMessages.append(message.collapsingToolCalls())

        var nextIndex = index + 1
        while nextIndex < messages.count, messages[nextIndex].hasToolResults {
          normalizedMessages.append(messages[nextIndex].collapsingToolResults())
          nextIndex += 1
        }
        skipUntilIndex = nextIndex
      } else {
        normalizedMessages.append(message)
      }
    }

    return normalizedMessages
  }
}
