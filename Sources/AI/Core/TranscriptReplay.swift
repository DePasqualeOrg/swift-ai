// Copyright © Anthony DePasquale

import Foundation

enum TranscriptReplayTarget {
  case anthropic(thinkingEnabled: Bool)
  case gemini
  case responses
}

enum TranscriptReplay {
  static func prepare(
    _ messages: [Message],
    for target: TranscriptReplayTarget,
  ) -> [Message] {
    switch target {
      case let .anthropic(thinkingEnabled) where thinkingEnabled:
        collapseAnthropicThinkingHistory(in: sanitizeAnthropicToolHistory(in: messages))
      case .anthropic:
        sanitizeAnthropicToolHistory(in: messages)
      case .gemini, .responses:
        Message.patchingOrphanedToolCalls(messages)
    }
  }

  private static func sanitizeAnthropicToolHistory(in messages: [Message]) -> [Message] {
    var preparedMessages: [Message] = []
    var pendingToolCalls: [ToolCall] = []

    func appendSyntheticToolResultsIfNeeded() {
      guard !pendingToolCalls.isEmpty else { return }
      preparedMessages.append(Message(role: .tool, content: pendingToolCalls.map { toolCall in
        .toolResult(ToolResult(
          name: toolCall.name,
          id: toolCall.id,
          content: [.text("Function call was not executed. The request may have been canceled or timed out.")],
          isError: true,
        ))
      }))
      pendingToolCalls.removeAll()
    }

    for message in messages {
      switch message.role {
        case .assistant:
          appendSyntheticToolResultsIfNeeded()
          preparedMessages.append(message)
          pendingToolCalls = message.toolCalls
        case .tool:
          var validToolContent: [Message.Content] = []
          var invalidToolResultContent: [Message.Content] = []
          var matchableToolCallIDs = Set(pendingToolCalls.map(\.id))

          for item in message.content {
            if case let .toolResult(toolResult) = item {
              if matchableToolCallIDs.contains(toolResult.id) {
                validToolContent.append(item)
                matchableToolCallIDs.remove(toolResult.id)
              } else {
                invalidToolResultContent.append(item)
              }
            } else {
              validToolContent.append(item)
            }
          }

          let matchedToolResultIDs = Set(validToolContent.compactMap { item -> String? in
            guard case let .toolResult(toolResult) = item else { return nil }
            return toolResult.id
          })

          if matchedToolResultIDs.isEmpty {
            appendSyntheticToolResultsIfNeeded()
            let collapsedMessage = Message(role: .tool, content: message.content).collapsingToolResults()
            if !collapsedMessage.content.isEmpty {
              preparedMessages.append(collapsedMessage)
            }
            continue
          }

          preparedMessages.append(Message(role: .tool, content: validToolContent))
          pendingToolCalls.removeAll { matchedToolResultIDs.contains($0.id) }

          if !invalidToolResultContent.isEmpty {
            appendSyntheticToolResultsIfNeeded()
            let collapsedMessage = Message(role: .tool, content: invalidToolResultContent).collapsingToolResults()
            if !collapsedMessage.content.isEmpty {
              preparedMessages.append(collapsedMessage)
            }
          }
        case .system, .developer, .user:
          appendSyntheticToolResultsIfNeeded()
          preparedMessages.append(message)
      }
    }

    appendSyntheticToolResultsIfNeeded()
    return preparedMessages
  }

  private static func collapseAnthropicThinkingHistory(in messages: [Message]) -> [Message] {
    var preparedMessages: [Message] = []
    var skipUntilIndex = 0

    for (index, message) in messages.enumerated() {
      if index < skipUntilIndex {
        continue
      }

      if message.role == .assistant, message.hasToolCalls, !message.hasNativeAnthropicThinkingBlocks {
        preparedMessages.append(message.collapsingToolCalls())

        var nextIndex = index + 1
        while nextIndex < messages.count, messages[nextIndex].hasToolResults {
          preparedMessages.append(messages[nextIndex].collapsingToolResults())
          nextIndex += 1
        }
        skipUntilIndex = nextIndex
      } else {
        preparedMessages.append(message)
      }
    }

    return preparedMessages
  }
}

extension Message {
  var hasToolCalls: Bool {
    content.contains {
      if case .toolCall = $0 { return true }
      return false
    }
  }

  var hasToolResults: Bool {
    content.contains {
      if case .toolResult = $0 { return true }
      return false
    }
  }

  var toolCalls: [ToolCall] {
    content.compactMap {
      guard case let .toolCall(toolCall) = $0 else { return nil }
      return toolCall
    }
  }
}
