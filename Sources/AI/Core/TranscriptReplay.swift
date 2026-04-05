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
    let patchedMessages = Message.patchingOrphanedToolCalls(messages)

    return switch target {
      case let .anthropic(thinkingEnabled) where thinkingEnabled:
        collapseAnthropicThinkingHistory(in: patchedMessages)
      case .anthropic, .gemini, .responses:
        patchedMessages
    }
  }

  private static func collapseAnthropicThinkingHistory(in messages: [Message]) -> [Message] {
    var preparedMessages: [Message] = []
    var skipNext = false

    for (index, message) in messages.enumerated() {
      if skipNext {
        skipNext = false
        continue
      }

      if message.role == .assistant, message.hasToolCalls, !message.hasNativeAnthropicThinkingBlocks {
        preparedMessages.append(message.collapsingToolCalls())

        if index + 1 < messages.count, messages[index + 1].hasToolResults {
          preparedMessages.append(messages[index + 1].collapsingToolResults())
          skipNext = true
        }
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
}
