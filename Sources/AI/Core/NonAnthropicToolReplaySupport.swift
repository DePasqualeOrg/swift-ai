// Copyright © Anthony DePasquale

import Foundation

enum NonAnthropicToolReplaySupport {
  static func normalizedMessages(_ messages: [Message]) -> [Message] {
    let knownToolCallIDs = Set(messages.flatMap(\.toolCalls).map(\.id))
    let toolResultIDs = Set(messages.flatMap { message in
      message.content.compactMap { item -> String? in
        guard case let .toolResult(toolResult) = item else { return nil }
        return toolResult.id
      }
    })
    let orphanedToolCallIDs = knownToolCallIDs.subtracting(toolResultIDs)

    var normalizedMessages: [Message] = []
    var emittedToolResultIDs = Set<String>()

    for message in messages {
      switch message.role {
        case .assistant:
          normalizedMessages.append(message)
          let orphanedToolCalls = message.toolCalls.filter { orphanedToolCallIDs.contains($0.id) }
          if let syntheticResultMessage = ToolReplaySupport.syntheticToolResultMessage(for: orphanedToolCalls) {
            normalizedMessages.append(syntheticResultMessage)
          }
        case .tool:
          normalizedMessages.append(contentsOf: normalizedToolMessages(
            from: message,
            knownToolCallIDs: knownToolCallIDs,
            emittedToolResultIDs: &emittedToolResultIDs,
          ))
        case .system, .developer, .user:
          normalizedMessages.append(message)
      }
    }

    return normalizedMessages
  }

  private enum SegmentKind {
    case nativeToolResult
    case collapsedUser
  }

  private static func normalizedToolMessages(
    from message: Message,
    knownToolCallIDs: Set<String>,
    emittedToolResultIDs: inout Set<String>,
  ) -> [Message] {
    var normalizedMessages: [Message] = []
    var currentKind: SegmentKind?
    var currentContent: [Message.Content] = []

    func flushCurrentSegment() {
      guard !currentContent.isEmpty, let segmentKind = currentKind else { return }
      switch segmentKind {
        case .nativeToolResult:
          normalizedMessages.append(Message(role: .tool, content: currentContent))
        case .collapsedUser:
          let collapsedMessage = Message(role: .tool, content: currentContent).collapsingToolResults()
          if !collapsedMessage.content.isEmpty {
            normalizedMessages.append(collapsedMessage)
          }
      }
      currentContent.removeAll(keepingCapacity: true)
      currentKind = nil
    }

    for item in message.content {
      let nextKind: SegmentKind
      if case let .toolResult(toolResult) = item,
         knownToolCallIDs.contains(toolResult.id),
         !emittedToolResultIDs.contains(toolResult.id)
      {
        emittedToolResultIDs.insert(toolResult.id)
        nextKind = .nativeToolResult
      } else {
        nextKind = .collapsedUser
      }

      if currentKind != nextKind {
        flushCurrentSegment()
        currentKind = nextKind
      }
      currentContent.append(item)
    }

    flushCurrentSegment()
    return normalizedMessages
  }
}
