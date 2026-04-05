// Copyright © Anthony DePasquale

enum ToolHistoryNormalizer {
  static func normalize(
    _ messages: [Message],
    policy: ReplayProviderProfile.ToolHistoryPolicy = .strict,
  ) -> [Message] {
    var normalizedMessages: [Message] = []
    var pendingToolCalls: [ToolCall] = []

    func flushPendingToolCalls() {
      defer { pendingToolCalls.removeAll() }
      guard policy.synthesizesMissingToolResults,
            let syntheticResultMessage = ToolReplaySupport.syntheticToolResultMessage(for: pendingToolCalls)
      else {
        return
      }
      normalizedMessages.append(syntheticResultMessage)
    }

    for message in messages {
      switch message.role {
        case .assistant:
          if policy.requiresStrictAdjacency {
            flushPendingToolCalls()
          }
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
            normalizedMessages.append(Message(
              role: .tool,
              content: matchedToolResults.map(Message.Content.toolResult),
            ))
            let matchedToolResultIDs = Set(matchedToolResults.map(\.id))
            pendingToolCalls.removeAll { matchedToolResultIDs.contains($0.id) }
          }

          guard !invalidContent.isEmpty else { continue }

          if policy.requiresStrictAdjacency {
            flushPendingToolCalls()
          }

          if policy.collapsesStrayToolContent {
            let collapsedMessage = Message(role: .tool, content: invalidContent).collapsingToolResults()
            if !collapsedMessage.content.isEmpty {
              normalizedMessages.append(collapsedMessage)
            }
          } else if policy.splitsMixedToolMessages || matchedToolResults.isEmpty {
            normalizedMessages.append(Message(role: .tool, content: invalidContent))
          }

        case .system, .developer, .user:
          if policy.requiresStrictAdjacency {
            flushPendingToolCalls()
          }
          normalizedMessages.append(message)
      }
    }

    if policy.requiresStrictAdjacency {
      flushPendingToolCalls()
    }

    return normalizedMessages
  }
}
