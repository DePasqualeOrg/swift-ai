// Copyright © Anthony DePasquale

enum ReasoningHistoryNormalizer {
  static func normalize(
    _ messages: [Message],
    target: ReplayTarget,
    policy: ReplayProviderProfile.ReasoningReplayPolicy,
  ) -> [Message] {
    guard policy.collapsesToolExchangesWithoutNativeReasoning else {
      return messages
    }

    var normalizedMessages: [Message] = []
    var skipUntilIndex = 0

    for (index, message) in messages.enumerated() {
      if index < skipUntilIndex {
        continue
      }

      if message.role == .assistant,
         message.hasToolCalls,
         !message.hasNativeReasoningContent(for: target)
      {
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
