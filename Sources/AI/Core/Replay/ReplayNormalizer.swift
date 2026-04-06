// Copyright © Anthony DePasquale

struct ReplayNormalizationPlan {
  let messages: [Message]
}

enum ReplayNormalizer {
  static func normalize(
    _ messages: [Message],
    profile: ReplayProviderProfile,
  ) -> ReplayNormalizationPlan {
    let toolRepairedMessages = ToolHistoryNormalizer.normalize(messages, policy: profile.toolHistory)
    return ReplayNormalizationPlan(
      messages: ReasoningHistoryNormalizer.normalize(
        toolRepairedMessages,
        target: profile.target,
        policy: profile.reasoning,
      ),
    )
  }
}
