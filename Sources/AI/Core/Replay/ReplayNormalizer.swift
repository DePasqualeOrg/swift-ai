// Copyright © Anthony DePasquale

struct ReplayNormalizationPlan {
  let messages: [Message]
}

enum ReplayNormalizer {
  static func normalize(
    _ messages: [Message],
    profile: ReplayProviderProfile,
  ) -> ReplayNormalizationPlan {
    ReplayNormalizationPlan(
      messages: ToolHistoryNormalizer.normalize(messages, policy: profile.toolHistory),
    )
  }
}
