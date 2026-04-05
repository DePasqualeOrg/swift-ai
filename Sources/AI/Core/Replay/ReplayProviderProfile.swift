// Copyright © Anthony DePasquale

struct ReplayProviderProfile {
  struct ToolHistoryPolicy {
    var synthesizesMissingToolResults: Bool
    var requiresStrictAdjacency: Bool
    var collapsesStrayToolContent: Bool
    var splitsMixedToolMessages: Bool

    static let strict = Self(
      synthesizesMissingToolResults: true,
      requiresStrictAdjacency: true,
      collapsesStrayToolContent: true,
      splitsMixedToolMessages: true,
    )
  }

  struct ReasoningReplayPolicy {
    var collapsesToolExchangesWithoutNativeReasoning: Bool

    static let passthrough = Self(collapsesToolExchangesWithoutNativeReasoning: false)
  }

  let target: ReplayTarget
  let toolHistory: ToolHistoryPolicy
  let reasoning: ReasoningReplayPolicy
}

extension ReplayProviderProfile {
  static func forAnthropic(thinkingEnabled: Bool) -> Self {
    Self(
      target: .anthropic,
      toolHistory: .strict,
      reasoning: .init(collapsesToolExchangesWithoutNativeReasoning: thinkingEnabled),
    )
  }

  static var responses: Self {
    Self(
      target: .responses,
      toolHistory: .strict,
      reasoning: .passthrough,
    )
  }

  static var chatCompletions: Self {
    Self(
      target: .chatCompletions,
      toolHistory: .strict,
      reasoning: .passthrough,
    )
  }

  static var gemini: Self {
    Self(
      target: .gemini,
      toolHistory: .strict,
      reasoning: .passthrough,
    )
  }
}
