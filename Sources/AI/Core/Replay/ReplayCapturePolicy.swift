// Copyright © Anthony DePasquale

import Foundation

enum ReplayCaptureRequirement: Hashable {
  case openAIResponsesReasoningEncryptedContent
}

struct ReplayCaptureRequirements {
  let requiredFields: Set<ReplayCaptureRequirement>

  init(requiredFields: Set<ReplayCaptureRequirement> = []) {
    self.requiredFields = requiredFields
  }

  func contains(_ requirement: ReplayCaptureRequirement) -> Bool {
    requiredFields.contains(requirement)
  }

  var requiresOpenAIResponsesReasoningEncryptedContent: Bool {
    contains(.openAIResponsesReasoningEncryptedContent)
  }
}

enum ReplayRequestTarget {
  case anthropic(modelId: String, configuration: AnthropicClient.Configuration)
  case responses(modelId: String, backend: ResolvedResponsesBackend?)
  case chatCompletions(modelId: String, configuration: ChatCompletionsClient.Configuration)
  case gemini(modelId: String, configuration: GeminiClient.Configuration)
}

enum ReplayCapturePolicy {
  static func requirements(for target: ReplayRequestTarget) -> ReplayCaptureRequirements {
    switch target {
      case let .responses(_, backend?) where backend.requiresEncryptedReasoningCapture:
        ReplayCaptureRequirements(requiredFields: [.openAIResponsesReasoningEncryptedContent])
      case .anthropic, .responses, .chatCompletions, .gemini:
        ReplayCaptureRequirements()
    }
  }
}
