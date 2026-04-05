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
  case responses(modelId: String, configuration: ResponsesClient.Configuration, endpoint: URL)
  case chatCompletions(modelId: String, configuration: ChatCompletionsClient.Configuration)
  case gemini(modelId: String, configuration: GeminiClient.Configuration)
}

enum ReplayCapturePolicy {
  static func requirements(for target: ReplayRequestTarget) -> ReplayCaptureRequirements {
    switch target {
      case let .responses(_, _, endpoint) where shouldCaptureOpenAIResponsesReasoning(for: endpoint):
        ReplayCaptureRequirements(requiredFields: [.openAIResponsesReasoningEncryptedContent])
      case .anthropic, .responses, .chatCompletions, .gemini:
        ReplayCaptureRequirements()
    }
  }

  private static func shouldCaptureOpenAIResponsesReasoning(for endpoint: URL) -> Bool {
    let host = endpoint.host?.lowercased()
    return host != ResponsesClient.Endpoint.xAI.url.host?.lowercased()
  }
}
