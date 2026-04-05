// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ReplayCapturePolicyTests {
  @Test
  func `Responses capture policy requests encrypted reasoning content for OpenAI-compatible endpoints`() {
    let requirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "o3",
      configuration: .init(),
      endpoint: ResponsesClient.Endpoint.openAI.url,
    ))

    #expect(requirements.requiresOpenAIResponsesReasoningEncryptedContent == true)
  }

  @Test
  func `Responses capture policy skips encrypted reasoning content for xAI endpoints`() {
    let requirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "grok-4",
      configuration: .init(),
      endpoint: ResponsesClient.Endpoint.xAI.url,
    ))

    #expect(requirements.requiresOpenAIResponsesReasoningEncryptedContent == false)
  }
}
