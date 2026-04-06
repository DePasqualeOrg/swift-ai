// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ReplayCapturePolicyTests {
  @Test
  func `Responses capture policy requests encrypted reasoning content for OpenAI-compatible endpoints`() {
    let backend = try? resolveResponsesBackend(for: ResponsesClient.Endpoint.openAI.url, provider: nil)
    let requirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "o3",
      backend: backend,
    ))

    #expect(requirements.requiresOpenAIResponsesReasoningEncryptedContent == true)
  }

  @Test
  func `Responses capture policy skips encrypted reasoning content for xAI endpoints`() {
    let backend = try? resolveResponsesBackend(for: ResponsesClient.Endpoint.xAI.url, provider: nil)
    let requirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "grok-4",
      backend: backend,
    ))

    #expect(requirements.requiresOpenAIResponsesReasoningEncryptedContent == false)
  }

  @Test
  func `Responses capture policy respects explicit provider for custom endpoints`() throws {
    let customEndpoint = try #require(URL(string: "https://proxy.example.test/v1/responses"))
    let openAIBackend = try resolveResponsesBackend(for: customEndpoint, provider: .openAI)
    let xAIBackend = try resolveResponsesBackend(for: customEndpoint, provider: .xAI)

    let openAIRequirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "o3",
      backend: openAIBackend,
    ))
    #expect(openAIRequirements.requiresOpenAIResponsesReasoningEncryptedContent == true)

    let xAIRequirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "grok-4",
      backend: xAIBackend,
    ))
    #expect(xAIRequirements.requiresOpenAIResponsesReasoningEncryptedContent == false)
  }

  @Test
  func `Responses capture policy skips encrypted reasoning content for ambiguous custom endpoints`() throws {
    let customEndpoint = try #require(URL(string: "https://proxy.example.test/v1/responses"))
    let backend = try resolveResponsesBackend(for: customEndpoint, provider: nil)

    let requirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: "o3",
      backend: backend,
    ))

    #expect(requirements.requiresOpenAIResponsesReasoningEncryptedContent == false)
  }
}
