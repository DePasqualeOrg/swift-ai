// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct GeminiReplayNormalizerTests {
  @Test
  func `Gemini normalizer emits explicit roles and request ready contents`() async throws {
    let client = try GeminiClient(
      session: makeMockSession(),
      modelsEndpoint: #require(URL(string: "https://mock.test/normalizer")),
    )
    let messages = [Message(role: .system, content: "Follow policy")] + ReplayFixtures.mixedToolTurnHistory()

    let plan = try await GeminiReplayNormalizer.normalize(
      messages,
      systemPrompt: "Top level system prompt",
      apiKey: "test-key",
      requestParts: { message, apiKey in
        try await client.requestParts(for: message, apiKey: apiKey)
      },
    )

    let systemTexts = plan.systemParts.compactMap { $0["text"] as? String }
    #expect(systemTexts == ["Top level system prompt", "Follow policy"])
    #expect(plan.contents.count == 5)

    #expect(plan.contents[0]["role"] as? String == "model")
    let matchedFunctionResponse = try #require((plan.contents[1]["parts"] as? [[String: Any]])?.first?["functionResponse"] as? [String: Any])
    #expect(plan.contents[1]["role"] as? String == "user")
    #expect(matchedFunctionResponse["id"] as? String == "call_1")

    let syntheticFunctionResponse = try #require((plan.contents[2]["parts"] as? [[String: Any]])?.first?["functionResponse"] as? [String: Any])
    let syntheticPayload = try #require(syntheticFunctionResponse["response"] as? [String: Any])
    #expect(plan.contents[2]["role"] as? String == "user")
    #expect(syntheticFunctionResponse["id"] as? String == "call_2")
    #expect((syntheticPayload["error"] as? String)?.contains(ToolReplaySupport.syntheticToolResultErrorText) == true)

    let collapsedStrayText = try #require((plan.contents[3]["parts"] as? [[String: Any]])?.first?["text"] as? String)
    #expect(plan.contents[3]["role"] as? String == "user")
    #expect(collapsedStrayText.contains(ReplayFixtures.strayToolResultText))
  }
}
