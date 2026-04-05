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
    let messages = [
      Message(role: .system, content: "Follow policy"),
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
        .toolCall(ToolCall(name: "lookup", id: "call_2", parameters: ["id": "42"])),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text("Matched result"))),
        .toolResult(ToolResult(name: "stale", id: "call_stray", content: .text("Stray result"))),
      ]),
      Message(role: .user, content: "Continue"),
    ]

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
    let syntheticFunctionResponse = try #require((plan.contents[1]["parts"] as? [[String: Any]])?.first?["functionResponse"] as? [String: Any])
    let syntheticPayload = try #require(syntheticFunctionResponse["response"] as? [String: Any])
    #expect(plan.contents[1]["role"] as? String == "user")
    #expect((syntheticPayload["error"] as? String)?.contains(ToolReplaySupport.syntheticToolResultErrorText) == true)

    let matchedFunctionResponse = try #require((plan.contents[2]["parts"] as? [[String: Any]])?.first?["functionResponse"] as? [String: Any])
    #expect(plan.contents[2]["role"] as? String == "user")
    #expect(matchedFunctionResponse["id"] as? String == "call_1")

    let collapsedStrayText = try #require((plan.contents[3]["parts"] as? [[String: Any]])?.first?["text"] as? String)
    #expect(plan.contents[3]["role"] as? String == "user")
    #expect(collapsedStrayText.contains("Stray result"))
  }
}
