// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ReplayInvariantFixtureTests {
  @Test
  func `Cross-provider fixture preserves mixed tool history invariants`() async throws {
    let messages = ReplayFixtures.mixedToolTurnHistory()

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(messages)
    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: false)
    let geminiPlan = try await geminiPlan(for: messages)

    let responsesOutputs = responsesPlan.inputItems.filter { $0["type"] as? String == "function_call_output" }
    #expect(responsesOutputs.count == 2)
    let matchedOutput = try #require(responsesOutputs.first { $0["call_id"] as? String == "call_1" })
    #expect(matchedOutput["output"] as? String == ReplayFixtures.matchedToolResultText)
    let syntheticOutput = try #require(responsesOutputs.first { $0["call_id"] as? String == "call_2" })
    let syntheticError = try #require(syntheticOutput["output"] as? String)
    #expect(syntheticError.contains(ToolReplaySupport.syntheticToolResultErrorText))
    #expect(responsesTexts(in: responsesPlan).contains { $0.contains(ReplayFixtures.strayToolResultText) })

    #expect(anthropicPlan.messages.count == 5)
    let anthropicMatched = try #require(anthropicPlan.messages[1].contentBlocks?.first?.toolResult)
    #expect(anthropicMatched.toolUseId == "call_1")
    let anthropicSynthetic = try #require(anthropicPlan.messages[2].contentBlocks?.first?.toolResult)
    #expect(anthropicSynthetic.toolUseId == "call_2")
    #expect(anthropicSynthetic.isError == true)
    #expect(anthropicTexts(in: anthropicPlan).contains { $0.contains(ReplayFixtures.strayToolResultText) })

    let geminiFunctionResponses = geminiPlan.contents.compactMap { content -> [String: Any]? in
      let parts = content["parts"] as? [[String: Any]]
      return parts?.first?["functionResponse"] as? [String: Any]
    }
    #expect(geminiFunctionResponses.count == 2)
    let geminiMatched = try #require(geminiFunctionResponses.first { $0["id"] as? String == "call_1" })
    let geminiMatchedPayload = try #require(geminiMatched["response"] as? [String: Any])
    #expect(geminiMatchedPayload["output"] as? String == ReplayFixtures.matchedToolResultText)
    let geminiSynthetic = try #require(geminiFunctionResponses.first { $0["id"] as? String == "call_2" })
    let geminiSyntheticPayload = try #require(geminiSynthetic["response"] as? [String: Any])
    #expect((geminiSyntheticPayload["error"] as? String)?.contains(ToolReplaySupport.syntheticToolResultErrorText) == true)
    #expect(geminiTexts(in: geminiPlan).contains { $0.contains(ReplayFixtures.strayToolResultText) })
  }

  @Test
  func `Cross-provider fixture repairs late tool results after unrelated turns`() async throws {
    let messages = ReplayFixtures.lateToolResultHistory()

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(messages)
    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: false)
    let geminiPlan = try await geminiPlan(for: messages)

    let responsesOutput = try #require(responsesPlan.inputItems.first { $0["type"] as? String == "function_call_output" })
    #expect(responsesOutput["call_id"] as? String == "call_1")
    let responsesSyntheticError = try #require(responsesOutput["output"] as? String)
    #expect(responsesSyntheticError.contains(ToolReplaySupport.syntheticToolResultErrorText))
    #expect(responsesTexts(in: responsesPlan).contains { $0.contains(ReplayFixtures.lateToolResultText) })

    #expect(anthropicPlan.messages.count == 4)
    let anthropicSynthetic = try #require(anthropicPlan.messages[1].contentBlocks?.first?.toolResult)
    #expect(anthropicSynthetic.toolUseId == "call_1")
    #expect(anthropicSynthetic.isError == true)
    #expect(anthropicTexts(in: anthropicPlan).contains { $0.contains(ReplayFixtures.lateToolResultText) })

    let geminiSynthetic = try #require(geminiFunctionResponses(in: geminiPlan).first)
    #expect(geminiSynthetic["id"] as? String == "call_1")
    let geminiSyntheticPayload = try #require(geminiSynthetic["response"] as? [String: Any])
    #expect((geminiSyntheticPayload["error"] as? String)?.contains(ToolReplaySupport.syntheticToolResultErrorText) == true)
    #expect(geminiTexts(in: geminiPlan).contains { $0.contains(ReplayFixtures.lateToolResultText) })
  }

  @Test
  func `Cross-provider fixture synthesizes trailing unresolved tool calls consistently`() async throws {
    let messages = ReplayFixtures.trailingUnresolvedToolCallHistory()

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(messages)
    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: false)
    let geminiPlan = try await geminiPlan(for: messages)

    let responsesOutput = try #require(responsesPlan.inputItems.first { $0["type"] as? String == "function_call_output" })
    #expect(responsesOutput["call_id"] as? String == "call_1")
    let responsesSyntheticError = try #require(responsesOutput["output"] as? String)
    #expect(responsesSyntheticError.contains(ToolReplaySupport.syntheticToolResultErrorText))

    #expect(anthropicPlan.messages.count == 2)
    let anthropicSynthetic = try #require(anthropicPlan.messages[1].contentBlocks?.first?.toolResult)
    #expect(anthropicSynthetic.toolUseId == "call_1")
    #expect(anthropicSynthetic.isError == true)

    #expect(geminiPlan.contents.count == 2)
    let geminiSynthetic = try #require((geminiPlan.contents[1]["parts"] as? [[String: Any]])?.first?["functionResponse"] as? [String: Any])
    #expect(geminiSynthetic["id"] as? String == "call_1")
    let geminiSyntheticPayload = try #require(geminiSynthetic["response"] as? [String: Any])
    #expect((geminiSyntheticPayload["error"] as? String)?.contains(ToolReplaySupport.syntheticToolResultErrorText) == true)
  }

  @Test
  func `Cross-provider fixture downgrades opaque response text when native payload is unavailable`() async throws {
    let messages = [Message(role: .assistant, content: [
      .providerOpaque(OpaqueBlock(
        provider: OpaqueBlock.ProviderID.gemini,
        type: OpaqueBlock.GeminiType.toolResponse,
        content: "Gemini response text",
        isResponseContent: true,
      )),
      .providerOpaque(OpaqueBlock(
        provider: OpaqueBlock.ProviderID.openAIResponses,
        type: OpaqueBlock.OpenAIResponsesType.refusal,
        content: "Responses refusal text",
        isResponseContent: true,
      )),
      .providerOpaque(OpaqueBlock(
        provider: OpaqueBlock.ProviderID.anthropic,
        type: OpaqueBlock.AnthropicType.webFetchToolResult,
        content: "Fetched article body",
        isResponseContent: true,
      )),
    ])]

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(messages)
    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: false)
    let geminiPlan = try await geminiPlan(for: messages)

    let expectedTexts = [
      "Gemini response text",
      "Responses refusal text",
      "Fetched article body",
    ]

    for expectedText in expectedTexts {
      #expect(responsesTexts(in: responsesPlan).contains(expectedText))
      #expect(anthropicTexts(in: anthropicPlan).contains(expectedText))
      #expect(geminiTexts(in: geminiPlan).contains(expectedText))
    }
  }

  @Test
  func `Cross-provider fixtures preserve native reasoning replay for each provider`() async throws {
    let responsesMessages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.openAIResponses,
          type: OpaqueBlock.OpenAIResponsesType.reasoning,
          content: "Let me think through it.",
          signature: "rs_reasoning_1",
          data: "encrypted_reasoning_payload",
        )),
        .text("Visible answer."),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(responsesMessages)
    let responsesReasoningItem = try #require(responsesPlan.inputItems.first { $0["type"] as? String == "reasoning" })
    #expect(responsesReasoningItem["id"] as? String == "rs_reasoning_1")
    #expect(responsesReasoningItem["encrypted_content"] as? String == "encrypted_reasoning_payload")
    let responsesSummary = try #require(responsesReasoningItem["summary"] as? [[String: Any]])
    #expect(responsesSummary.first?["text"] as? String == "Let me think through it.")
    #expect(responsesTexts(in: responsesPlan).contains("Visible answer."))

    let anthropicMessages = [
      Message(role: .assistant, content: [
        .thinking(text: "Signed chain of thought", signature: "sig_reasoning_1"),
        .redactedThinking(data: "redacted_reasoning_payload"),
        .text("Visible answer."),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(anthropicMessages, thinkingEnabled: true)
    let anthropicAssistantBlocks = try #require(anthropicPlan.messages.first?.contentBlocks)
    let anthropicThinking = try #require(anthropicAssistantBlocks.first { $0.type == .thinking })
    #expect(anthropicThinking.thinking == "Signed chain of thought")
    #expect(anthropicThinking.signature == "sig_reasoning_1")
    let anthropicRedacted = try #require(anthropicAssistantBlocks.first { $0.type == .redactedThinking })
    #expect(anthropicRedacted.data == "redacted_reasoning_payload")
    #expect(anthropicTexts(in: anthropicPlan).contains("Visible answer."))

    let geminiMessages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.gemini,
          type: OpaqueBlock.GeminiType.thinking,
          content: "Signed Gemini reasoning",
          signature: "gem_sig_1",
        )),
        .text("Visible answer."),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let geminiPlan = try await geminiPlan(for: geminiMessages)
    let geminiThoughtPart = try #require(geminiParts(in: geminiPlan).first { ($0["thought"] as? Bool) == true })
    #expect(geminiThoughtPart["text"] as? String == "Signed Gemini reasoning")
    #expect(geminiThoughtPart["thoughtSignature"] as? String == "gem_sig_1")
    #expect(geminiTexts(in: geminiPlan).contains("Visible answer."))
  }

  @Test
  func `Cross-provider fixtures round-trip native opaque blocks when raw payload exists`() async throws {
    let responsesMessages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.openAIResponses,
          type: OpaqueBlock.OpenAIResponsesType.messageMetadata,
          data: #"{"id":"msg_annotated","status":"completed"}"#,
        )),
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.openAIResponses,
          type: OpaqueBlock.OpenAIResponsesType.annotatedOutputText,
          content: "Cited answer.",
          data: #"[{"type":"url_citation","url":"https://example.com/docs","title":"Example Docs"}]"#,
          isResponseContent: true,
        )),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(responsesMessages)
    let responsesAssistantMessage = try #require(responsesPlan.inputItems.first {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    })
    #expect(responsesAssistantMessage["id"] as? String == "msg_annotated")
    let responsesAssistantContent = try #require(responsesAssistantMessage["content"] as? [[String: Any]])
    let responsesOutput = try #require(responsesAssistantContent.first)
    #expect(responsesOutput["type"] as? String == "output_text")
    #expect(responsesOutput["text"] as? String == "Cited answer.")
    let responsesAnnotations = try #require(responsesOutput["annotations"] as? [[String: Any]])
    #expect(responsesAnnotations.first?["url"] as? String == "https://example.com/docs")
    let anthropicFromResponses = try await AnthropicReplayNormalizer.normalize(responsesMessages, thinkingEnabled: false)
    let geminiFromResponses = try await geminiPlan(for: responsesMessages)
    #expect(anthropicTexts(in: anthropicFromResponses).contains("Cited answer."))
    #expect(geminiTexts(in: geminiFromResponses).contains("Cited answer."))

    let anthropicMessages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.anthropic,
          type: OpaqueBlock.AnthropicType.webFetchToolResult,
          content: "Fetched excerpt",
          data: #"{"type":"web_fetch_tool_result","content":{"type":"web_fetch_result","url":"https://example.com/docs","title":"Example Docs","content":{"type":"text","text":"Fetched excerpt"}}}"#,
          isResponseContent: true,
        )),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(anthropicMessages, thinkingEnabled: false)
    let anthropicAssistantMessage = try #require(anthropicPlan.messages.first { $0.role == .assistant })
    let anthropicBlocks = try #require(anthropicAssistantMessage.contentBlocks)
    #expect(anthropicBlocks.contains { $0.type == .webFetchToolResult })
    let responsesFromAnthropic = try await ResponsesReplayNormalizer.normalize(anthropicMessages)
    let geminiFromAnthropic = try await geminiPlan(for: anthropicMessages)
    #expect(responsesTexts(in: responsesFromAnthropic).contains("Fetched excerpt"))
    #expect(geminiTexts(in: geminiFromAnthropic).contains("Fetched excerpt"))

    let geminiMessages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.gemini,
          type: OpaqueBlock.GeminiType.executableCode,
          content: "print(42)",
          data: #"{"language":"PYTHON","code":"print(42)"}"#,
          isResponseContent: true,
        )),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let geminiPlan = try await geminiPlan(for: geminiMessages)
    let geminiExecutablePart = try #require(geminiParts(in: geminiPlan).first { $0["executableCode"] != nil })
    let geminiExecutable = try #require(geminiExecutablePart["executableCode"] as? [String: Any])
    #expect(geminiExecutable["language"] as? String == "PYTHON")
    #expect(geminiExecutable["code"] as? String == "print(42)")
    let responsesFromGemini = try await ResponsesReplayNormalizer.normalize(geminiMessages)
    let anthropicFromGemini = try await AnthropicReplayNormalizer.normalize(geminiMessages, thinkingEnabled: false)
    #expect(responsesTexts(in: responsesFromGemini).contains("print(42)"))
    #expect(anthropicTexts(in: anthropicFromGemini).contains("print(42)"))
  }

  @Test
  func `Cross-provider fixtures fall back to text when opaque raw payload is missing`() async throws {
    let messages = [Message(role: .assistant, content: [
      .providerOpaque(OpaqueBlock(
        provider: OpaqueBlock.ProviderID.anthropic,
        type: OpaqueBlock.AnthropicType.webFetchToolResult,
        content: "Fetched excerpt",
        isResponseContent: true,
      )),
      .providerOpaque(OpaqueBlock(
        provider: OpaqueBlock.ProviderID.openAIResponses,
        type: OpaqueBlock.OpenAIResponsesType.annotatedOutputText,
        content: "Cited answer.",
        isResponseContent: true,
      )),
      .providerOpaque(OpaqueBlock(
        provider: OpaqueBlock.ProviderID.gemini,
        type: OpaqueBlock.GeminiType.executableCode,
        content: "print(42)",
        isResponseContent: true,
      )),
    ])]

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(messages)
    let anthropicPlan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: false)
    let geminiPlan = try await geminiPlan(for: messages)

    let expectedTexts = [
      "Fetched excerpt",
      "Cited answer.",
      "print(42)",
    ]

    for expectedText in expectedTexts {
      #expect(responsesTexts(in: responsesPlan).contains(expectedText))
      #expect(anthropicTexts(in: anthropicPlan).contains(expectedText))
      #expect(geminiTexts(in: geminiPlan).contains(expectedText))
    }

    #expect(responsesPlan.inputItems.allSatisfy { $0["type"] as? String == "message" })
    #expect(anthropicBlockTypes(in: anthropicPlan).allSatisfy { $0 == .text })
    #expect(geminiParts(in: geminiPlan).allSatisfy { $0["executableCode"] == nil && $0["codeExecutionResult"] == nil && $0["toolCall"] == nil && $0["toolResponse"] == nil })
  }

  @Test
  func `Cross-provider reasoning falls back to assistant text in Responses when native payload is unavailable`() async throws {
    let messages = [
      Message(role: .assistant, content: [
        .thinking(text: "Signed Anthropic reasoning", signature: "anthropic_sig_1"),
        .providerOpaque(OpaqueBlock(
          provider: OpaqueBlock.ProviderID.gemini,
          type: OpaqueBlock.GeminiType.thinking,
          content: "Signed Gemini reasoning",
          signature: "gemini_sig_1",
        )),
        .text("Visible answer."),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let responsesPlan = try await ResponsesReplayNormalizer.normalize(messages)

    let assistantMessage = try #require(responsesPlan.inputItems.first {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    })
    let assistantContent = try #require(assistantMessage["content"] as? [[String: Any]])
    let assistantTexts = assistantContent.compactMap { $0["text"] as? String }

    #expect(assistantTexts == [
      "Signed Anthropic reasoning",
      "Signed Gemini reasoning",
      "Visible answer.",
    ])
    #expect(responsesPlan.inputItems.allSatisfy { $0["type"] as? String != "reasoning" })
  }

  private func geminiPlan(for messages: [Message]) async throws -> GeminiReplayNormalizer.Plan {
    let client = GeminiClient(
      session: makeMockSession(),
      modelsEndpoint: URL(string: "https://mock.test/replay-fixtures")!,
    )
    return try await GeminiReplayNormalizer.normalize(
      messages,
      systemPrompt: nil,
      apiKey: "test-key",
      requestParts: { message, apiKey in
        try await client.requestParts(for: message, apiKey: apiKey)
      },
    )
  }

  private func responsesTexts(in plan: ResponsesReplayNormalizer.Plan) -> [String] {
    plan.inputItems.flatMap { item in
      let content = item["content"] as? [[String: Any]] ?? []
      // Assistant messages can contain native refusal content blocks where the text
      // lives under the `refusal` key rather than `text`. Surface both.
      return content.compactMap { ($0["text"] as? String) ?? ($0["refusal"] as? String) }
    }
  }

  private func anthropicTexts(in plan: AnthropicReplayNormalizer.Plan) -> [String] {
    plan.messages.flatMap { message in
      (message.contentBlocks ?? []).compactMap(\.text)
    }
  }

  private func anthropicBlockTypes(in plan: AnthropicReplayNormalizer.Plan) -> [AnthropicClient.ContentBlockType] {
    plan.messages.flatMap { message in
      (message.contentBlocks ?? []).map(\.type)
    }
  }

  private func geminiFunctionResponses(in plan: GeminiReplayNormalizer.Plan) -> [[String: Any]] {
    plan.contents.compactMap { content in
      let parts = content["parts"] as? [[String: Any]]
      return parts?.first?["functionResponse"] as? [String: Any]
    }
  }

  private func geminiParts(in plan: GeminiReplayNormalizer.Plan) -> [[String: Any]] {
    plan.contents.flatMap { content in
      (content["parts"] as? [[String: Any]]) ?? []
    }
  }

  private func geminiTexts(in plan: GeminiReplayNormalizer.Plan) -> [String] {
    plan.contents.flatMap { content in
      let parts = content["parts"] as? [[String: Any]] ?? []
      return parts.compactMap { $0["text"] as? String }
    }
  }
}
