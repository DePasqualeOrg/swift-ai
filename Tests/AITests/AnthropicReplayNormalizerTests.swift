// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct AnthropicReplayNormalizerTests {
  @Test
  func `Anthropic normalizer emits request ready params for mixed tool history`() async throws {
    let messages = [
      Message(role: .system, content: "Follow policy"),
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
        .toolCall(ToolCall(name: "lookup", id: "call_2", parameters: ["id": "42"])),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "stale", id: "call_stray", content: .text("Stray result"))),
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text("Matched result"))),
      ]),
      Message(role: .user, content: "Continue"),
    ]

    let plan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: false)

    #expect(plan.systemTexts == ["Follow policy"])
    #expect(plan.messages.count == 5)
    #expect(plan.messages[0].role == .assistant)
    #expect(plan.messages[0].contentBlocks?.first?.type == .toolUse)

    let matchedResultBlock = try #require(plan.messages[1].contentBlocks?.first?.toolResult)
    #expect(plan.messages[1].role == .user)
    #expect(matchedResultBlock.toolUseId == "call_1")

    let syntheticResultBlock = try #require(plan.messages[2].contentBlocks?.first?.toolResult)
    #expect(syntheticResultBlock.toolUseId == "call_2")
    #expect(syntheticResultBlock.isError == true)

    let collapsedStrayText = plan.messages[3].contentBlocks?.compactMap(\.text).joined(separator: "\n")
    #expect(collapsedStrayText == "\n\n[Result from tool \"stale\": Stray result]")
  }

  @Test
  func `Anthropic normalizer collapses unsigned thinking tool exchanges after structural repair`() async throws {
    let messages = [
      Message(role: .assistant, content: [
        .text("Intermediate reasoning"),
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
      ]),
      Message(role: .tool, content: [
        .providerOpaque(OpaqueBlock(
          provider: "gemini",
          type: "codeExecutionResult",
          content: "Execution output: 42",
          isResponseContent: true,
        )),
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text("Matched result"))),
      ]),
      Message(role: .user, content: "Continue"),
    ]

    let plan = try await AnthropicReplayNormalizer.normalize(messages, thinkingEnabled: true)
    #expect(plan.messages.count == 4)
    let collapsedResultText = plan.messages[1].contentBlocks?.compactMap(\.text).joined(separator: "\n")
    let collapsedOpaqueText = plan.messages[2].contentBlocks?.compactMap(\.text).joined(separator: "\n")
    #expect(collapsedOpaqueText?.contains("Execution output: 42") == true)
    #expect(collapsedResultText?.contains(#"[Result from tool "search": Matched result]"#) == true)
  }
}
