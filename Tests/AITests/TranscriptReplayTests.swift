// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct TranscriptReplayTests {
  @Test
  func `Replay prep patches orphaned tool calls for provider requests`() throws {
    let messages: [Message] = [
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
      ]),
      Message(role: .user, content: "Continue"),
    ]

    let preparedMessages = TranscriptReplay.prepare(messages, for: .gemini)

    #expect(preparedMessages.count == 3)

    let syntheticResultMessage = try #require(preparedMessages[safe: 1])
    #expect(syntheticResultMessage.role == .tool)
    let toolResult = try #require(syntheticResultMessage.content.first?.toolResult)
    #expect(toolResult.id == "call_1")
    #expect(toolResult.isError == true)
  }

  @Test
  func `Anthropic thinking replay collapses tool history without native thinking blocks`() {
    let messages: [Message] = [
      Message(role: .assistant, content: [
        .text("Intermediate reasoning"),
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text("Result body"))),
      ]),
      Message(role: .user, content: "Continue"),
    ]

    let preparedMessages = TranscriptReplay.prepare(messages, for: .anthropic(thinkingEnabled: true))

    #expect(preparedMessages.count == 3)
    #expect(preparedMessages[0].role == .assistant)
    #expect(preparedMessages[1].role == .user)
    #expect(preparedMessages[2].role == .user)
    #expect(preparedMessages[0].content == [.text("Intermediate reasoning\n\n[Called tool \"search\" with: {\"query\":\"swift\"}]")])
    #expect(preparedMessages[1].content == [.text("\n\n[Result from tool \"search\": Result body]")])
    #expect(preparedMessages[2].content == [.text("Continue")])
  }

  @Test
  func `Anthropic thinking replay preserves native tool history when signed thinking exists`() {
    let messages: [Message] = [
      Message(role: .assistant, content: [
        .thinking(text: "Reasoning", signature: "sig_123"),
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text("Result body"))),
      ]),
    ]

    let preparedMessages = TranscriptReplay.prepare(messages, for: .anthropic(thinkingEnabled: true))

    #expect(preparedMessages == messages)
  }
}

private extension Message.Content {
  var toolResult: ToolResult? {
    guard case let .toolResult(toolResult) = self else { return nil }
    return toolResult
  }
}

private extension Array {
  subscript(safe index: Int) -> Element? {
    guard indices.contains(index) else { return nil }
    return self[index]
  }
}
