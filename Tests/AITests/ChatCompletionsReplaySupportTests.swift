// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ChatCompletionsReplaySupportTests {
  @Test
  func `Patching orphaned tool calls inserts synthetic results after each affected message`() throws {
    let orphanedCall = ToolCall(
      name: "search",
      id: "call_orphaned",
      parameters: ["query": .string("history")],
    )
    let matchedCall = ToolCall(
      name: "lookup",
      id: "call_matched",
      parameters: ["id": .string("42")],
    )
    let matchedResult = ToolResult(
      name: "lookup",
      id: "call_matched",
      content: [.text("Found record 42")],
    )

    let messages = [
      Message(role: .assistant, content: [
        .text("Let me check that."),
        .toolCall(orphanedCall),
        .toolCall(matchedCall),
      ]),
      Message(role: .tool, content: [
        .toolResult(matchedResult),
      ]),
      Message(role: .user, content: "Continue"),
    ]

    let patched = ChatCompletionsReplaySupport.patchingOrphanedToolCalls(messages)

    #expect(patched.count == 4)
    #expect(patched[0] == messages[0])
    #expect(patched[2] == messages[1])
    #expect(patched[3] == messages[2])

    let syntheticToolMessage = patched[1]
    #expect(syntheticToolMessage.role == .tool)
    #expect(syntheticToolMessage.content.count == 1)

    guard case let .toolResult(toolResult) = try #require(syntheticToolMessage.content.first) else {
      Issue.record("Expected synthetic tool result")
      return
    }

    #expect(toolResult.name == "search")
    #expect(toolResult.id == "call_orphaned")
    #expect(toolResult.isError == true)
    #expect(toolResult.content == [.text(ToolReplaySupport.syntheticToolResultErrorText)])
  }

  @Test
  func `Patching orphaned tool calls treats later matching results as satisfied`() {
    let toolCall = ToolCall(
      name: "search",
      id: "call_late",
      parameters: ["query": .string("history")],
    )
    let lateResult = ToolResult(
      name: "search",
      id: "call_late",
      content: [.text("Late result")],
    )

    let messages = [
      Message(role: .assistant, content: [
        .text("Let me check that."),
        .toolCall(toolCall),
      ]),
      Message(role: .user, content: "Still waiting"),
      Message(role: .tool, content: [
        .toolResult(lateResult),
      ]),
    ]

    #expect(ChatCompletionsReplaySupport.patchingOrphanedToolCalls(messages) == messages)
  }

  @Test
  func `Patching orphaned tool calls accepts matching results spread across consecutive tool messages`() {
    let firstCall = ToolCall(
      name: "search",
      id: "call_1",
      parameters: ["query": .string("history")],
    )
    let secondCall = ToolCall(
      name: "lookup",
      id: "call_2",
      parameters: ["id": .string("42")],
    )

    let messages = [
      Message(role: .assistant, content: [
        .toolCall(firstCall),
        .toolCall(secondCall),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text("Result 1"))),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "lookup", id: "call_2", content: .text("Result 2"))),
      ]),
    ]

    #expect(ChatCompletionsReplaySupport.patchingOrphanedToolCalls(messages) == messages)
  }
}
