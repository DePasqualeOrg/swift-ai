// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct MessageTests {
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

    let patched = Message.patchingOrphanedToolCalls(messages)

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
    #expect(toolResult.content == [.text("Function call was not executed. The request may have been canceled or timed out.")])
  }
}
