// Copyright © Anthony DePasquale

@testable import AI
import Foundation

enum ReplayFixtures {
  static let continueText = "Continue"
  static let lateToolResultText = "Late result body"
  static let matchedToolResultText = "Matched result"
  static let strayToolResultText = "Stray result"

  static func lateToolResultHistory() -> [Message] {
    [
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
      ]),
      Message(role: .user, content: continueText),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text(lateToolResultText))),
      ]),
    ]
  }

  static func mixedToolTurnHistory() -> [Message] {
    [
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
        .toolCall(ToolCall(name: "lookup", id: "call_2", parameters: ["id": "42"])),
      ]),
      Message(role: .tool, content: [
        .toolResult(ToolResult(name: "search", id: "call_1", content: .text(matchedToolResultText))),
        .toolResult(ToolResult(name: "stale", id: "call_stray", content: .text(strayToolResultText))),
      ]),
      Message(role: .user, content: continueText),
    ]
  }

  static func trailingUnresolvedToolCallHistory() -> [Message] {
    [
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "search", id: "call_1", parameters: ["query": "swift"])),
      ]),
    ]
  }
}
