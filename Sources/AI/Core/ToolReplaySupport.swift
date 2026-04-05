// Copyright © Anthony DePasquale

import Foundation

enum ToolReplaySupport {
  static let syntheticToolResultErrorText = "Function call was not executed. The request may have been canceled or timed out."

  static func syntheticToolResultContent(for toolCall: ToolCall) -> Message.Content {
    .toolResult(ToolResult(
      name: toolCall.name,
      id: toolCall.id,
      content: [.text(syntheticToolResultErrorText)],
      isError: true,
    ))
  }

  static func syntheticToolResultMessage(for toolCalls: [ToolCall]) -> Message? {
    guard !toolCalls.isEmpty else { return nil }
    return Message(role: .tool, content: toolCalls.map(syntheticToolResultContent(for:)))
  }
}
