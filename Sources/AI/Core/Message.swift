// Copyright © Anthony DePasquale

import Foundation

/// A message in a conversation with an LLM.
///
/// This is the library's message type, designed for API communication without app-specific dependencies.
public struct Message: Sendable, Hashable {
  /// The role of the message sender in the conversation.
  public enum Role: String, Sendable, Hashable, Codable {
    /// System-level instructions (used by most providers).
    case system
    /// Developer-level instructions (used by OpenAI Responses API).
    case developer
    /// A message from the user.
    case user
    /// A message from the assistant.
    case assistant
    /// A message containing tool results.
    case tool
  }

  /// A content item within a message.
  public enum Content: Sendable, Hashable {
    case text(String)
    case thinking(text: String, signature: String?)
    case endnotes(String)
    case redactedThinking(data: String)
    case toolCall(ToolCall)
    case toolResult(ToolResult)
    case attachment(Attachment)
    case providerOpaque(OpaqueBlock)

    var opaqueBlock: OpaqueBlock? {
      switch self {
        case let .thinking(text, signature) where signature != nil:
          OpaqueBlock(provider: "anthropic", type: "thinking", content: text, signature: signature)
        case let .redactedThinking(data):
          OpaqueBlock(provider: "anthropic", type: "redacted_thinking", data: data)
        case let .providerOpaque(opaqueBlock):
          opaqueBlock
        default:
          nil
      }
    }
  }

  /// The role of this message's sender.
  public let role: Role

  /// Ordered content items for this message.
  public let content: [Content]

  /// Creates a new message with content items.
  ///
  /// - Parameters:
  ///   - role: The role of the message sender.
  ///   - content: Ordered content items.
  public init(role: Role, content: [Content]) {
    self.role = role
    self.content = content
  }

  /// Creates a new message with text content.
  ///
  /// - Parameters:
  ///   - role: The role of the message sender.
  ///   - content: The text content of the message.
  public init(role: Role, content: String) {
    self.role = role
    self.content = [.text(content)]
  }

  /// Builds an ordered array of assistant content items from optional text components and tool calls.
  static func assistantContent(
    reasoningText: String? = nil,
    responseText: String? = nil,
    notesText: String? = nil,
    toolCalls: [ToolCall] = [],
  ) -> [Content] {
    var content: [Content] = []
    if let reasoningText, !reasoningText.isEmpty {
      content.append(.thinking(text: reasoningText, signature: nil))
    }
    if let responseText, !responseText.isEmpty {
      content.append(.text(responseText))
    }
    if let notesText, !notesText.isEmpty {
      content.append(.endnotes(notesText))
    }
    content.append(contentsOf: toolCalls.map(Content.toolCall))
    return content
  }
}

// MARK: - Orphaned Tool Call Handling

public extension Message {
  /// Returns messages with synthetic error results inserted for any tool calls that lack matching tool results.
  /// Each synthetic result is placed immediately after the message containing its tool call.
  /// This can happen when generation is canceled or times out mid-tool-call.
  static func patchingOrphanedToolCalls(_ messages: [Message]) -> [Message] {
    var toolCallIds = Set<String>()
    var toolResultIds = Set<String>()

    for message in messages {
      for item in message.content {
        switch item {
          case let .toolCall(toolCall):
            toolCallIds.insert(toolCall.id)
          case let .toolResult(toolResult):
            toolResultIds.insert(toolResult.id)
          default:
            break
        }
      }
    }

    let orphanedIds = toolCallIds.subtracting(toolResultIds)
    guard !orphanedIds.isEmpty else { return messages }

    var result: [Message] = []
    for message in messages {
      result.append(message)

      // Collect orphaned tool calls in this message
      var orphanedResults: [Content] = []
      for item in message.content {
        guard case let .toolCall(toolCall) = item, orphanedIds.contains(toolCall.id) else { continue }
        orphanedResults.append(.toolResult(ToolResult(
          name: toolCall.name,
          id: toolCall.id,
          content: [.text("Function call was not executed. The request may have been canceled or timed out.")],
          isError: true,
        )))
      }

      if !orphanedResults.isEmpty {
        result.append(Message(role: .tool, content: orphanedResults))
      }
    }

    return result
  }
}

// MARK: - Tool Collapsing Utilities

public extension Message {
  /// Returns a new message with tool_use content collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolCalls() -> Message {
    let attachments = content.compactMap { item -> Attachment? in
      guard case let .attachment(attachment) = item else { return nil }
      return attachment
    }
    var text = content.compactMap { item -> String? in
      guard case let .text(text) = item else { return nil }
      return text
    }.joined()
    for item in content {
      guard case let .toolCall(toolCall) = item else { continue }
      let paramsJSON: String = if let data = toolCall.parametersToData(),
                                  let jsonString = String(data: data, encoding: .utf8)
      {
        jsonString
      } else {
        "{}"
      }
      text += "\n\n[Called tool \"\(toolCall.name)\" with: \(paramsJSON)]"
    }

    var collapsed: [Content] = []
    if !text.isEmpty {
      collapsed.append(.text(text))
    }
    collapsed.append(contentsOf: attachments.map(Content.attachment))
    return Message(role: role, content: collapsed)
  }

  /// Returns a new message with tool_result content collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolResults() -> Message {
    let attachments = content.compactMap { item -> Attachment? in
      guard case let .attachment(attachment) = item else { return nil }
      return attachment
    }
    var text = content.compactMap { item -> String? in
      guard case let .text(text) = item else { return nil }
      return text
    }.joined()

    for item in content {
      guard case let .toolResult(toolResult) = item else { continue }
      if toolResult.isError == true {
        let errorText = toolResult.content.compactMap { content -> String? in
          if case let .text(str) = content { return str }
          return nil
        }.joined(separator: " ")
        text += "\n\n[Error from tool \"\(toolResult.name)\": \(errorText)]"
      } else {
        let resultText = toolResult.content.compactMap { content -> String? in
          if case let .text(str) = content { return str }
          return nil
        }.joined(separator: " ")
        text += "\n\n[Result from tool \"\(toolResult.name)\": \(resultText)]"
      }
    }

    var collapsed: [Content] = []
    if !text.isEmpty {
      collapsed.append(.text(text))
    }
    collapsed.append(contentsOf: attachments.map(Content.attachment))
    return Message(role: .user, content: collapsed)
  }
}
