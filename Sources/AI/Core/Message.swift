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

  /// The role of this message's sender.
  public let role: Role

  /// The text content of the message, if any.
  public let content: String?

  /// File attachments included with the message (images, documents, etc.).
  public let attachments: [Attachment]

  /// Tool calls made by the assistant in this message.
  public let toolCalls: [GenerationResponse.ToolCall]?

  /// Results from tool executions, when `role` is `.tool`.
  public let toolResults: [ToolResult]?

  /// Provider-specific opaque blocks for round-tripping (e.g., Anthropic thinking signatures).
  public let opaqueBlocks: [OpaqueBlock]?

  /// Creates a new message.
  ///
  /// - Parameters:
  ///   - role: The role of the message sender.
  ///   - content: The text content of the message.
  ///   - attachments: File attachments to include.
  ///   - toolCalls: Tool calls made by the assistant.
  ///   - toolResults: Results from tool executions.
  ///   - opaqueBlocks: Provider-specific opaque blocks for round-tripping.
  public init(
    role: Role,
    content: String?,
    attachments: [Attachment] = [],
    toolCalls: [GenerationResponse.ToolCall]? = nil,
    toolResults: [ToolResult]? = nil,
    opaqueBlocks: [OpaqueBlock]? = nil
  ) {
    self.role = role
    self.content = content
    self.attachments = attachments
    self.toolCalls = toolCalls
    self.toolResults = toolResults
    self.opaqueBlocks = opaqueBlocks
  }
}

// MARK: - Tool Collapsing Utilities

public extension Message {
  /// Returns a new message with tool_use blocks collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolCalls() -> Message {
    guard let toolCalls, !toolCalls.isEmpty else { return self }
    var text = content ?? ""
    for toolCall in toolCalls {
      let paramsJSON: String = if let data = toolCall.parametersToData(),
                                  let jsonString = String(data: data, encoding: .utf8)
      {
        jsonString
      } else {
        "{}"
      }
      text += "\n\n[Called tool \"\(toolCall.name)\" with: \(paramsJSON)]"
    }
    return Message(
      role: role,
      content: text,
      attachments: attachments,
      toolCalls: nil,
      toolResults: nil,
      opaqueBlocks: nil
    )
  }

  /// Returns a new message with tool_result blocks collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolResults() -> Message {
    guard let toolResults, !toolResults.isEmpty else { return self }
    var text = content ?? ""
    for toolResult in toolResults {
      if let isError = toolResult.isError, isError {
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
    return Message(
      role: .user,
      content: text,
      attachments: attachments,
      toolCalls: nil,
      toolResults: nil,
      opaqueBlocks: nil
    )
  }
}
