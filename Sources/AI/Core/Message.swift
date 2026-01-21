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

  /// Creates a new message.
  ///
  /// - Parameters:
  ///   - role: The role of the message sender.
  ///   - content: The text content of the message.
  ///   - attachments: File attachments to include.
  ///   - toolCalls: Tool calls made by the assistant.
  ///   - toolResults: Results from tool executions.
  public init(
    role: Role,
    content: String?,
    attachments: [Attachment] = [],
    toolCalls: [GenerationResponse.ToolCall]? = nil,
    toolResults: [ToolResult]? = nil
  ) {
    self.role = role
    self.content = content
    self.attachments = attachments
    self.toolCalls = toolCalls
    self.toolResults = toolResults
  }
}
