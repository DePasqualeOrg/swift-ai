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

extension Message.Content {
  func portableReplayText(
    includeEndnotes: Bool = true,
    attachmentFallback: ((Attachment) -> String)? = nil,
  ) -> String? {
    switch self {
      case let .text(text) where !text.isEmpty:
        text
      case let .endnotes(text) where includeEndnotes && !text.isEmpty:
        text
      case let .providerOpaque(opaque):
        opaque.portableReplayText
      case let .attachment(attachment):
        attachmentFallback?(attachment)
      default:
        nil
    }
  }
}

// MARK: - Tool Collapsing Utilities

public extension Message {
  func replayableTextSegments(
    includeEndnotes: Bool = true,
    attachmentFallback: ((Attachment) -> String)? = nil,
  ) -> [String] {
    content.compactMap { $0.portableReplayText(includeEndnotes: includeEndnotes, attachmentFallback: attachmentFallback) }
  }

  var hasToolCalls: Bool {
    content.contains {
      if case .toolCall = $0 { return true }
      return false
    }
  }

  var hasToolResults: Bool {
    content.contains {
      if case .toolResult = $0 { return true }
      return false
    }
  }

  var toolCalls: [ToolCall] {
    content.compactMap {
      guard case let .toolCall(toolCall) = $0 else { return nil }
      return toolCall
    }
  }

  /// Collects visible text from content that should survive lossy history rewrites,
  /// including provider-opaque response text marked as portable display content.
  private func collapsedVisibleText() -> String {
    replayableTextSegments().joined(separator: "\n\n")
  }

  /// Returns a new message with tool_use content collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolCalls() -> Message {
    let attachments = content.compactMap { item -> Attachment? in
      guard case let .attachment(attachment) = item else { return nil }
      return attachment
    }
    var text = collapsedVisibleText()
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
    var text = collapsedVisibleText()

    for item in content {
      guard case let .toolResult(toolResult) = item else { continue }
      let label = toolResult.isError == true ? "Error from" : "Result from"
      let resultText = toolResult.content.map(\.fallbackDescription).joined(separator: " ")
      text += "\n\n[\(label) tool \"\(toolResult.name)\": \(resultText)]"
    }

    var collapsed: [Content] = []
    if !text.isEmpty {
      collapsed.append(.text(text))
    }
    collapsed.append(contentsOf: attachments.map(Content.attachment))
    return Message(role: .user, content: collapsed)
  }
}
