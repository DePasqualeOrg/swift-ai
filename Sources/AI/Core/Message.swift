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

  /// Ordered blocks contained in a message.
  public enum Block: Sendable, Hashable {
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

  /// Ordered content blocks for this message.
  public let blocks: [Block]

  /// Creates a new message.
  ///
  /// - Parameters:
  ///   - role: The role of the message sender.
  ///   - blocks: Ordered content blocks.
  public init(role: Role, blocks: [Block]) {
    self.role = role
    self.blocks = blocks
  }
}

// MARK: - Orphaned Tool Call Handling

public extension Message {
  /// Returns messages with synthetic error results appended for any tool calls that lack matching tool results.
  /// This can happen when generation is canceled or times out mid-tool-call.
  static func patchingOrphanedToolCalls(_ messages: [Message]) -> [Message] {
    var toolCallIds = Set<String>()
    var toolResultIds = Set<String>()

    for message in messages {
      for block in message.blocks {
        switch block {
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

    var orphanedResults: [ToolResult] = []
    for message in messages {
      for block in message.blocks {
        guard case let .toolCall(toolCall) = block, orphanedIds.contains(toolCall.id) else { continue }
        orphanedResults.append(ToolResult(
          name: toolCall.name,
          id: toolCall.id,
          content: [.text("Function call was not executed. The request may have been canceled or timed out.")],
          isError: true,
        ))
      }
    }

    return messages + [Message(role: .tool, blocks: orphanedResults.map(Block.toolResult))]
  }
}

// MARK: - Tool Collapsing Utilities

public extension Message {
  /// Returns a new message with tool_use blocks collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolCalls() -> Message {
    let attachments = blocks.compactMap { block -> Attachment? in
      guard case let .attachment(attachment) = block else { return nil }
      return attachment
    }
    var text = blocks.compactMap { block -> String? in
      guard case let .text(text) = block else { return nil }
      return text
    }.joined()
    for block in blocks {
      guard case let .toolCall(toolCall) = block else { continue }
      let paramsJSON: String = if let data = toolCall.parametersToData(),
                                  let jsonString = String(data: data, encoding: .utf8)
      {
        jsonString
      } else {
        "{}"
      }
      text += "\n\n[Called tool \"\(toolCall.name)\" with: \(paramsJSON)]"
    }

    var collapsedBlocks: [Block] = []
    if !text.isEmpty {
      collapsedBlocks.append(.text(text))
    }
    collapsedBlocks.append(contentsOf: attachments.map(Block.attachment))
    return Message(role: role, blocks: collapsedBlocks)
  }

  /// Returns a new message with tool_result blocks collapsed to descriptive text.
  /// Used when a provider can't satisfy metadata requirements for historical tool turns.
  func collapsingToolResults() -> Message {
    let attachments = blocks.compactMap { block -> Attachment? in
      guard case let .attachment(attachment) = block else { return nil }
      return attachment
    }
    var text = blocks.compactMap { block -> String? in
      guard case let .text(text) = block else { return nil }
      return text
    }.joined()

    for block in blocks {
      guard case let .toolResult(toolResult) = block else { continue }
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

    var collapsedBlocks: [Block] = []
    if !text.isEmpty {
      collapsedBlocks.append(.text(text))
    }
    collapsedBlocks.append(contentsOf: attachments.map(Block.attachment))
    return Message(role: .user, blocks: collapsedBlocks)
  }
}
