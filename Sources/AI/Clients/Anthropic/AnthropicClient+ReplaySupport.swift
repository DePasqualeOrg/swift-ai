// Copyright © Anthony DePasquale

extension OpaqueBlock {
  var isAnthropicThinkingBlock: Bool {
    isAnthropicThinking || isAnthropicRedactedThinking
  }

  var isAnthropicThinking: Bool {
    provider == Self.ProviderID.anthropic && type == Self.AnthropicType.thinking
  }

  var isAnthropicRedactedThinking: Bool {
    provider == Self.ProviderID.anthropic && type == Self.AnthropicType.redactedThinking
  }

  var isAnthropicCitationCarrier: Bool {
    provider == Self.ProviderID.anthropic
      && (type == Self.AnthropicType.webSearchToolResult || type == Self.AnthropicType.webFetchToolResult)
  }

  var isAnthropicNativeStructuredBlock: Bool {
    provider == Self.ProviderID.anthropic
      && (type == Self.AnthropicType.serverToolUse
        || type == Self.AnthropicType.webSearchToolResult
        || type == Self.AnthropicType.webFetchToolResult
        || type == Self.AnthropicType.codeExecutionToolResult)
  }
}

extension Message.Content {
  var isAnthropicThinkingContent: Bool {
    switch self {
      case let .thinking(_, signature):
        signature != nil
      case .redactedThinking:
        true
      case let .providerOpaque(opaque):
        opaque.isAnthropicThinkingBlock
      default:
        false
    }
  }
}

extension Message {
  var hasNativeAnthropicThinkingBlocks: Bool {
    content.contains { $0.isAnthropicThinkingContent }
  }
}
