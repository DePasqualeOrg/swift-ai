// Copyright © Anthony DePasquale

extension OpaqueBlock {
  var isAnthropicThinkingBlock: Bool {
    isAnthropicThinking || isAnthropicRedactedThinking
  }

  var isAnthropicThinking: Bool {
    provider == "anthropic" && type == "thinking"
  }

  var isAnthropicRedactedThinking: Bool {
    provider == "anthropic" && type == "redacted_thinking"
  }

  var isAnthropicCitationCarrier: Bool {
    provider == "anthropic" && (type == "web_search_tool_result" || type == "web_fetch_tool_result")
  }

  var isAnthropicNativeStructuredBlock: Bool {
    provider == "anthropic"
      && (type == "server_tool_use"
        || type == "web_search_tool_result"
        || type == "web_fetch_tool_result"
        || type == "code_execution_tool_result")
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
