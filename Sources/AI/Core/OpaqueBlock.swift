// Copyright © Anthony DePasquale

import Foundation

/// A generic container for provider-specific content blocks that must be round-tripped.
///
/// Used to preserve provider metadata (e.g., Anthropic thinking blocks with cryptographic signatures)
/// through the response → Core Data → message rebuild cycle.
///
/// Cross-provider replay follows a simple contract:
/// - same-provider clients should prefer `data` for lossless native replay
/// - other providers should fall back to `content` when `isResponseContent` is true
/// so visible output survives provider switches even when native metadata does not.
///
/// `provider` and `type` intentionally remain raw strings because this value is serialized into
/// Core Data by the consuming app. A closed enum would make previously persisted unknown block
/// kinds fail to decode as providers add new structured output types over time.
public struct OpaqueBlock: Sendable, Hashable, Codable {
  /// The provider that produced this block (e.g., "anthropic", "gemini").
  public let provider: String

  /// The block type (e.g., "thinking", "redacted_thinking").
  public let type: String

  /// The thinking text content, if any. Nil for redacted blocks.
  public let content: String?

  /// The verification token / cryptographic signature.
  public let signature: String?

  /// Encrypted content for redacted blocks.
  public let data: String?

  /// Whether `content` should be surfaced through `GenerationResponse.responseText`
  /// and downgraded to plain text by non-native provider clients during replay.
  /// Used for server-side tool output (e.g., code execution results, fetched web content)
  /// that is part of the response but must be round-tripped as structured data.
  public var isResponseContent: Bool = false

  enum CodingKeys: String, CodingKey {
    case provider, type, content, signature, data, isResponseContent
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    provider = try container.decode(String.self, forKey: .provider)
    type = try container.decode(String.self, forKey: .type)
    content = try container.decodeIfPresent(String.self, forKey: .content)
    signature = try container.decodeIfPresent(String.self, forKey: .signature)
    data = try container.decodeIfPresent(String.self, forKey: .data)
    // Default to false for data persisted before this field was added
    isResponseContent = try container.decodeIfPresent(Bool.self, forKey: .isResponseContent) ?? false
  }

  public init(provider: String, type: String, content: String? = nil,
              signature: String? = nil, data: String? = nil,
              isResponseContent: Bool = false)
  {
    self.provider = provider
    self.type = type
    self.content = content
    self.signature = signature
    self.data = data
    self.isResponseContent = isResponseContent
  }
}

public extension OpaqueBlock {
  enum ProviderID {
    public static let anthropic = "anthropic"
    public static let openAIResponses = "openai-responses"
    public static let openAIChatCompletions = "openai-chat-completions"
    public static let gemini = "gemini"
  }

  enum AnthropicType {
    public static let thinking = "thinking"
    public static let redactedThinking = "redacted_thinking"
    public static let serverToolUse = "server_tool_use"
    public static let webSearchToolResult = "web_search_tool_result"
    public static let webFetchToolResult = "web_fetch_tool_result"
    public static let codeExecutionToolResult = "code_execution_tool_result"
  }

  enum OpenAIResponsesType {
    public static let annotatedOutputText = "annotated_output_text"
    public static let refusal = "refusal"
    public static let messageMetadata = "message_metadata"
    public static let reasoning = "reasoning"
  }

  enum OpenAIChatCompletionsType {
    public static let refusal = "refusal"
    public static let annotations = "annotations"
  }

  enum GeminiType {
    public static let thinking = "thinking"
    public static let executableCode = "executableCode"
    public static let codeExecutionResult = "codeExecutionResult"
    public static let toolCall = "toolCall"
    public static let toolResponse = "toolResponse"
    public static let urlContextMetadata = "urlContextMetadata"
  }
}

extension OpaqueBlock {
  var portableReplayText: String? {
    guard isResponseContent else { return nil }
    return content
  }

  var nativeReplayTarget: ReplayTarget? {
    switch provider {
      case Self.ProviderID.anthropic:
        .anthropic
      case Self.ProviderID.openAIResponses:
        .responses
      case Self.ProviderID.openAIChatCompletions:
        .chatCompletions
      case Self.ProviderID.gemini:
        .gemini
      default:
        nil
    }
  }

  func canReplayNatively(on target: ReplayTarget) -> Bool {
    switch target {
      case .anthropic:
        if provider != Self.ProviderID.anthropic {
          return false
        }
        return type == Self.AnthropicType.thinking
          || type == Self.AnthropicType.redactedThinking
          || ((type == Self.AnthropicType.serverToolUse
              || type == Self.AnthropicType.webSearchToolResult
              || type == Self.AnthropicType.webFetchToolResult
              || type == Self.AnthropicType.codeExecutionToolResult) && data != nil)

      case .responses:
        if provider != Self.ProviderID.openAIResponses {
          return false
        }
        if type == Self.OpenAIResponsesType.annotatedOutputText
          || type == Self.OpenAIResponsesType.refusal
          || type == Self.OpenAIResponsesType.reasoning
        {
          return true
        }
        return data != nil

      case .chatCompletions:
        return provider == Self.ProviderID.openAIChatCompletions
          && (type == Self.OpenAIChatCompletionsType.refusal || type == Self.OpenAIChatCompletionsType.annotations)

      case .gemini:
        if provider != Self.ProviderID.gemini {
          return false
        }
        if type == Self.GeminiType.thinking {
          return true
        }
        return (type == Self.GeminiType.executableCode
          || type == Self.GeminiType.codeExecutionResult
          || type == Self.GeminiType.toolCall
          || type == Self.GeminiType.toolResponse) && data != nil
    }
  }

  func replayDowngradeText(for target: ReplayTarget) -> String? {
    guard let text = portableReplayText, !text.isEmpty else {
      return nil
    }
    return canReplayNatively(on: target) ? nil : text
  }
}
