// Copyright © Anthony DePasquale

import Foundation
import os.log

/// A tool call made by the model.
public struct ToolCall: Sendable, Codable, Hashable {
  /// The name of the tool to call.
  public var name: String

  /// The unique identifier for this tool call.
  public let id: String

  /// The parameters to pass to the tool.
  public var parameters: [String: Value]

  /// Provider-specific metadata (e.g., Gemini's thoughtSignature).
  public var providerMetadata: [String: String]?

  /// Creates a new tool call.
  ///
  /// - Parameters:
  ///   - name: The name of the tool to call.
  ///   - id: The unique identifier for this tool call.
  ///   - parameters: The parameters to pass to the tool.
  ///   - providerMetadata: Provider-specific metadata.
  public init(name: String, id: String, parameters: [String: Value], providerMetadata: [String: String]? = nil) {
    self.name = name
    self.id = id
    self.parameters = parameters
    self.providerMetadata = providerMetadata
  }

  /// Converts the parameters dictionary to JSON Data.
  /// - Returns: Data representation of parameters, or nil if serialization fails.
  public func parametersToData() -> Data? {
    do {
      var anyDictionary: [String: any Sendable] = [:]
      for (key, value) in parameters {
        anyDictionary[key] = value.toAny()
      }
      return try JSONSerialization.data(withJSONObject: anyDictionary, options: [])
    } catch {
      generationResponseLogger.error("Error serializing parameters to Data: \(error.localizedDescription)")
      return nil
    }
  }

  /// Converts JSON data to a dictionary of Value parameters.
  /// - Parameter data: The JSON data to convert.
  /// - Returns: Dictionary of parameter names to Value objects.
  public static func dataToParameters(_ data: Data) -> [String: Value]? {
    do {
      var parameters: [String: Value] = [:]
      if let decodedParameters = try JSONSerialization.jsonObject(with: data, options: []) as? [String: any Sendable] {
        for (key, value) in decodedParameters {
          parameters[key] = try Value.fromAny(value)
        }
      }
      return parameters
    } catch {
      generationResponseLogger.error("Error while decoding parameters for ToolCall: \(error.localizedDescription)")
      return nil
    }
  }
}

/// A response from an LLM generation request.
///
/// The canonical content model is the ordered `content` array.
public struct GenerationResponse: Sendable, Hashable {
  /// Metadata about the generation response.
  public struct Metadata: Sendable, Hashable {
    /// The unique identifier for this response from the provider.
    public var responseId: String?

    /// The model that generated the response.
    public var model: String?

    /// When the response was created.
    public var createdAt: Date?

    /// The reason the model stopped generating.
    public var finishReason: FinishReason?

    /// Number of tokens in the input/prompt.
    public var inputTokens: Int?

    /// Number of tokens in the output/response.
    public var outputTokens: Int?

    /// Total tokens used (input + output).
    public var totalTokens: Int?

    /// Tokens written to cache (Anthropic, Gemini).
    public var cacheCreationInputTokens: Int?

    /// Tokens read from cache (Anthropic, Gemini).
    public var cacheReadInputTokens: Int?

    /// Tokens used for reasoning (Anthropic, OpenAI, Gemini).
    public var reasoningTokens: Int?

    /// Creates new response metadata.
    public init(
      responseId: String? = nil,
      model: String? = nil,
      createdAt: Date? = nil,
      finishReason: FinishReason? = nil,
      inputTokens: Int? = nil,
      outputTokens: Int? = nil,
      totalTokens: Int? = nil,
      cacheCreationInputTokens: Int? = nil,
      cacheReadInputTokens: Int? = nil,
      reasoningTokens: Int? = nil,
    ) {
      self.responseId = responseId
      self.model = model
      self.createdAt = createdAt
      self.finishReason = finishReason
      self.inputTokens = inputTokens
      self.outputTokens = outputTokens
      self.totalTokens = totalTokens
      self.cacheCreationInputTokens = cacheCreationInputTokens
      self.cacheReadInputTokens = cacheReadInputTokens
      self.reasoningTokens = reasoningTokens
    }
  }

  /// The reason the model stopped generating.
  public enum FinishReason: String, Sendable, Hashable {
    /// The model reached a natural stopping point.
    case stop
    /// The response was truncated due to reaching the maximum token limit.
    case maxTokens
    /// The model made a tool call and is waiting for results.
    case toolUse
    /// The response was filtered due to content policy.
    case contentFilter
    /// The model refused to generate a response due to a policy violation.
    case refusal
    /// A long-running turn was paused and can be resumed.
    case pauseTurn
    /// An unrecognized finish reason from the provider.
    case other
  }

  /// Ordered content items returned by the model.
  public var content: [Message.Content]

  /// Metadata about the response (token usage, finish reason, etc.).
  public var metadata: Metadata?

  /// The assistant message representing this response, suitable for adding to conversation history.
  public var message: Message {
    Message(role: .assistant, content: content)
  }

  /// Creates a new generation response.
  ///
  /// - Parameters:
  ///   - content: Ordered content items emitted by the model.
  ///   - metadata: Metadata about the response.
  public init(content: [Message.Content] = [], metadata: Metadata? = nil) {
    self.content = content
    self.metadata = metadata
  }

  // MARK: - Convenience Accessors

  /// The joined response text from all `.text` content items and any opaque blocks
  /// flagged as response content (e.g., server-side tool output), or `nil` if there are none.
  public var responseText: String? {
    let text = content.compactMap { item -> String? in
      switch item {
        case let .text(text):
          return text
        case let .providerOpaque(opaque) where opaque.isResponseContent:
          return opaque.content
        default:
          return nil
      }
    }.joined()
    return text.isEmpty ? nil : text
  }

  /// The joined reasoning text from all `.thinking` content items and any opaque blocks
  /// containing thinking content (e.g., Gemini signed thinking), or `nil` if there are none.
  public var reasoningText: String? {
    let text = content.compactMap { item -> String? in
      switch item {
        case let .thinking(text, _):
          return text
        case let .providerOpaque(opaque) where opaque.type == "thinking":
          return opaque.content
        default:
          return nil
      }
    }.joined()
    return text.isEmpty ? nil : text
  }

  /// The joined endnotes text from all `.endnotes` content items, or `nil` if there are none.
  public var endnotesText: String? {
    let text = content.compactMap { item -> String? in
      guard case let .endnotes(text) = item else { return nil }
      return text
    }.joined()
    return text.isEmpty ? nil : text
  }

  /// All tool calls from `.toolCall` content items.
  public var toolCalls: [ToolCall] {
    content.compactMap { item in
      guard case let .toolCall(toolCall) = item else { return nil }
      return toolCall
    }
  }
}

private let generationResponseLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "GenerationResponse")
