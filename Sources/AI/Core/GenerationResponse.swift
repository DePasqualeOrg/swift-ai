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
/// The canonical content model is the ordered `blocks` array. Flattened text/tool-call
/// projections are derived convenience helpers only.
public struct GenerationResponse: Sendable, Hashable {
  /// Text content returned by the model.
  @available(*, deprecated, message: "Use blocks as the canonical response content model.")
  public struct Texts: Sendable, Hashable {
    /// Reasoning or chain-of-thought content (when using reasoning models).
    public var reasoning: String?

    /// The main response text from the model.
    public var response: String?

    /// Additional notes or annotations from the model.
    public var notes: String?

    /// Creates a new texts container.
    public init(reasoning: String? = nil, response: String? = nil, notes: String? = nil) {
      self.reasoning = reasoning
      self.response = response
      self.notes = notes
    }
  }

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

  @available(*, deprecated, renamed: "ToolCall")
  public typealias ToolCall = AI.ToolCall

  /// Ordered content blocks returned by the model.
  public var blocks: [Message.Block]

  /// Metadata about the response (token usage, finish reason, etc.).
  public var metadata: Metadata?

  /// The assistant message representing this response, suitable for adding to conversation history.
  public var message: Message {
    Message(role: .assistant, blocks: blocks)
  }

  /// Creates a new generation response.
  ///
  /// - Parameters:
  ///   - blocks: Ordered content blocks emitted by the model.
  ///   - metadata: Metadata about the response.
  public init(blocks: [Message.Block] = [], metadata: Metadata? = nil) {
    self.blocks = blocks
    self.metadata = metadata
  }

  /// Creates a new generation response from the deprecated flattened surface.
  @available(*, deprecated, message: "Use init(blocks:metadata:) instead.")
  public init(texts: Texts = Texts(), toolCalls: [ToolCall] = [], metadata: Metadata? = nil, opaqueBlocks: [OpaqueBlock]? = nil) {
    var blocks = (opaqueBlocks ?? []).map(Self.block(from:))
    let hasThinkingBlocks = blocks.contains { block in
      switch block {
        case .thinking, .redactedThinking:
          true
        default:
          false
      }
    }
    if let reasoning = texts.reasoning, !reasoning.isEmpty, !hasThinkingBlocks {
      blocks.append(.thinking(text: reasoning, signature: nil))
    }
    if let response = texts.response, !response.isEmpty {
      blocks.append(.text(response))
    }
    if let notes = texts.notes, !notes.isEmpty {
      blocks.append(.endnotes(notes))
    }
    blocks.append(contentsOf: toolCalls.map(Message.Block.toolCall))
    self.init(blocks: blocks, metadata: metadata)
  }

  /// Deprecated flattened text projection derived from `blocks`.
  @available(*, deprecated, message: "Use blocks as the canonical response content model.")
  public var texts: Texts {
    let reasoning = blocks.compactMap { block -> String? in
      guard case let .thinking(text, _) = block else { return nil }
      return text
    }.joined()
    let response = blocks.compactMap { block -> String? in
      guard case let .text(text) = block else { return nil }
      return text
    }.joined()
    let notes = blocks.compactMap { block -> String? in
      guard case let .endnotes(text) = block else { return nil }
      return text
    }.joined()
    return Texts(
      reasoning: reasoning.isEmpty ? nil : reasoning,
      response: response.isEmpty ? nil : response,
      notes: notes.isEmpty ? nil : notes,
    )
  }

  /// Deprecated tool-call projection derived from `blocks`.
  @available(*, deprecated, message: "Use blocks as the canonical response content model.")
  public var toolCalls: [ToolCall] {
    blocks.compactMap { block in
      guard case let .toolCall(toolCall) = block else { return nil }
      return toolCall
    }
  }

  /// Deprecated opaque-block projection derived from `blocks`.
  @available(*, deprecated, message: "Use blocks as the canonical response content model.")
  public var opaqueBlocks: [OpaqueBlock]? {
    let opaqueBlocks = blocks.compactMap(\.opaqueBlock)
    return opaqueBlocks.isEmpty ? nil : opaqueBlocks
  }

  private static func block(from opaqueBlock: OpaqueBlock) -> Message.Block {
    switch (opaqueBlock.provider, opaqueBlock.type) {
      case ("anthropic", "thinking"):
        .thinking(text: opaqueBlock.content ?? "", signature: opaqueBlock.signature)
      case ("anthropic", "redacted_thinking"):
        .redactedThinking(data: opaqueBlock.data ?? "")
      default:
        .providerOpaque(opaqueBlock)
    }
  }
}

private let generationResponseLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "GenerationResponse")
