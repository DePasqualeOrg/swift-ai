// Copyright © Anthony DePasquale

import Foundation
import os.log

/// A response from an LLM generation request.
///
/// Contains the generated text, any tool calls made by the model, and metadata
/// about the response such as token usage and finish reason.
public struct GenerationResponse: Sendable, Hashable {
  /// Text content returned by the model.
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
      reasoningTokens: Int? = nil
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

  /// The text content of the response.
  public var texts: Texts

  /// Metadata about the response (token usage, finish reason, etc.).
  public var metadata: Metadata?

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

  /// Tool calls made by the model in this response.
  public var toolCalls: [ToolCall]

  /// Provider-specific opaque blocks that must be round-tripped (e.g., Anthropic thinking blocks with signatures).
  public var opaqueBlocks: [OpaqueBlock]?

  /// The assistant message representing this response, suitable for adding to conversation history.
  public var message: Message {
    Message(
      role: Message.Role.assistant,
      content: texts.response,
      toolCalls: toolCalls.isEmpty ? nil : toolCalls,
      opaqueBlocks: opaqueBlocks
    )
  }

  /// Creates a new generation response.
  ///
  /// - Parameters:
  ///   - texts: The text content of the response.
  ///   - toolCalls: Tool calls made by the model.
  ///   - metadata: Metadata about the response.
  ///   - opaqueBlocks: Provider-specific opaque blocks for round-tripping.
  public init(texts: Texts = Texts(), toolCalls: [ToolCall] = [], metadata: Metadata? = nil, opaqueBlocks: [OpaqueBlock]? = nil) {
    self.texts = texts
    self.toolCalls = toolCalls
    self.metadata = metadata
    self.opaqueBlocks = opaqueBlocks
  }
}

private let generationResponseLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "GenerationResponse")
