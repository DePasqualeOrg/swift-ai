// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log
import SSE

public extension ChatCompletionsClient {
  /// Configuration options for Chat Completions API requests.
  struct Configuration: Sendable {
    /// Additional parameters to include in the API request body.
    /// Use this to pass provider-specific options not covered by the standard interface.
    public var extraParameters: [String: any Sendable]?

    /// Creates a new configuration with optional extra parameters.
    ///
    /// - Parameter extraParameters: Additional parameters for the API request.
    public init(extraParameters: [String: any Sendable]? = nil) {
      self.extraParameters = extraParameters
    }
  }
}

/// A client for OpenAI-compatible Chat Completions APIs.
///
/// Works with OpenAI, xAI (Grok), and other providers that implement the
/// Chat Completions API format. Supports tool use and streaming.
///
/// ## Example
///
/// ```swift
/// let client = ChatCompletionsClient()
/// let response = try await client.generateText(
///   modelId: "gpt-4o",
///   prompt: "Hello!",
///   apiKey: "your-api-key"
/// )
/// print(response.blocks)
/// ```
@Observable
public final class ChatCompletionsClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text]

  /// Predefined API endpoints for the Chat Completions API.
  public enum Endpoint {
    /// OpenAI's Chat Completions API endpoint.
    case openAI
    /// xAI's Chat Completions API endpoint.
    case xAI

    /// The URL for this endpoint.
    public var url: URL {
      switch self {
        case .openAI: URL(string: "https://api.openai.com/v1/chat/completions")!
        case .xAI: URL(string: "https://api.x.ai/v1/chat/completions")!
      }
    }
  }

  private let enableStrictModeForTools = true

  /// The API endpoint URL used by this client.
  public let endpoint: URL

  @MainActor public private(set) var isGenerating: Bool = false
  @MainActor private var currentTask: Task<GenerationResponse, Error>?

  private let session: URLSession

  /// Creates a new Chat Completions client with a predefined endpoint.
  ///
  /// - Parameters:
  ///   - endpoint: The API endpoint to use (OpenAI or xAI).
  ///   - session: URLSession to use for requests.
  public init(endpoint: Endpoint = .openAI, session: URLSession = .shared) {
    self.endpoint = endpoint.url
    self.session = session
  }

  /// Creates a new Chat Completions client with a custom endpoint URL.
  ///
  /// - Parameters:
  ///   - endpoint: Custom endpoint URL for the Chat Completions API.
  ///   - session: URLSession to use for requests.
  public init(endpoint: URL, session: URLSession = .shared) {
    self.endpoint = endpoint
    self.session = session
  }

  private static func assistantBlocks(
    reasoningText: String? = nil,
    responseText: String? = nil,
    notesText: String? = nil,
    toolCalls: [AI.ToolCall] = [],
  ) -> [Message.Block] {
    Message.assistantBlocks(reasoningText: reasoningText, responseText: responseText, notesText: notesText, toolCalls: toolCalls)
  }

  private static func assistantSnapshot(from response: GenerationResponse) -> (reasoning: String?, response: String?, notes: String?, toolCalls: [AI.ToolCall]) {
    var reasoningParts: [String] = []
    var responseParts: [String] = []
    var notesParts: [String] = []
    var toolCalls: [AI.ToolCall] = []

    for block in response.blocks {
      switch block {
        case let .thinking(text, _):
          reasoningParts.append(text)
        case let .text(text):
          responseParts.append(text)
        case let .endnotes(text):
          notesParts.append(text)
        case let .toolCall(toolCall):
          toolCalls.append(toolCall)
        default:
          break
      }
    }

    return (
      reasoningParts.isEmpty ? nil : reasoningParts.joined(),
      responseParts.isEmpty ? nil : responseParts.joined(),
      notesParts.isEmpty ? nil : notesParts.joined(),
      toolCalls,
    )
  }

  private static func serializedToolCall(_ toolCall: AI.ToolCall) throws -> [String: any Sendable] {
    let argumentsData = try JSONSerialization.data(withJSONObject: Value.toSendable(toolCall.parameters), options: [])
    guard let argumentsString = String(data: argumentsData, encoding: .utf8) else {
      throw AIError.invalidRequest(message: "Failed to serialize function call arguments to JSON string")
    }
    return [
      "id": toolCall.id,
      "type": "function",
      "function": [
        "name": toolCall.name,
        "arguments": argumentsString,
      ],
    ]
  }

  private static func requestMessages(for message: Message) async throws -> [[String: any Sendable]] {
    if message.role == .tool {
      return message.blocks.compactMap { block -> [String: any Sendable]? in
        guard case let .toolResult(toolResult) = block else { return nil }

        let resultContent = toolResult.content.map { content -> String in
          switch content {
            case let .text(text):
              return text
            case .image, .audio, .file:
              openAILogger.warning("Tool '\(toolResult.name)' returned \(content.type.rawValue), which is not supported by ChatCompletions. Using fallback text.")
              return content.fallbackDescription
          }
        }.joined(separator: "\n")

        return [
          "role": "function",
          "name": toolResult.name,
          "content": resultContent,
        ]
      }
    }

    var textParts: [String] = []
    var toolCalls: [[String: any Sendable]] = []
    var multimodalContent: [[String: any Sendable]] = []
    var hasNonTextContent = false

    for block in message.blocks {
      switch block {
        case let .text(text) where !text.isEmpty:
          textParts.append(text)
          multimodalContent.append([
            "type": "text",
            "text": text,
          ])
        case let .attachment(attachment):
          hasNonTextContent = true
          switch attachment.kind {
            case let .image(data, mimeType):
              let processedImageData = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
              multimodalContent.append([
                "type": "image_url",
                "image_url": [
                  "url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: mimeType),
                ],
              ])
            case .video, .audio:
              break
            case let .document(data, mimeType):
              var fileDict: [String: any Sendable] = [
                "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
              ]
              if let filename = attachment.filename {
                fileDict["filename"] = filename
              }
              multimodalContent.append([
                "type": "file",
                "file": fileDict,
              ])
          }
        case let .toolCall(toolCall):
          try toolCalls.append(serializedToolCall(toolCall))
        default:
          break
      }
    }

    var requestMessage: [String: any Sendable] = [
      "role": message.role.rawValue,
    ]

    if !toolCalls.isEmpty {
      requestMessage["tool_calls"] = toolCalls
      requestMessage["content"] = textParts.joined()
    } else if hasNonTextContent {
      requestMessage["content"] = multimodalContent
    } else if !textParts.isEmpty {
      requestMessage["content"] = textParts.joined()
    } else {
      return []
    }

    return [requestMessage]
  }

  private func streamResponse(
    messages: [Message],
    systemPrompt: String?,
    modelId: String,
    apiKey: String?,
    maxTokens: Int?,
    temperature: Float?,
    stream: Bool,
    tools: [Tool] = [],
    extraParameters: [String: any Sendable]?,
    endpoint: URL,
  ) async throws -> AsyncThrowingStream<GenerationResponse, Error> {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    let patchedMessages = Message.patchingOrphanedToolCalls(messages)
    var processedMessages: [[String: any Sendable]] = []
    if let systemPrompt, !systemPrompt.isEmpty {
      processedMessages.append([
        "role": "system",
        "content": systemPrompt,
      ])
    }
    for message in patchedMessages {
      try await processedMessages.append(contentsOf: Self.requestMessages(for: message))
    }
    var body: [String: any Sendable] = [
      "model": modelId,
      "messages": processedMessages,
      "stream": stream,
    ]
    if let maxTokens {
      body["max_tokens"] = maxTokens
    }
    if let temperature {
      body["temperature"] = temperature
    }
    // Add tools if provided - rawInputSchema is always populated
    if !tools.isEmpty {
      var toolsArray: [[String: any Sendable]] = []
      for tool in tools {
        let parameters: [String: any Sendable]
          // Transform schema for strict mode compliance if enabled
          = if enableStrictModeForTools
        {
          Value.schemaForStrictMode(tool.rawInputSchema)
        } else {
          Value.toSendable(tool.rawInputSchema)
        }
        toolsArray.append([
          "type": "function",
          "function": [
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
            "strict": enableStrictModeForTools,
          ] as [String: any Sendable],
        ])
      }
      body["tools"] = toolsArray
      // Set tool_choice to auto by default when tools are present
      body["tool_choice"] = "auto"
    }
    // Extra parameters
    if let extraParameters {
      body.merge(extraParameters) { _, new in new }
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let finalRequest = request
    return AsyncThrowingStream { continuation in
      Task { @Sendable in
        let request = finalRequest
        do {
          if stream {
            // Handle streaming response
            let (result, response) = try await session.bytes(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
              throw AIError.network(underlying: URLError(.badServerResponse))
            }
            if !(200 ... 299).contains(httpResponse.statusCode) {
              var errorData = Data()
              for try await byte in result {
                try Task.checkCancellation()
                errorData.append(byte)
              }
              try handleErrorResponse(httpResponse, data: errorData)
            }
            var toolCalls: [AI.ToolCall] = []
            var functionCallArguments: [Int: String] = [:] // Accumulate arguments by index
            var metadata = GenerationResponse.Metadata()
            var lastFinishReason: String?
            for try await event in result.events {
              try Task.checkCancellation()
              let jsonString = event.data

              if jsonString == "[DONE]" {
                break
              }

              guard let jsonData = jsonString.data(using: .utf8) else {
                throw AIError.parsing(message: "Failed to convert streamed response to data")
              }
              do {
                let chunk = try JSONDecoder().decode(StreamChunk.self, from: jsonData)
                // Check if this is an error response
                if let errorMessage = chunk.error?.message {
                  throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
                }
                // Update metadata from chunk
                if let id = chunk.id { metadata.responseId = id }
                if let model = chunk.model { metadata.model = model }
                if let created = chunk.created { metadata.createdAt = Date(timeIntervalSince1970: TimeInterval(created)) }
                if let usage = chunk.usage {
                  metadata.inputTokens = usage.promptTokens
                  metadata.outputTokens = usage.completionTokens
                  metadata.totalTokens = usage.totalTokens
                  metadata.cacheReadInputTokens = usage.promptTokensDetails?.cachedTokens
                  metadata.reasoningTokens = usage.completionTokensDetails?.reasoningTokens
                }
                if let finishReason = chunk.choices?.first?.finishReason {
                  lastFinishReason = finishReason
                }
                // Check if choices array exists and is not empty
                if let choices = chunk.choices, !choices.isEmpty, let delta = choices.first?.delta {
                  let reasoningText = delta.reasoningContent
                  let responseText = delta.content

                  // Handle tool calls
                  if let deltaToolCalls = delta.toolCalls {
                    for deltaToolCall in deltaToolCalls {
                      guard let index = deltaToolCall.index else { continue }

                      // Create new tool call if this is the first chunk for this index
                      if let id = deltaToolCall.id,
                         let function = deltaToolCall.function,
                         let name = function.name,
                         index >= toolCalls.count
                      {
                        toolCalls.append(AI.ToolCall(
                          name: name,
                          id: id,
                          parameters: [:],
                        ))
                        functionCallArguments[index] = function.arguments ?? ""
                      } else if let function = deltaToolCall.function, let arguments = function.arguments {
                        // Accumulate arguments for existing tool call
                        functionCallArguments[index, default: ""] += arguments
                      }

                      // Try to parse accumulated arguments
                      if index < toolCalls.count, let accumulatedArgs = functionCallArguments[index], !accumulatedArgs.isEmpty {
                        if let argsData = accumulatedArgs.data(using: .utf8),
                           let parsedArgs = try? JSONDecoder().decode([String: Value].self, from: argsData)
                        {
                          toolCalls[index] = AI.ToolCall(
                            name: toolCalls[index].name,
                            id: toolCalls[index].id,
                            parameters: parsedArgs,
                          )
                        }
                      }
                    }
                  }
                  // Perplexity citations
                  let notesText = formatCitations(chunk.citations)
                  // Yield if we have content, function calls, or a finish reason (final chunk with metadata)
                  if reasoningText != nil || responseText != nil || notesText != nil || !toolCalls.isEmpty || lastFinishReason != nil {
                    var currentMetadata = metadata
                    currentMetadata.finishReason = parseFinishReason(lastFinishReason)
                    continuation.yield(GenerationResponse(
                      blocks: Self.assistantBlocks(
                        reasoningText: reasoningText,
                        responseText: responseText,
                        notesText: notesText,
                        toolCalls: toolCalls,
                      ),
                      metadata: currentMetadata,
                    ))
                  } else {
                    continue
                  }
                } else {
                  // Handle empty choices/delta (final chunk with just usage data)
                  if lastFinishReason != nil {
                    var currentMetadata = metadata
                    currentMetadata.finishReason = parseFinishReason(lastFinishReason)
                    continuation.yield(GenerationResponse(
                      blocks: Self.assistantBlocks(toolCalls: toolCalls),
                      metadata: currentMetadata,
                    ))
                  }
                  continue
                }
              } catch let error as AIError {
                // Re-throw LLM errors
                throw error
              } catch {
                openAILogger.error("Failed to parse streamed JSON: \(jsonString)")
                throw AIError.parsing(message: "Failed to parse streamed JSON. The server returned an incomplete or invalid response.")
              }
            }
          } else {
            // Handle non-streaming response
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
              throw AIError.network(underlying: URLError(.badServerResponse))
            }
            if !(200 ... 299).contains(httpResponse.statusCode) {
              try handleErrorResponse(httpResponse, data: data)
            }
            do {
              let completionResponse = try JSONDecoder().decode(CompletionResponse.self, from: data)
              guard let choices = completionResponse.choices,
                    let firstChoice = choices.first,
                    let message = firstChoice.message
              else {
                throw AIError.parsing(message: "Failed to parse JSON from non-streamed response")
              }
              let reasoningText = message.reasoningContent
              let responseText = message.content
              var toolCalls: [AI.ToolCall] = []

              // Get tool calls if available
              if let messageToolCalls = message.toolCalls {
                for messageToolCall in messageToolCalls {
                  if let function = messageToolCall.function,
                     let name = function.name,
                     let arguments = function.arguments,
                     let id = messageToolCall.id
                  {
                    var parameters: [String: Value] = [:]
                    if let argumentsData = arguments.data(using: .utf8),
                       let parsedArgs = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
                    {
                      parameters = parsedArgs
                    } else if !arguments.isEmpty {
                      openAILogger.error("Failed to parse function call arguments for call ID \(id): \(arguments)")
                      parameters = ["_parseError": .string("Failed to parse arguments JSON"), "_rawArguments": .string(arguments)]
                    }
                    toolCalls.append(AI.ToolCall(
                      name: name,
                      id: id,
                      parameters: parameters,
                    ))
                  }
                }
              }
              // Build metadata
              let metadata = GenerationResponse.Metadata(
                responseId: completionResponse.id,
                model: completionResponse.model,
                createdAt: completionResponse.created.map { Date(timeIntervalSince1970: TimeInterval($0)) },
                finishReason: parseFinishReason(firstChoice.finishReason),
                inputTokens: completionResponse.usage?.promptTokens,
                outputTokens: completionResponse.usage?.completionTokens,
                totalTokens: completionResponse.usage?.totalTokens,
                cacheReadInputTokens: completionResponse.usage?.promptTokensDetails?.cachedTokens,
                reasoningTokens: completionResponse.usage?.completionTokensDetails?.reasoningTokens,
              )
              // Perplexity citations
              let notesText = formatCitations(completionResponse.citations)
              continuation.yield(.init(
                blocks: Self.assistantBlocks(
                  reasoningText: reasoningText,
                  responseText: responseText,
                  notesText: notesText,
                  toolCalls: toolCalls,
                ),
                metadata: metadata,
              ))
            } catch {
              openAILogger.error("Failed to parse non-streamed content: \(error.localizedDescription)")
              throw error
            }
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }

  private func formatCitations(_ citations: [String]?) -> String? {
    guard let citations, !citations.isEmpty else { return nil }
    return citations.enumerated()
      .map { index, url in "\(index + 1). \(url)" }
      .joined(separator: "\n")
  }

  private func handleErrorResponse(_ httpResponse: HTTPURLResponse, data: Data) throws {
    try AIError.throwOpenAIHTTPError(httpResponse, data: data, logger: openAILogger)
  }

  /// Generates a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
  ///   - configuration: Additional configuration options.
  /// - Returns: The generation response with text and metadata.
  public func generateText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) async throws -> GenerationResponse {
    try await _generate(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: messages,
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      stream: false,
      configuration: configuration,
      update: { _ in },
    )
  }

  /// Streams a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
  ///   - configuration: Additional configuration options.
  /// - Returns: An async stream of generation responses as they arrive.
  public func streamText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let finalResponse = try await _generate(
            modelId: modelId,
            tools: tools,
            systemPrompt: systemPrompt,
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            apiKey: apiKey,
            stream: true,
            configuration: configuration,
            update: { response in
              continuation.yield(response)
            },
          )
          // Yield the final response with metadata
          continuation.yield(finalResponse)
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { @Sendable _ in
        task.cancel()
      }
    }
  }

  /// Generate a text response using a simple prompt string.
  public func generateText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) async throws -> GenerationResponse {
    try await generateText(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, blocks: [.text(prompt)])],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
    )
  }

  /// Generate a text response with streaming using a simple prompt string.
  public func streamText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    streamText(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, blocks: [.text(prompt)])],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
    )
  }

  private func _generate(
    modelId: String,
    tools: [Tool],
    systemPrompt: String?,
    messages: [Message],
    maxTokens: Int?,
    temperature: Float?,
    apiKey: String?,
    stream: Bool,
    configuration: Configuration,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    await MainActor.run { isGenerating = true }
    let task = Task<GenerationResponse, Error> {
      defer {
        Task { @MainActor in
          isGenerating = false
          currentTask = nil
        }
      }
      var fullReasoningText = ""
      var fullResponseText = ""
      var notesText: String?
      var toolCalls: [AI.ToolCall] = []
      var finalMetadata: GenerationResponse.Metadata?
      do {
        let stream = try await streamResponse(
          messages: messages,
          systemPrompt: systemPrompt,
          modelId: modelId,
          apiKey: apiKey,
          maxTokens: maxTokens,
          temperature: temperature,
          stream: stream,
          tools: tools,
          extraParameters: configuration.extraParameters,
          endpoint: endpoint,
        )
        for try await chunk in stream {
          try Task.checkCancellation()
          let snapshot = Self.assistantSnapshot(from: chunk)
          if let reasoningTextChunk = snapshot.reasoning {
            fullReasoningText += reasoningTextChunk
          }
          if let responseTextChunk = snapshot.response {
            fullResponseText += responseTextChunk
          }
          notesText = snapshot.notes
          if !snapshot.toolCalls.isEmpty {
            toolCalls = snapshot.toolCalls
          }
          // Capture metadata from chunks (metadata accumulates, later values override)
          if let chunkMetadata = chunk.metadata {
            finalMetadata = chunkMetadata
          }
          let fullReasoningTextCopy = fullReasoningText
          let fullResponseTextCopy = fullResponseText
          let notesTextCopy = notesText
          let toolCallsCopy = toolCalls
          let metadataCopy = finalMetadata
          await MainActor.run {
            update(.init(
              blocks: Self.assistantBlocks(
                reasoningText: fullReasoningTextCopy.isEmpty ? nil : fullReasoningTextCopy,
                responseText: fullResponseTextCopy.isEmpty ? nil : fullResponseTextCopy,
                notesText: notesTextCopy,
                toolCalls: toolCallsCopy,
              ),
              metadata: metadataCopy,
            ))
          }
        }
        return .init(
          blocks: Self.assistantBlocks(
            reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
            responseText: fullResponseText.isEmpty ? nil : fullResponseText,
            notesText: notesText,
            toolCalls: toolCalls,
          ),
          metadata: finalMetadata,
        )
      } catch {
        if error is CancellationError {
          return .init(
            blocks: Self.assistantBlocks(
              reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
              responseText: fullResponseText.isEmpty ? nil : fullResponseText,
              notesText: notesText,
              toolCalls: toolCalls,
            ),
            metadata: finalMetadata,
          )
        } else {
          throw error
        }
      }
    }
    await MainActor.run {
      currentTask = task
    }
    return try await task.value
  }

  /// Cancels any ongoing generation task.
  @MainActor
  public func stop() {
    currentTask?.cancel()
  }

  // MARK: - Helpers

  private func parseFinishReason(_ reason: String?) -> GenerationResponse.FinishReason? {
    guard let reason else { return nil }
    switch reason {
      case "stop", "end_turn": return .stop
      case "length", "max_tokens": return .maxTokens
      case "tool_calls", "tool_use", "function_call": return .toolUse
      case "content_filter": return .contentFilter
      default: return .other
    }
  }

  // MARK: - Codable Types for Chat Completions API

  struct StreamChunk: Decodable {
    let id: String?
    let model: String?
    let created: Int?
    let choices: [StreamChoice]?
    let citations: [String]?
    let usage: Usage?
    let error: ErrorObject?
  }

  struct Usage: Decodable {
    let promptTokens: Int?
    let completionTokens: Int?
    let totalTokens: Int?
    let promptTokensDetails: PromptTokensDetails?
    let completionTokensDetails: CompletionTokensDetails?

    enum CodingKeys: String, CodingKey {
      case promptTokens = "prompt_tokens"
      case completionTokens = "completion_tokens"
      case totalTokens = "total_tokens"
      case promptTokensDetails = "prompt_tokens_details"
      case completionTokensDetails = "completion_tokens_details"
    }
  }

  struct PromptTokensDetails: Decodable {
    let cachedTokens: Int?

    enum CodingKeys: String, CodingKey {
      case cachedTokens = "cached_tokens"
    }
  }

  struct CompletionTokensDetails: Decodable {
    let reasoningTokens: Int?

    enum CodingKeys: String, CodingKey {
      case reasoningTokens = "reasoning_tokens"
    }
  }

  struct StreamChoice: Decodable {
    let delta: Delta?
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
      case delta
      case finishReason = "finish_reason"
    }
  }

  struct Delta: Decodable {
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ToolCallDelta]?

    enum CodingKeys: String, CodingKey {
      case content
      case reasoningContent = "reasoning_content"
      case toolCalls = "tool_calls"
    }
  }

  struct ToolCallDelta: Decodable {
    let id: String?
    let index: Int?
    let function: FunctionDelta?
  }

  struct FunctionDelta: Decodable {
    let name: String?
    let arguments: String?
  }

  struct CompletionResponse: Decodable {
    let id: String?
    let model: String?
    let created: Int?
    let choices: [CompletionChoice]?
    let citations: [String]?
    let usage: Usage?
    let error: ErrorObject?
  }

  struct CompletionChoice: Decodable {
    let message: APIResponseMessage?
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
      case message
      case finishReason = "finish_reason"
    }
  }

  struct APIResponseMessage: Decodable {
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ToolCall]?

    enum CodingKeys: String, CodingKey {
      case content
      case reasoningContent = "reasoning_content"
      case toolCalls = "tool_calls"
    }
  }

  struct ToolCall: Decodable {
    let id: String?
    let function: FunctionCall?
  }

  struct FunctionCall: Decodable {
    let name: String?
    let arguments: String?
  }

  struct ErrorObject: Decodable {
    let message: String?
    let code: String?
  }
}

private let openAILogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "ChatCompletionsClient")
