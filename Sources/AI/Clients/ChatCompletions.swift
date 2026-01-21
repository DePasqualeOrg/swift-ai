// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log

public extension ChatCompletionsClient {
  /// Configuration options for Chat Completions API requests.
  struct Configuration: Sendable {
    /// Additional parameters to include in the API request body.
    /// Use this to pass provider-specific options not covered by the standard interface.
    public var extraParameters: [String: any Sendable]?

    /// A configuration with no extra parameters.
    public static let disabled = Configuration()

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
/// print(response.texts.response ?? "")
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
    endpoint: URL
  ) async throws -> AsyncThrowingStream<GenerationResponse, Error> {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    var processedMessages: [[String: any Sendable]] = []
    if let systemPrompt, !systemPrompt.isEmpty {
      processedMessages.append([
        "role": "system",
        "content": systemPrompt,
      ])
    }
    for message in messages {
      if let toolResults = message.toolResults, !toolResults.isEmpty {
        // Handle function results (tool results)
        // ChatCompletions only supports text in tool results
        // TODO: Determine optimal handling for multi-content tool results.
        // Current approach: concatenate text, use fallbackDescription for non-text.
        // Alternatives to consider:
        // - Base64 encode binary data inline (preserves data but bloated)
        // - Upload to external storage and return URL
        // - Structured JSON representation of all content
        for toolResult in toolResults {
          let resultContent: String
          var texts: [String] = []
          for content in toolResult.content {
            switch content {
              case let .text(text):
                texts.append(text)
              case .image, .audio, .file:
                openAILogger.warning("Tool '\(toolResult.name)' returned \(content.type.rawValue), which is not supported by ChatCompletions. Using fallback text.")
                texts.append(content.fallbackDescription)
            }
          }
          resultContent = texts.joined(separator: "\n")
          processedMessages.append([
            "role": "function",
            "name": toolResult.name,
            "content": resultContent,
          ])
        }
        // Add the user message text if it exists
        if let content = message.content, !content.isEmpty {
          processedMessages.append([
            "role": message.role.rawValue,
            "content": content,
          ])
        }
      } else if !message.attachments.isEmpty {
        var messageContent: [[String: any Sendable]] = []
        // Add text content if present
        if let content = message.content, !content.isEmpty {
          messageContent.append([
            "type": "text",
            "text": content,
          ])
        }
        // Process attachments
        for attachment in message.attachments {
          switch attachment.kind {
            case let .image(data, mimeType):
              do {
                // Resize image if necessary before encoding
                let processedImageData = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
                messageContent.append([
                  "type": "image_url",
                  "image_url": [
                    "url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: mimeType),
                  ],
                ])
              } catch {
                openAILogger.error("Failed to process image: \(error.localizedDescription)")
                throw error
              }
            case .video:
              // Not supported
              break
            case .audio:
              // Not supported
              break
            case let .document(data, mimeType):
              // Some OpenAI models support PDF files.
              var fileDict: [String: any Sendable] = [
                "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
              ]
              if let filename = attachment.filename {
                fileDict["filename"] = filename
              }
              messageContent.append([
                "type": "file",
                "file": fileDict,
              ])
          }
        }
        processedMessages.append([
          "role": message.role.rawValue,
          "content": messageContent,
        ])
      } else {
        // Handle text-only messages
        if let content = message.content {
          processedMessages.append([
            "role": message.role.rawValue,
            "content": content,
          ])
        }
      }
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
          Self.convertSchemaForStrictMode(tool.rawInputSchema)
        } else {
          Self.convertValueToSendable(tool.rawInputSchema)
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
            var toolCalls: [GenerationResponse.ToolCall] = []
            var functionCallArguments: [Int: String] = [:] // Accumulate arguments by index
            var metadata = GenerationResponse.Metadata()
            var lastFinishReason: String?
            for try await jsonString in SSEParser.dataPayloads(from: result) {
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
                        toolCalls.append(GenerationResponse.ToolCall(
                          name: name,
                          id: id,
                          parameters: [:]
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
                          toolCalls[index] = GenerationResponse.ToolCall(
                            name: toolCalls[index].name,
                            id: toolCalls[index].id,
                            parameters: parsedArgs
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
                    continuation.yield(GenerationResponse(texts: .init(
                      reasoning: reasoningText,
                      response: responseText,
                      notes: notesText
                    ), toolCalls: toolCalls, metadata: currentMetadata))
                  } else {
                    continue
                  }
                } else {
                  // Handle empty choices/delta (final chunk with just usage data)
                  if lastFinishReason != nil {
                    var currentMetadata = metadata
                    currentMetadata.finishReason = parseFinishReason(lastFinishReason)
                    continuation.yield(GenerationResponse(texts: .init(
                      reasoning: nil,
                      response: nil,
                      notes: nil
                    ), toolCalls: toolCalls, metadata: currentMetadata))
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
              var toolCalls: [GenerationResponse.ToolCall] = []

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
                    toolCalls.append(GenerationResponse.ToolCall(
                      name: name,
                      id: id,
                      parameters: parameters
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
                reasoningTokens: completionResponse.usage?.completionTokensDetails?.reasoningTokens
              )
              // Perplexity citations
              let notesText = formatCitations(completionResponse.citations)
              continuation.yield(.init(texts: .init(
                reasoning: reasoningText,
                response: responseText,
                notes: notesText
              ), toolCalls: toolCalls, metadata: metadata))
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

  private func handleHTTPError(_ statusCode: Int, message: String?) throws -> Never {
    let errorMessage = message ?? "HTTP error \(statusCode)"

    switch statusCode {
      case 401:
        throw AIError.authentication(message: "Ensure the correct API key is being used.")
      case 403:
        throw AIError.authentication(message: "You may be accessing the API from an unsupported country, region, or territory.")
      case 429:
        throw AIError.rateLimit(retryAfter: nil)
      case 500 ... 599:
        throw AIError.serverError(statusCode: statusCode, message: errorMessage, context: nil)
      default:
        throw AIError.invalidRequest(message: errorMessage)
    }
  }

  private func handleErrorResponse(_ httpResponse: HTTPURLResponse, data: Data) throws {
    // Log raw error response for debugging
    //    if let rawResponse = String(data: data, encoding: .utf8) {
    //      openAILogger.error("Raw API error response: \(rawResponse)")
    //    }
    // Try to parse the error response
    if let errorJson = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable] {
      openAILogger.warning("Error: \(errorJson)")
      // Try OpenAI nested format
      if let error = errorJson["error"] as? [String: any Sendable], let message = error["message"] as? String {
        try handleHTTPError(httpResponse.statusCode, message: message)
      }
      // Try Fireworks format
      if let message = errorJson["error"] as? String {
        try handleHTTPError(httpResponse.statusCode, message: message)
      }
      // Try Mistral format
      if let message = errorJson["message"] as? String {
        try handleHTTPError(httpResponse.statusCode, message: message)
      }
    }
    // Fall back to default error handling
    try handleHTTPError(httpResponse.statusCode, message: nil)
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
    configuration: Configuration = .init()
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
      update: { _ in }
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
    configuration: Configuration = .init()
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
            }
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
    configuration: Configuration = .init()
  ) async throws -> GenerationResponse {
    try await generateText(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration
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
    configuration: Configuration = .init()
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    streamText(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration
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
    update: @Sendable @escaping (GenerationResponse) -> Void
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
      var toolCalls: [GenerationResponse.ToolCall] = []
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
          endpoint: endpoint
        )
        for try await chunk in stream {
          try Task.checkCancellation()
          if let reasoningTextChunk = chunk.texts.reasoning {
            fullReasoningText += reasoningTextChunk
          }
          if let responseTextChunk = chunk.texts.response {
            fullResponseText += responseTextChunk
          }
          notesText = chunk.texts.notes
          // Add any function calls from the response
          if !chunk.toolCalls.isEmpty {
            toolCalls = chunk.toolCalls
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
            update(.init(texts: .init(
              reasoning: fullReasoningTextCopy.isEmpty ? nil : fullReasoningTextCopy,
              response: fullResponseTextCopy.isEmpty ? nil : fullResponseTextCopy,
              notes: notesTextCopy
            ), toolCalls: toolCallsCopy, metadata: metadataCopy))
          }
        }
        return .init(texts: .init(
          reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
          response: fullResponseText.isEmpty ? nil : fullResponseText,
          notes: notesText
        ), toolCalls: toolCalls, metadata: finalMetadata)
      } catch {
        if error is CancellationError {
          return .init(texts: .init(
            reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
            response: fullResponseText.isEmpty ? nil : fullResponseText,
            notes: notesText
          ), toolCalls: toolCalls, metadata: finalMetadata)
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

  /// Converts a Value dictionary to a Sendable dictionary for JSON serialization.
  static func convertValueToSendable(_ dict: [String: Value]) -> [String: any Sendable] {
    var result: [String: any Sendable] = [:]
    for (key, value) in dict {
      result[key] = convertValueToSendableValue(value)
    }
    return result
  }

  private static func convertValueToSendableValue(_ value: Value) -> any Sendable {
    switch value {
      case let .string(s): s
      case let .int(i): i
      case let .double(d): d
      case let .bool(b): b
      case .null: NSNull()
      case let .array(arr): arr.map { convertValueToSendableValue($0) }
      case let .object(obj): convertValueToSendable(obj)
    }
  }

  /// Converts a raw JSON schema for OpenAI strict mode compliance.
  /// Ensures "additionalProperties": false is set on all object types.
  /// In strict mode, ALL properties must be in the "required" array.
  static func convertSchemaForStrictMode(_ schema: [String: Value]) -> [String: any Sendable] {
    var result: [String: any Sendable] = [:]

    // Check if this schema is an object type
    let isObjectType = if case let .string(typeStr) = schema["type"] {
      typeStr == "object"
    } else {
      false
    }

    // Collect property names to ensure all are in required array
    var propertyNames: [String] = []

    for (key, value) in schema {
      if key == "properties" {
        // Recursively convert property schemas
        if case let .object(props) = value {
          var convertedProps: [String: any Sendable] = [:]
          for (propName, propSchema) in props {
            propertyNames.append(propName)
            if case let .object(propSchemaDict) = propSchema {
              convertedProps[propName] = convertSchemaForStrictMode(propSchemaDict)
            } else {
              convertedProps[propName] = convertValueToSendableValue(propSchema)
            }
          }
          result[key] = convertedProps
        } else {
          result[key] = convertValueToSendableValue(value)
        }
      } else if key == "items" {
        // Recursively convert array item schema
        if case let .object(itemSchema) = value {
          result[key] = convertSchemaForStrictMode(itemSchema)
        } else {
          result[key] = convertValueToSendableValue(value)
        }
      } else if key == "additionalProperties" {
        // Always set to false for strict mode
        result[key] = false
      } else if key == "required" {
        // Don't copy required here - we'll set it later to include all properties
        continue
      } else {
        result[key] = convertValueToSendableValue(value)
      }
    }

    // Add additionalProperties: false if this is an object type and it's not already set
    if isObjectType {
      if result["additionalProperties"] == nil {
        result["additionalProperties"] = false
      }
      // In strict mode, ALL properties must be in required
      if !propertyNames.isEmpty {
        result["required"] = propertyNames.sorted()
      }
    }

    return result
  }
}

private let openAILogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "ChatCompletionsClient")
