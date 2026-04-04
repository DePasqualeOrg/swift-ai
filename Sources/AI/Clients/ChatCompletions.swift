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

    /// When true, sends `max_tokens` instead of `max_completion_tokens` in the request body.
    /// Many OpenAI-compatible servers (Ollama, vLLM, etc.) only accept the legacy `max_tokens`
    /// field, while OpenAI's own API uses `max_completion_tokens` (required for reasoning models).
    /// Defaults to false, which sends `max_completion_tokens`.
    public var useLegacyMaxTokensField: Bool

    /// When true, tool schemas are rewritten for OpenAI strict mode compliance and sent
    /// with `strict: true`. This ensures the model's output exactly matches the schema,
    /// but requires all properties to be in `required` and optional properties to be nullable.
    /// Defaults to true. Disable for tools with optional non-nullable parameters or for
    /// compatible endpoints that don't support strict mode.
    public var enableStrictModeForTools: Bool

    /// Creates a new configuration with optional extra parameters.
    ///
    /// - Parameters:
    ///   - extraParameters: Additional parameters for the API request.
    ///   - useLegacyMaxTokensField: Send `max_tokens` instead of `max_completion_tokens` for
    ///     compatibility with endpoints that don't support the newer field.
    ///   - enableStrictModeForTools: Rewrite tool schemas for strict mode compliance.
    public init(
      extraParameters: [String: any Sendable]? = nil,
      useLegacyMaxTokensField: Bool = false,
      enableStrictModeForTools: Bool = true,
    ) {
      self.extraParameters = extraParameters
      self.useLegacyMaxTokensField = useLegacyMaxTokensField
      self.enableStrictModeForTools = enableStrictModeForTools
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
/// print(response.content)
/// ```
@Observable
public final class ChatCompletionsClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text]
  private static let refusalOpaqueProvider = "openai-chat-completions"
  private static let refusalOpaqueType = "refusal"

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

  private static func assistantContent(
    reasoningText: String? = nil,
    responseText: String? = nil,
    refusalText: String? = nil,
    notesText: String? = nil,
    toolCalls: [AI.ToolCall] = [],
  ) -> [Message.Content] {
    var content: [Message.Content] = []
    if let reasoningText, !reasoningText.isEmpty {
      content.append(.thinking(text: reasoningText, signature: nil))
    }
    if let responseText, !responseText.isEmpty {
      content.append(.text(responseText))
    }
    if let refusalText, !refusalText.isEmpty {
      content.append(.providerOpaque(OpaqueBlock(
        provider: refusalOpaqueProvider,
        type: refusalOpaqueType,
        content: refusalText,
        isResponseContent: true,
      )))
    }
    if let notesText, !notesText.isEmpty {
      content.append(.endnotes(notesText))
    }
    content.append(contentsOf: toolCalls.map(Message.Content.toolCall))
    return content
  }

  private static func refusalText(from block: Message.Content) -> String? {
    guard case let .providerOpaque(opaque) = block,
          opaque.provider == refusalOpaqueProvider,
          opaque.type == refusalOpaqueType,
          let refusalText = opaque.content,
          !refusalText.isEmpty
    else {
      return nil
    }
    return refusalText
  }

  private static func assistantSnapshot(from response: GenerationResponse) -> (reasoning: String?, response: String?, refusal: String?, notes: String?, toolCalls: [AI.ToolCall]) {
    var reasoningParts: [String] = []
    var responseParts: [String] = []
    var refusalParts: [String] = []
    var notesParts: [String] = []
    var toolCalls: [AI.ToolCall] = []

    for block in response.content {
      if let refusalText = Self.refusalText(from: block) {
        refusalParts.append(refusalText)
        continue
      }

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
      refusalParts.isEmpty ? nil : refusalParts.joined(),
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
      return message.content.compactMap { block -> [String: any Sendable]? in
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
          "role": "tool",
          "tool_call_id": toolResult.id,
          "content": resultContent,
        ]
      }
    }

    var textParts: [String] = []
    var refusalParts: [String] = []
    var toolCalls: [[String: any Sendable]] = []
    var multimodalContent: [[String: any Sendable]] = []
    var hasNonTextContent = false

    for block in message.content {
      if let refusalText = Self.refusalText(from: block) {
        refusalParts.append(refusalText)
        continue
      }

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
              let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
              multimodalContent.append([
                "type": "image_url",
                "image_url": [
                  "url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: processedMimeType),
                ],
              ])
            case let .audio(data, mimeType):
              // Map MIME type to the format string expected by the API ("wav" or "mp3")
              let format: String? = switch mimeType {
                case "audio/wav", "audio/x-wav", "audio/wave": "wav"
                case "audio/mpeg", "audio/mp3": "mp3"
                default: nil
              }
              if let format {
                multimodalContent.append([
                  "type": "input_audio",
                  "input_audio": [
                    "data": data.base64EncodedString(),
                    "format": format,
                  ] as [String: any Sendable],
                ])
              } else {
                openAILogger.warning("Audio format '\(mimeType)' is not supported by ChatCompletions (only wav and mp3). Attachment will be omitted.")
              }
            case .video:
              openAILogger.warning("Video attachments are not supported by ChatCompletions and will be omitted.")
            case let .document(data, mimeType):
              // The API expects a data URL (e.g. "data:application/pdf;base64,...") for file_data,
              // despite the OpenAI TS SDK describing it as "base64-encoded data".
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
    let refusalText = refusalParts.joined()
    var hasSerializedContent = false

    if !toolCalls.isEmpty {
      requestMessage["tool_calls"] = toolCalls
      requestMessage["content"] = textParts.joined()
      hasSerializedContent = true
    } else if hasNonTextContent {
      requestMessage["content"] = multimodalContent
      hasSerializedContent = true
    } else if !textParts.isEmpty {
      requestMessage["content"] = textParts.joined()
      hasSerializedContent = true
    }

    if !refusalText.isEmpty {
      requestMessage["refusal"] = refusalText
    }

    if !hasSerializedContent, refusalText.isEmpty {
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
    useLegacyMaxTokensField: Bool = false,
    enableStrictModeForTools: Bool = true,
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
      body[useLegacyMaxTokensField ? "max_tokens" : "max_completion_tokens"] = maxTokens
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
          try Value.schemaForStrictMode(tool.rawInputSchema)
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
    let (resultStream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
    let task = Task { @Sendable in
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
          var toolCallsById: [Int: AI.ToolCall] = [:] // Accumulate tool calls by index
          var functionCallArguments: [Int: String] = [:] // Accumulate arguments by index
          var metadata = GenerationResponse.Metadata()
          var lastFinishReason: String?
          var hasRefusal = false
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
                let refusalText = delta.refusal
                if refusalText != nil {
                  hasRefusal = true
                }

                // Handle tool calls
                if let deltaToolCalls = delta.toolCalls {
                  for deltaToolCall in deltaToolCalls {
                    guard let index = deltaToolCall.index else { continue }

                    // Ensure entry exists for this index
                    var toolCall = toolCallsById[index] ?? AI.ToolCall(name: "", id: "", parameters: [:])

                    // Merge fields independently as they arrive
                    if let id = deltaToolCall.id {
                      toolCall = AI.ToolCall(name: toolCall.name, id: id, parameters: toolCall.parameters)
                    }
                    if let function = deltaToolCall.function {
                      if let name = function.name {
                        toolCall = AI.ToolCall(name: name, id: toolCall.id, parameters: toolCall.parameters)
                      }
                      if let arguments = function.arguments {
                        functionCallArguments[index, default: ""] += arguments
                      }
                    }

                    // Try to parse accumulated arguments
                    if let accumulatedArgs = functionCallArguments[index], !accumulatedArgs.isEmpty,
                       let argsData = accumulatedArgs.data(using: .utf8),
                       let parsedArgs = try? JSONDecoder().decode([String: Value].self, from: argsData)
                    {
                      toolCall = AI.ToolCall(name: toolCall.name, id: toolCall.id, parameters: parsedArgs)
                    }

                    toolCallsById[index] = toolCall
                  }
                }
                // Perplexity citations
                let notesText = formatCitations(chunk.citations)
                let toolCalls = toolCallsById.keys.sorted().compactMap { toolCallsById[$0] }
                // Yield if we have content, function calls, or a finish reason (final chunk with metadata)
                if reasoningText != nil || responseText != nil || refusalText != nil || notesText != nil || !toolCalls.isEmpty || lastFinishReason != nil {
                  var currentMetadata = metadata
                  currentMetadata.finishReason = hasRefusal ? .refusal : parseFinishReason(lastFinishReason)
                  continuation.yield(GenerationResponse(
                    content: Self.assistantContent(
                      reasoningText: reasoningText,
                      responseText: responseText,
                      refusalText: refusalText,
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
                let toolCalls = toolCallsById.keys.sorted().compactMap { toolCallsById[$0] }
                if lastFinishReason != nil {
                  var currentMetadata = metadata
                  currentMetadata.finishReason = hasRefusal ? .refusal : parseFinishReason(lastFinishReason)
                  continuation.yield(GenerationResponse(
                    content: Self.assistantContent(toolCalls: toolCalls),
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
            let refusalText = message.refusal
            let hasRefusal = refusalText != nil
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
              finishReason: hasRefusal ? .refusal : parseFinishReason(firstChoice.finishReason),
              inputTokens: completionResponse.usage?.promptTokens,
              outputTokens: completionResponse.usage?.completionTokens,
              totalTokens: completionResponse.usage?.totalTokens,
              cacheReadInputTokens: completionResponse.usage?.promptTokensDetails?.cachedTokens,
              reasoningTokens: completionResponse.usage?.completionTokensDetails?.reasoningTokens,
            )
            // Citations: Perplexity uses top-level `citations`, OpenAI uses `message.annotations`
            let notesText = formatCitations(completionResponse.citations)
              ?? Self.formatAnnotations(message.annotations)
            continuation.yield(.init(
              content: Self.assistantContent(
                reasoningText: reasoningText,
                responseText: responseText,
                refusalText: refusalText,
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
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return resultStream
  }

  private func formatCitations(_ citations: [String]?) -> String? {
    guard let citations, !citations.isEmpty else { return nil }
    return citations.enumerated()
      .map { index, url in "\(index + 1). \(url)" }
      .joined(separator: "\n")
  }

  private static func formatAnnotations(_ annotations: [Annotation]?) -> String? {
    guard let annotations, !annotations.isEmpty else { return nil }
    let lines = annotations.compactMap { annotation -> String? in
      guard annotation.type == "url_citation",
            let citation = annotation.urlCitation,
            let url = citation.url
      else { return nil }
      let label = citation.title ?? url
      return "- [\(label)](\(url))"
    }
    guard !lines.isEmpty else { return nil }
    return lines.joined(separator: "\n") + "\n"
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
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) async throws -> GenerationResponse {
    try await _generate(
      modelId: modelId,
      tools: Array(tools),
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
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    let tools = Array(tools)
    let (stream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
    let task = Task {
      do {
        let didYield = OSAllocatedUnfairLock(initialState: false)
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
            didYield.withLock { $0 = true }
            continuation.yield(response)
          },
        )
        if !didYield.withLock({ $0 }) {
          continuation.yield(finalResponse)
        }
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return stream
  }

  /// Generate a text response using a simple prompt string.
  public func generateText(
    modelId: String,
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) async throws -> GenerationResponse {
    try await generateText(
      modelId: modelId,
      tools: Array(tools),
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
    )
  }

  /// Generate a text response with streaming using a simple prompt string.
  public func streamText(
    modelId: String,
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    streamText(
      modelId: modelId,
      tools: Array(tools),
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
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
      var fullReasoningText = ""
      var fullResponseText = ""
      var fullRefusalText = ""
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
          useLegacyMaxTokensField: configuration.useLegacyMaxTokensField,
          enableStrictModeForTools: configuration.enableStrictModeForTools,
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
          if let refusalTextChunk = snapshot.refusal {
            fullRefusalText += refusalTextChunk
          }
          if let notes = snapshot.notes {
            notesText = notes
          }
          if !snapshot.toolCalls.isEmpty {
            toolCalls = snapshot.toolCalls
          }
          // Capture metadata from chunks (metadata accumulates, later values override)
          if let chunkMetadata = chunk.metadata {
            finalMetadata = chunkMetadata
          }
          let fullReasoningTextCopy = fullReasoningText
          let fullResponseTextCopy = fullResponseText
          let fullRefusalTextCopy = fullRefusalText
          let notesTextCopy = notesText
          let toolCallsCopy = toolCalls
          let metadataCopy = finalMetadata
          await MainActor.run {
            update(.init(
              content: Self.assistantContent(
                reasoningText: fullReasoningTextCopy.isEmpty ? nil : fullReasoningTextCopy,
                responseText: fullResponseTextCopy.isEmpty ? nil : fullResponseTextCopy,
                refusalText: fullRefusalTextCopy.isEmpty ? nil : fullRefusalTextCopy,
                notesText: notesTextCopy,
                toolCalls: toolCallsCopy,
              ),
              metadata: metadataCopy,
            ))
          }
        }
        return .init(
          content: Self.assistantContent(
            reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
            responseText: fullResponseText.isEmpty ? nil : fullResponseText,
            refusalText: fullRefusalText.isEmpty ? nil : fullRefusalText,
            notesText: notesText,
            toolCalls: toolCalls,
          ),
          metadata: finalMetadata,
        )
      } catch {
        if error is CancellationError {
          return .init(
            content: Self.assistantContent(
              reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
              responseText: fullResponseText.isEmpty ? nil : fullResponseText,
              refusalText: fullRefusalText.isEmpty ? nil : fullRefusalText,
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
    let result = await task.result
    await cleanUpGeneration()
    return try result.get()
  }

  @MainActor
  private func cleanUpGeneration() {
    isGenerating = false
    currentTask = nil
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
    let refusal: String?
    let reasoningContent: String?
    let toolCalls: [ToolCallDelta]?

    enum CodingKeys: String, CodingKey {
      case content
      case refusal
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
    let refusal: String?
    let reasoningContent: String?
    let toolCalls: [ToolCall]?
    let annotations: [Annotation]?

    enum CodingKeys: String, CodingKey {
      case content
      case refusal
      case reasoningContent = "reasoning_content"
      case toolCalls = "tool_calls"
      case annotations
    }
  }

  struct Annotation: Decodable {
    let type: String?
    let urlCitation: URLCitation?

    enum CodingKeys: String, CodingKey {
      case type
      case urlCitation = "url_citation"
    }
  }

  struct URLCitation: Decodable {
    let title: String?
    let url: String?
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
