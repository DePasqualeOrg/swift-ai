// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log
import SSE

/// A client for the OpenAI Responses API.
///
/// The Responses API is OpenAI's newer API format that supports built-in tools
/// like web search and file search, as well as custom function tools. Also works
/// with xAI (Grok) models.
///
/// ## Example
///
/// ```swift
/// let client = ResponsesClient()
/// let response = try await client.generateText(
///   modelId: "gpt-4o",
///   prompt: "Hello!",
///   apiKey: "your-api-key"
/// )
/// print(response.content)
/// ```
@Observable
public final class ResponsesClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image, .file]

  /// Predefined API endpoints for the Responses API.
  public enum Endpoint {
    /// OpenAI's Responses API endpoint.
    case openAI
    /// xAI's Responses API endpoint.
    case xAI

    /// The URL for this endpoint.
    public var url: URL {
      switch self {
        case .openAI: URL(string: "https://api.openai.com/v1/responses")!
        case .xAI: URL(string: "https://api.x.ai/v1/responses")!
      }
    }
  }

  /// The API endpoint URL used by this client.
  public let endpoint: URL

  @MainActor public private(set) var isGenerating: Bool = false
  @MainActor private var currentTask: Task<GenerationResponse, Error>?
  /// The ID of the currently active background response, if any
  /// This can be used to manually interrupt and resume background streams
  @MainActor public private(set) var activeBackgroundResponseId: String?
  /// The API key associated with the active background response, used for authenticated cancellation
  @MainActor private var activeBackgroundResponseApiKey: String?

  private let session: URLSession

  /// Creates a new Responses client with a predefined endpoint.
  ///
  /// - Parameters:
  ///   - endpoint: The API endpoint to use (OpenAI or xAI).
  ///   - session: URLSession to use for requests.
  public init(endpoint: Endpoint = .openAI, session: URLSession = .shared) {
    self.endpoint = endpoint.url
    self.session = session
  }

  /// Creates a new Responses client with a custom endpoint URL.
  ///
  /// - Parameters:
  ///   - endpoint: Custom endpoint URL for the Responses API.
  ///   - session: URLSession to use for requests.
  public init(endpoint: URL, session: URLSession = .shared) {
    self.endpoint = endpoint
    self.session = session
  }

  // MARK: - Content Type Constants

  private enum ContentType {
    static let inputText = "input_text"
    static let inputImage = "input_image"
    static let inputFile = "input_file"
    static let outputText = "output_text"
    static let message = "message"
    static let functionCall = "function_call"
    static let functionCallOutput = "function_call_output"
  }

  private struct StreamingResponseState {
    var indexedContent: [Int: Message.Content] = [:]
    var fallbackContent: [Message.Content] = []
    var toolCallArgumentBuffers: [Int: String] = [:]

    var content: [Message.Content] {
      indexedContent.keys.sorted().compactMap { indexedContent[$0] } + fallbackContent
    }

    mutating func appendTextDelta(_ delta: String, outputIndex: Int?) {
      append(delta: delta, outputIndex: outputIndex, as: { .text($0) })
    }

    mutating func appendReasoningDelta(_ delta: String, outputIndex: Int?) {
      append(delta: delta, outputIndex: outputIndex) { existing in
        let separator = existing.isEmpty || delta.isEmpty || existing.hasSuffix("\n") ? "" : "\n"
        return .thinking(text: existing + separator + delta, signature: nil)
      } create: {
        .thinking(text: delta, signature: nil)
      }
    }

    mutating func setToolCall(_ toolCall: ToolCall, outputIndex: Int?) {
      guard let outputIndex else {
        appendFallback(.toolCall(toolCall))
        return
      }
      indexedContent[outputIndex] = .toolCall(toolCall)
      toolCallArgumentBuffers[outputIndex] = ""
    }

    mutating func appendToolCallArgumentsDelta(_ delta: String, outputIndex: Int) {
      let existingArgsString = toolCallArgumentBuffers[outputIndex] ?? ""
      let newArgsString = existingArgsString + delta
      toolCallArgumentBuffers[outputIndex] = newArgsString

      guard case let .toolCall(currentToolCall)? = indexedContent[outputIndex] else { return }
      guard let argsData = newArgsString.data(using: .utf8),
            let partialArgs = try? JSONDecoder().decode([String: Value].self, from: argsData)
      else {
        return
      }

      var updatedToolCall = currentToolCall
      updatedToolCall.parameters = partialArgs
      indexedContent[outputIndex] = .toolCall(updatedToolCall)
    }

    mutating func completeToolCallArguments(_ argumentsString: String, outputIndex: Int) {
      toolCallArgumentBuffers.removeValue(forKey: outputIndex)
      guard case let .toolCall(currentToolCall)? = indexedContent[outputIndex] else { return }

      var updatedToolCall = currentToolCall
      if let argumentsData = argumentsString.data(using: .utf8),
         let parsedArguments = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
      {
        updatedToolCall.parameters = parsedArguments
      } else {
        openAIResponsesLogger.error("Failed to parse final function call arguments for output index \(outputIndex): \(argumentsString)")
        updatedToolCall.parameters = ["_parseError": .string("Failed to parse arguments JSON")]
      }

      indexedContent[outputIndex] = .toolCall(updatedToolCall)
    }

    private mutating func append(
      delta: String,
      outputIndex: Int?,
      as createBlock: (String) -> Message.Content,
    ) {
      append(delta: delta, outputIndex: outputIndex) { existing in
        createBlock(existing + delta)
      } create: {
        createBlock(delta)
      }
    }

    private mutating func append(
      delta _: String,
      outputIndex: Int?,
      update: (String) -> Message.Content,
      create: () -> Message.Content,
    ) {
      guard let outputIndex else {
        appendFallback(create())
        return
      }

      switch indexedContent[outputIndex] {
        case let .text(existingText)?:
          indexedContent[outputIndex] = update(existingText)
        case let .thinking(text: existingText, signature: nil)?:
          indexedContent[outputIndex] = update(existingText)
        default:
          indexedContent[outputIndex] = create()
      }
    }

    private mutating func appendFallback(_ block: Message.Content) {
      guard let lastBlock = fallbackContent.last else {
        fallbackContent.append(block)
        return
      }

      switch (lastBlock, block) {
        case let (.text(existing), .text(delta)):
          fallbackContent[fallbackContent.count - 1] = .text(existing + delta)
        case let (.thinking(text: existing, signature: existingSignature), .thinking(text: delta, signature: nil)):
          fallbackContent[fallbackContent.count - 1] = .thinking(text: existing + (existing.hasSuffix("\n") ? "" : "\n") + delta, signature: existingSignature)
        default:
          fallbackContent.append(block)
      }
    }
  }

  private static func inputItems(for message: Message) async throws -> [[String: any Sendable]] {
    switch message.role {
      case .user:
        var contentItems: [[String: any Sendable]] = []
        for block in message.content {
          switch block {
            case let .text(text) where !text.isEmpty:
              contentItems.append([
                "type": ContentType.inputText,
                "text": text,
              ])
            case let .attachment(attachment):
              switch attachment.kind {
                case let .image(data, mimeType):
                  let processedImageData = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
                  contentItems.append([
                    "type": ContentType.inputImage,
                    "detail": "auto",
                    "image_url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: mimeType),
                  ])
                case let .document(data, mimeType):
                  // The API expects a data URL (e.g. "data:application/pdf;base64,...") for file_data,
                  // despite the OpenAI TS SDK describing it as "base64-encoded data".
                  var contentItem: [String: any Sendable] = [
                    "type": ContentType.inputFile,
                    "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
                  ]
                  if let fileName = attachment.filename {
                    contentItem["filename"] = fileName
                  }
                  contentItems.append(contentItem)
                case .video, .audio:
                  break
              }
            default:
              break
          }
        }
        guard !contentItems.isEmpty else { return [] }
        return [[
          "type": ContentType.message,
          "role": "user",
          "content": contentItems,
        ]]

      case .assistant:
        var items: [[String: any Sendable]] = []
        var contentItems: [[String: any Sendable]] = []
        // Extract phase from opaque blocks if present
        var phase: String?
        for block in message.content {
          if case let .providerOpaque(opaqueBlock) = block,
             opaqueBlock.provider == "openai-responses",
             opaqueBlock.type == "phase"
          {
            phase = opaqueBlock.content
          }
        }

        func flushContentItems() {
          guard !contentItems.isEmpty else { return }
          var messageItem: [String: any Sendable] = [
            "type": ContentType.message,
            "role": "assistant",
            "content": contentItems,
          ]
          if let phase {
            messageItem["phase"] = phase
          }
          items.append(messageItem)
          contentItems.removeAll(keepingCapacity: true)
        }

        for block in message.content {
          switch block {
            case let .text(text) where !text.isEmpty:
              contentItems.append([
                "type": ContentType.inputText,
                "text": text,
              ])
            case let .toolCall(toolCall):
              flushContentItems()
              let foundationParams = Value.toSendable(toolCall.parameters)
              let argumentsData = try JSONSerialization.data(withJSONObject: foundationParams, options: [])
              guard let argumentsString = String(data: argumentsData, encoding: .utf8) else {
                throw AIError.invalidRequest(message: "Failed to serialize function call arguments to JSON string")
              }
              items.append([
                "type": ContentType.functionCall,
                "call_id": toolCall.id,
                "name": toolCall.name,
                "arguments": argumentsString,
              ])
            case let .providerOpaque(block) where block.provider == "openai-responses" && block.type == "reasoning":
              flushContentItems()
              var reasoningItem: [String: any Sendable] = [
                "type": OutputItemType.reasoning,
              ]
              if let id = block.signature {
                reasoningItem["id"] = id
              }
              if let summaryText = block.content {
                reasoningItem["summary"] = [["type": "summary_text", "text": summaryText]]
              }
              if let encryptedContent = block.data {
                reasoningItem["encrypted_content"] = encryptedContent
              }
              items.append(reasoningItem)
            default:
              break
          }
        }

        flushContentItems()
        return items

      case .tool:
        return message.content.compactMap { block -> [String: any Sendable]? in
          guard case let .toolResult(toolResult) = block else { return nil }

          let resultOutput: any Sendable
          if toolResult.isError == true {
            let errorText = toolResult.content.compactMap { content -> String? in
              if case let .text(text) = content { return text }
              return nil
            }.joined(separator: "\n")
            resultOutput = "{\"error\": \"\((errorText.isEmpty ? "Unknown error" : errorText).replacingOccurrences(of: "\"", with: "\\\""))\"}"
          } else {
            var outputItems: [[String: any Sendable]] = []
            var hasNonTextContent = false

            for content in toolResult.content {
              switch content {
                case let .text(text):
                  outputItems.append([
                    "type": ContentType.inputText,
                    "text": text,
                  ])
                case let .image(data, mimeType):
                  hasNonTextContent = true
                  let mediaType = mimeType ?? "image/png"
                  let dataURL = "data:\(mediaType);base64,\(data.base64EncodedString())"
                  outputItems.append([
                    "type": ContentType.inputImage,
                    "detail": "auto",
                    "image_url": dataURL,
                  ])
                case let .audio(data, mimeType):
                  openAIResponsesLogger.warning("Tool '\(toolResult.name)' returned audio, which is not supported by Responses API. Using fallback text.")
                  outputItems.append([
                    "type": ContentType.inputText,
                    "text": ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription,
                  ])
                case let .file(data, mimeType, filename):
                  hasNonTextContent = true
                  if mimeType.hasPrefix("image/") {
                    let dataURL = "data:\(mimeType);base64,\(data.base64EncodedString())"
                    outputItems.append([
                      "type": ContentType.inputImage,
                      "detail": "auto",
                      "image_url": dataURL,
                    ])
                  } else {
                    // Use data URL format consistent with input file encoding
                    var fileItem: [String: any Sendable] = [
                      "type": ContentType.inputFile,
                      "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
                    ]
                    if let filename {
                      fileItem["filename"] = filename
                    }
                    outputItems.append(fileItem)
                  }
              }
            }

            if hasNonTextContent {
              resultOutput = outputItems
            } else {
              // Text-only: use plain string shorthand
              let texts = outputItems.compactMap { $0["text"] as? String }
              resultOutput = texts.joined(separator: "\n")
            }
          }

          return [
            "type": ContentType.functionCallOutput,
            "call_id": toolResult.id,
            "output": resultOutput,
          ]
        }

      case .system, .developer:
        var contentItems: [[String: any Sendable]] = []
        for block in message.content {
          if case let .text(text) = block, !text.isEmpty {
            contentItems.append([
              "type": ContentType.inputText,
              "text": text,
            ])
          }
        }
        guard !contentItems.isEmpty else { return [] }
        return [[
          "type": ContentType.message,
          "role": message.role.rawValue,
          "content": contentItems,
        ]]
    }
  }

  // MARK: - Streaming Event Types

  private enum StreamEventType {
    static let outputTextDelta = "response.output_text.delta"
    static let reasoningTextDelta = "response.reasoning_text.delta"
    static let reasoningDelta = "response.reasoning.delta" // Deprecated, kept for older API versions
    static let reasoningSummaryDelta = "response.reasoning_summary_text.delta"
    static let outputItemAdded = "response.output_item.added"
    static let contentPartAdded = "response.content_part.added"
    static let refusalDelta = "response.refusal.delta"
    static let functionCallArgumentsDelta = "response.function_call_arguments.delta"
    static let functionCallArgumentsDone = "response.function_call_arguments.done"
    static let completed = "response.completed"
    static let created = "response.created"
  }

  // MARK: - Output Item Types

  private enum OutputItemType {
    static let functionCall = "function_call"
    static let reasoning = "reasoning"
    static let codeInterpreterCall = "code_interpreter_call"
    static let webSearchCall = "web_search_call"
    static let message = "message"
    static let outputText = "output_text"
    static let refusal = "refusal"
  }

  private let enableStrictModeForTools = true

  private func streamResponse(
    input: [Message],
    systemPrompt: String?,
    modelId: String,
    apiKey: String?,
    maxTokens: Int?,
    temperature: Float?,
    stream: Bool,
    reasoningEffortLevel: ReasoningEffortLevel?,
    verbosityLevel: VerbosityLevel?,
    serverSideTools: [ServerSideTool],
    backgroundMode: Bool,
    textFormat: ResponseFormat? = nil,
    tools: [Tool] = [],
  ) async throws -> AsyncThrowingStream<GenerationResponse, Error> {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = 600.0 // 10 minutes - suitable for o3's long response times
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let patchedInput = Message.patchingOrphanedToolCalls(input)
    var inputContent: [[String: any Sendable]] = []
    for message in patchedInput {
      try await inputContent.append(contentsOf: Self.inputItems(for: message))
    }

    var body: [String: any Sendable] = [
      "model": modelId,
      "input": inputContent, // Use the constructed input array
      "stream": stream,
    ]

    // Add background mode support
    if backgroundMode {
      body["background"] = true
      body["store"] = true // Required for background mode
    }

    var toolsArray: [[String: any Sendable]] = []

    // Server-side tools (provider-specific)
    for serverSideTool in serverSideTools {
      toolsArray.append(serverSideTool.definition)
    }

    // Function calling tools - rawInputSchema is always populated
    if !tools.isEmpty {
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
          "name": tool.name,
          "description": tool.description,
          "parameters": parameters,
          "strict": enableStrictModeForTools,
        ])
      }
    }

    if !toolsArray.isEmpty {
      body["tools"] = toolsArray
      // Set tool_choice to auto by default when tools are present
      body["tool_choice"] = "auto" // Can also be "required", "none", or {"type": "function", "name": "my_func"}
    }

    // Add system prompt as instructions
    if let systemPrompt, !systemPrompt.isEmpty {
      body["instructions"] = systemPrompt
    }

    if let maxTokens {
      body["max_output_tokens"] = maxTokens
    }

    if let temperature {
      body["temperature"] = temperature
    }

    if let reasoningEffortLevel {
      body["reasoning"] = [
        "effort": reasoningEffortLevel.rawValue,
        "summary": "auto",
      ]
    }

    // Build text configuration (format and/or verbosity)
    var textConfig: [String: any Sendable] = [:]
    if let textFormat {
      var formatConfig: [String: any Sendable] = [:]
      switch textFormat {
        case .text:
          formatConfig["type"] = "text"
        case .jsonObject:
          formatConfig["type"] = "json_object" // Use this for older JSON mode if needed
        case let .jsonSchema(schema, name, description):
          formatConfig["type"] = "json_schema" // Preferred for structured JSON
          formatConfig["schema"] = schema // Should be [String: any Sendable] representing the JSON schema
          if let name { formatConfig["name"] = name }
          if let description { formatConfig["description"] = description }
          // Add strict mode if desired, e.g., formatConfig["strict"] = true
      }
      textConfig["format"] = formatConfig
    }
    if let verbosityLevel {
      textConfig["verbosity"] = verbosityLevel.rawValue
    }
    if !textConfig.isEmpty {
      body["text"] = textConfig
    }

    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let finalRequest = request

    let (resultStream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
    let streamTask = Task { @Sendable in
      let request = finalRequest
      do {
        if backgroundMode, stream {
          openAIResponsesLogger.log("Initiating background mode response with streaming in OpenAI Responses client")
          // For background mode with streaming, stream directly but with proper retry logic
          try await streamBackgroundResponseDirect(
            request: request,
            apiKey: apiKey,
            continuation: continuation,
          )
        } else if backgroundMode {
          openAIResponsesLogger.log("Initiating background mode response without streaming in OpenAI Responses client")
          // For background mode without streaming, get response ID and poll
          let (data, response) = try await session.data(for: request)
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          if !(200 ... 299).contains(httpResponse.statusCode) {
            try handleErrorResponse(httpResponse, data: data)
          }

          let decodedResponse = try JSONDecoder().decode(ResponseObject.self, from: data)
          guard let responseId = decodedResponse.id else {
            throw AIError.parsing(message: "Failed to parse background response ID")
          }
          await MainActor.run {
            activeBackgroundResponseId = responseId
            activeBackgroundResponseApiKey = apiKey
          }
          // Poll for completion
          try await pollBackgroundResponse(responseId: responseId, apiKey: apiKey, continuation: continuation)
        } else if stream {
          openAIResponsesLogger.log("Initiating standard streamed response in OpenAI Responses client")
          try await performSSEStream(
            request: request,
            continuation: continuation,
            logPrefix: "Standard Stream",
          )
        } else {
          let (data, response) = try await session.data(for: request)
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          if !(200 ... 299).contains(httpResponse.statusCode) {
            try handleErrorResponse(httpResponse, data: data)
          }

          do {
            let response = try JSONDecoder().decode(ResponseObject.self, from: data)
            continuation.yield(response.toGenerationResponse())
          } catch {
            openAIResponsesLogger.error("Non-streaming response parsing error: \(error)")
            throw AIError.parsing(message: "Failed to parse non-streamed response: \(error.localizedDescription)")
          }
        }
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }

    // Set up cancellation handler to cancel the stream task when the consumer cancels
    continuation.onTermination = { @Sendable termination in
      if case .cancelled = termination {
        openAIResponsesLogger.log("AsyncThrowingStream cancelled by consumer - cancelling stream task")
        streamTask.cancel()
      }
    }
    return resultStream
  }

  private func handleErrorResponse(_ httpResponse: HTTPURLResponse, data: Data) throws {
    try AIError.throwOpenAIHTTPError(httpResponse, data: data, logger: openAIResponsesLogger)
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
        // Only yield the final response if no updates were emitted (e.g. early cancellation)
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
    await MainActor.run {
      isGenerating = true
    }

    let task = Task<GenerationResponse, Error> {
      var finalContent: [Message.Content] = []
      var finalMetadata: GenerationResponse.Metadata?

      do {
        let stream = try await streamResponse(
          input: messages,
          systemPrompt: systemPrompt,
          modelId: modelId,
          apiKey: apiKey,
          maxTokens: maxTokens,
          temperature: temperature,
          stream: stream,
          reasoningEffortLevel: configuration.reasoningEffortLevel,
          verbosityLevel: configuration.verbosityLevel,
          serverSideTools: configuration.serverSideTools,
          backgroundMode: configuration.backgroundMode,
          tools: tools,
        )

        for try await chunk in stream {
          try Task.checkCancellation()

          finalContent = chunk.content
          finalMetadata = chunk.metadata

          await MainActor.run {
            update(chunk)
          }
        }

        // If cancelled, return the state as it was when cancellation was detected
        if Task.isCancelled {
          openAIResponsesLogger.log("Generation task returning cancelled state")
          return .init(content: finalContent, metadata: finalMetadata)
        }

        // Stream finished normally, return the final state
        return .init(content: finalContent, metadata: finalMetadata)

      } catch {
        // Handle cancellation error specifically if it bubbles up
        if error is CancellationError || (error as NSError).code == NSURLErrorCancelled {
          openAIResponsesLogger.log("Generation task caught cancellation error.")
          return .init(content: finalContent, metadata: finalMetadata)
        } else {
          // Rethrow other errors
          openAIResponsesLogger.error("Generation failed: \(error)")
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
    activeBackgroundResponseId = nil
    activeBackgroundResponseApiKey = nil
  }

  /// Cancels any ongoing generation task and active background response.
  @MainActor
  public func stop() {
    openAIResponsesLogger.log("Stop called - cancelling current task")
    currentTask?.cancel()
    // Also cancel background response if one is active
    if let backgroundResponseId = activeBackgroundResponseId {
      let apiKey = activeBackgroundResponseApiKey
      openAIResponsesLogger.log("Stop called - cancelling background response \(backgroundResponseId)")
      Task {
        try? await cancelBackgroundResponse(responseId: backgroundResponseId, apiKey: apiKey)
      }
    }
  }

  // MARK: - Shared Event Processing

  private func processStreamingEvent(
    event: StreamEvent,
    streamingState: inout StreamingResponseState,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) throws {
    if let errorMessage = event.error?.message {
      throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
    }

    guard let eventType = event.type else { return }

    func yieldCurrentState() {
      continuation.yield(GenerationResponse(content: streamingState.content))
    }

    switch eventType {
      case StreamEventType.outputTextDelta:
        if let delta = event.delta {
          streamingState.appendTextDelta(delta, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.refusalDelta:
        if let delta = event.delta {
          streamingState.appendTextDelta(delta, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.reasoningTextDelta, StreamEventType.reasoningDelta, StreamEventType.reasoningSummaryDelta:
        if let delta = event.delta {
          streamingState.appendReasoningDelta(delta, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.contentPartAdded:
        break // Content parts are initialized via output_item.added; deltas arrive via reasoning_text.delta

      case StreamEventType.outputItemAdded:
        if let item = event.item, let itemType = item.type {
          switch itemType {
            case OutputItemType.functionCall:
              if let name = item.name, let callId = item.callId {
                streamingState.setToolCall(ToolCall(
                  name: name,
                  id: callId,
                  parameters: [:],
                ), outputIndex: event.outputIndex)
                yieldCurrentState()
              }
            case OutputItemType.reasoning:
              if let summaryArray = item.summary {
                let summaryText = summaryArray
                  .compactMap(\.text)
                  .joined(separator: "\n")
                if !summaryText.isEmpty {
                  streamingState.appendReasoningDelta(summaryText, outputIndex: event.outputIndex)
                  yieldCurrentState()
                }
              } else {
                openAIResponsesLogger.warning("Received reasoning item without expected summary array")
              }
            case OutputItemType.codeInterpreterCall:
              openAIResponsesLogger.log("Received code_interpreter_call item")
            case OutputItemType.webSearchCall:
              openAIResponsesLogger.log("Received web_search_call item")
            case OutputItemType.message:
              openAIResponsesLogger.log("Received message item")
            default:
              openAIResponsesLogger.log("Ignoring added output item type: \(itemType)")
          }
        }

      case StreamEventType.functionCallArgumentsDelta:
        if let delta = event.delta,
           let outputIndex = event.outputIndex
        {
          streamingState.appendToolCallArgumentsDelta(delta, outputIndex: outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.functionCallArgumentsDone:
        if let argumentsString = event.arguments,
           let outputIndex = event.outputIndex
        {
          streamingState.completeToolCallArguments(argumentsString, outputIndex: outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.completed:
        // Build and yield final response with metadata
        if let response = event.response {
          let generationResponse = response.toGenerationResponse()
          var finalContent = generationResponse.content

          // Some Responses API streams omit output_index on intermediate
          // function-call events. In that case, recover tool calls from the
          // final completed response payload instead of dropping them.
          var seenToolCallIDs = Set(finalContent.compactMap { block -> String? in
            guard case let .toolCall(toolCall) = block else { return nil }
            return toolCall.id
          })
          for toolCall in streamingState.content.compactMap({ block -> ToolCall? in
            guard case let .toolCall(toolCall) = block else { return nil }
            return toolCall
          }) where !seenToolCallIDs.contains(toolCall.id) {
            finalContent.append(.toolCall(toolCall))
            seenToolCallIDs.insert(toolCall.id)
          }

          continuation.yield(GenerationResponse(
            content: finalContent,
            metadata: generationResponse.metadata,
          ))
        }

      default:
        break
    }
  }

  // MARK: - Background Response Methods

  private func streamBackgroundResponseDirect(
    request: URLRequest,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    retryCount: Int = 0,
    maxRetries: Int = 3,
  ) async throws {
    openAIResponsesLogger.log("Background Stream Direct: Starting attempt \(retryCount + 1)/\(maxRetries + 1)")

    try await performBackgroundStream(
      request: request,
      responseId: nil,
      apiKey: apiKey,
      continuation: continuation,
      startingAfter: 0,
      retryCount: retryCount,
      maxRetries: maxRetries,
      isDirect: true,
    )
  }

  private func streamBackgroundResponse(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    startingAfter: Int? = nil,
    retryCount: Int = 0,
    maxRetries: Int = 3,
  ) async throws {
    openAIResponsesLogger.log("Background Stream: Starting for response \(responseId), attempt \(retryCount + 1)/\(maxRetries + 1), startingAfter: \(startingAfter ?? 0)")

    let streamUrl = endpoint.appendingPathComponent(responseId)
    guard var urlComponents = URLComponents(url: streamUrl, resolvingAgainstBaseURL: false) else {
      throw AIError.invalidRequest(message: "Failed to construct URL components for response: \(responseId)")
    }
    urlComponents.queryItems = [
      URLQueryItem(name: "stream", value: "true"),
    ]
    if let startingAfter {
      urlComponents.queryItems?.append(URLQueryItem(name: "starting_after", value: String(startingAfter)))
    }
    guard let requestURL = urlComponents.url else {
      throw AIError.invalidRequest(message: "Failed to construct request URL for response: \(responseId)")
    }

    var request = URLRequest(url: requestURL)
    request.httpMethod = "GET"
    request.timeoutInterval = 600.0 // 10 minutes - suitable for o3's long response times
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    try await performBackgroundStream(
      request: request,
      responseId: responseId,
      apiKey: apiKey,
      continuation: continuation,
      startingAfter: startingAfter ?? 0,
      retryCount: retryCount,
      maxRetries: maxRetries,
      isDirect: false,
    )
  }

  /// Shared streaming logic for both direct and resumption modes
  private func performBackgroundStream(
    request: URLRequest,
    responseId: String?,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    startingAfter: Int,
    retryCount: Int,
    maxRetries: Int,
    isDirect: Bool,
  ) async throws {
    let logPrefix = isDirect ? "Background Stream Direct" : "Background Stream"
    var lastSequenceNumber: Int = startingAfter
    var currentResponseId: String? = responseId

    do {
      try await performSSEStream(
        request: request,
        continuation: continuation,
        logPrefix: logPrefix,
        responseIdHandler: isDirect ? { id in
          guard !Task.isCancelled else { return }
          currentResponseId = id
          openAIResponsesLogger.log("\(logPrefix): Got response ID: \(id)")
          await MainActor.run { [weak self] in
            self?.activeBackgroundResponseId = id
            self?.activeBackgroundResponseApiKey = apiKey
          }
        } : nil,
        sequenceHandler: { sequenceNumber in
          guard !Task.isCancelled else { return }
          if sequenceNumber > lastSequenceNumber {
            // Only log every 10 sequences to reduce noise
            if sequenceNumber % 10 == 0 || sequenceNumber - lastSequenceNumber > 10 {
              openAIResponsesLogger.log("\(logPrefix): Progress update - sequence: \(sequenceNumber)")
            }
          }
          lastSequenceNumber = sequenceNumber
        },
      )
    } catch {
      // Check if this is a cancellation error (expected when user stops)
      let isCancellationError = error is CancellationError || (error as NSError).code == NSURLErrorCancelled

      if isCancellationError {
        openAIResponsesLogger.log("\(logPrefix): Stream cancelled by user")
        // Cancel the background response on the server if we have a response ID
        if let responseId = currentResponseId {
          openAIResponsesLogger.log("\(logPrefix): Cancelling background response \(responseId) on server")
          try? await cancelBackgroundResponse(responseId: responseId, apiKey: apiKey)
        }
        return // Don't retry or throw for cancellation
      }

      openAIResponsesLogger.log("\(logPrefix): Error occurred - \(error)")
      // Handle timeout and connection errors with retry logic
      if retryCount < maxRetries {
        let isTimeoutError = (error as NSError).code == NSURLErrorTimedOut ||
          (error as NSError).code == NSURLErrorNetworkConnectionLost ||
          (error as NSError).code == NSURLErrorCannotConnectToHost

        if isTimeoutError {
          openAIResponsesLogger.warning("\(logPrefix) timeout (attempt \(retryCount + 1)/\(maxRetries + 1))")

          // If we have a response ID and are in direct mode, switch to resumption mode
          if isDirect, let responseId = currentResponseId {
            openAIResponsesLogger.log("\(logPrefix): Switching to resumption mode with response ID: \(responseId)")
            try await streamBackgroundResponse(
              responseId: responseId,
              apiKey: apiKey,
              continuation: continuation,
              startingAfter: lastSequenceNumber,
              retryCount: retryCount,
              maxRetries: maxRetries,
            )
            return
          }

          // Check response status before retrying (only if we have a response ID)
          if let responseId = currentResponseId ?? responseId {
            openAIResponsesLogger.log("\(logPrefix): Checking response status before retry...")
            if try await checkResponseStatusAndHandle(
              responseId: responseId,
              apiKey: apiKey,
              continuation: continuation,
              logPrefix: logPrefix,
            ) {
              return // Response was completed or handled
            }
          }

          // Exponential backoff and retry
          let backoffDelay = TimeInterval(pow(2.0, Double(retryCount)))
          openAIResponsesLogger.log("\(logPrefix): Waiting \(backoffDelay)s before retry...")
          try await Task.sleep(nanoseconds: UInt64(backoffDelay * 1_000_000_000))

          if isDirect {
            openAIResponsesLogger.log("\(logPrefix): Retrying initial request...")
            try await streamBackgroundResponseDirect(
              request: request,
              apiKey: apiKey,
              continuation: continuation,
              retryCount: retryCount + 1,
              maxRetries: maxRetries,
            )
          } else {
            openAIResponsesLogger.log("\(logPrefix): Retrying from sequence \(lastSequenceNumber)...")
            try await streamBackgroundResponse(
              responseId: responseId!,
              apiKey: apiKey,
              continuation: continuation,
              startingAfter: lastSequenceNumber,
              retryCount: retryCount + 1,
              maxRetries: maxRetries,
            )
          }
          return
        } else {
          openAIResponsesLogger.log("\(logPrefix): Non-retryable error (code: \((error as NSError).code)): \(error)")
        }
      } else {
        openAIResponsesLogger.log("\(logPrefix): Max retries exceeded (\(maxRetries))")
      }

      // If not a timeout error or max retries exceeded, rethrow
      // But don't log as error for cancellation since that's expected
      if (error as NSError).code != NSURLErrorCancelled {
        openAIResponsesLogger.error("\(logPrefix) failed after \(retryCount) retries: \(error)")
      } else {
        openAIResponsesLogger.log("\(logPrefix): Stream cancelled after \(retryCount) retries")
      }
      throw error
    }
  }

  /// Shared SSE (Server-Sent Events) streaming logic for all stream types
  private func performSSEStream(
    request: URLRequest,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    logPrefix: String,
    responseIdHandler: ((String) async -> Void)? = nil,
    sequenceHandler: ((Int) -> Void)? = nil,
  ) async throws {
    openAIResponsesLogger.log("\(logPrefix): Connecting to stream...")

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

    var streamingState = StreamingResponseState()

    for try await event in result.events {
      try Task.checkCancellation()
      let jsonString = event.data

      if jsonString == "[DONE]" {
        break
      }

      guard let jsonData = jsonString.data(using: .utf8) else {
        throw AIError.parsing(message: "Failed to convert streamed response to data: \(jsonString)")
      }

      do {
        let event = try JSONDecoder().decode(StreamEvent.self, from: jsonData)

        // Extract response ID from first event (for background direct mode)
        if event.type == StreamEventType.created, let id = event.response?.id {
          if let responseIdHandler {
            await responseIdHandler(id)
          }
        }

        // Track sequence number for resumption (background mode only)
        if let sequenceHandler, let sequenceNumber = event.sequenceNumber {
          sequenceHandler(sequenceNumber)
        }

        try processStreamingEvent(
          event: event,
          streamingState: &streamingState,
          continuation: continuation,
        )
      } catch let error as AIError {
        throw error
      } catch {
        openAIResponsesLogger.error("\(logPrefix) parsing error for JSON: \(jsonString). Error: \(error)")
        throw AIError.parsing(message: "Failed to parse streamed JSON: \(jsonString)")
      }
    }
  }

  /// Helper method to check response status and handle completion/failure
  private func checkResponseStatusAndHandle(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    logPrefix: String,
  ) async throws -> Bool {
    do {
      let statusUrl = endpoint.appendingPathComponent(responseId)
      var statusRequest = URLRequest(url: statusUrl)
      statusRequest.httpMethod = "GET"
      if let apiKey {
        statusRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
      }
      let (statusData, statusResponse) = try await session.data(for: statusRequest)
      if let httpResponse = statusResponse as? HTTPURLResponse,
         (200 ... 299).contains(httpResponse.statusCode),
         let response = try? JSONDecoder().decode(ResponseObject.self, from: statusData),
         let statusString = response.status
      {
        let status = BackgroundResponseStatus(rawValue: statusString)

        switch status {
          case .completed:
            openAIResponsesLogger.log("\(logPrefix): Response completed during disconnection. Parsing final response.")
            parseCompletedResponse(response, continuation: continuation)
            return true
          case .failed:
            let errorMessage = response.error?.message ?? "Background response failed"
            openAIResponsesLogger.log("\(logPrefix): Response failed - \(errorMessage)")
            throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
          case .cancelled:
            openAIResponsesLogger.log("\(logPrefix): Response was cancelled")
            return true
          case .queued, .in_progress:
            openAIResponsesLogger.log("\(logPrefix): Response still in progress (status: \(statusString)). Continuing retry...")
            return false
          case .none:
            openAIResponsesLogger.warning("\(logPrefix): Unknown response status: \(statusString). Continuing retry...")
            return false
        }
      }
    } catch {
      openAIResponsesLogger.warning("\(logPrefix): Could not check response status: \(error). Continuing with retry...")
    }
    return false
  }

  /// Helper method to parse completed response
  private func parseCompletedResponse(
    _ response: ResponseObject,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) {
    continuation.yield(response.toGenerationResponse())
  }

  private func pollBackgroundResponse(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) async throws {
    let pollUrl = endpoint.appendingPathComponent(responseId)
    var request = URLRequest(url: pollUrl)
    request.httpMethod = "GET"
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    while true {
      try Task.checkCancellation()

      let (data, urlResponse) = try await session.data(for: request)
      guard let httpResponse = urlResponse as? HTTPURLResponse else {
        throw AIError.network(underlying: URLError(.badServerResponse))
      }
      if !(200 ... 299).contains(httpResponse.statusCode) {
        try handleErrorResponse(httpResponse, data: data)
      }
      let response = try JSONDecoder().decode(ResponseObject.self, from: data)
      guard let statusString = response.status,
            let status = BackgroundResponseStatus(rawValue: statusString)
      else {
        throw AIError.parsing(message: "Failed to parse background response status")
      }
      switch status {
        case .queued, .in_progress:
          // Continue polling
          try await Task.sleep(nanoseconds: 2_000_000_000) // Sleep for 2 seconds
          continue

        case .completed:
          parseCompletedResponse(response, continuation: continuation)
          return

        case .failed:
          let errorMessage = response.error?.message ?? "Background response failed"
          throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)

        case .cancelled:
          openAIResponsesLogger.log("Background response was cancelled")
          return
      }
    }
  }

  /// Gets the current status of a background response.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the background response to check.
  ///   - apiKey: API key for authentication.
  /// - Returns: The background response status and result if completed.
  public func getBackgroundResponseStatus(responseId: String, apiKey: String?) async throws -> BackgroundResponse {
    let statusUrl = endpoint.appendingPathComponent(responseId)
    var request = URLRequest(url: statusUrl)
    request.httpMethod = "GET"
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (data, urlResponse) = try await session.data(for: request)
    guard let httpResponse = urlResponse as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      try handleErrorResponse(httpResponse, data: data)
    }
    let response = try JSONDecoder().decode(ResponseObject.self, from: data)
    guard let statusString = response.status,
          let status = BackgroundResponseStatus(rawValue: statusString),
          let id = response.id
    else {
      throw AIError.parsing(message: "Failed to parse background response status")
    }

    // Parse response if completed
    let generationResponse: GenerationResponse? = if status == .completed {
      response.toGenerationResponse()
    } else {
      nil
    }

    return BackgroundResponse(
      id: id,
      status: status,
      response: generationResponse,
      error: response.error?.message,
    )
  }

  /// Cancels a background response that is in progress.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the background response to cancel.
  ///   - apiKey: API key for authentication.
  public func cancelBackgroundResponse(responseId: String, apiKey: String?) async throws {
    let cancelUrl = endpoint.appendingPathComponent(responseId).appendingPathComponent("cancel")
    var request = URLRequest(url: cancelUrl)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (_, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      // Cancelling twice is idempotent, so don't throw on certain error codes
      if httpResponse.statusCode != 409 { // Conflict - already cancelled
        throw AIError.fromHTTPStatusCode(httpResponse.statusCode)
      }
    }
  }

  /// Deletes a response and its associated data permanently.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the response to delete.
  ///   - apiKey: API key for authentication.
  public func deleteResponse(responseId: String, apiKey: String?) async throws {
    let deleteUrl = endpoint.appendingPathComponent(responseId)
    var request = URLRequest(url: deleteUrl)
    request.httpMethod = "DELETE"
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (_, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      throw AIError.fromHTTPStatusCode(httpResponse.statusCode)
    }
  }

  /// Resumes streaming from a background response at a specific sequence number.
  ///
  /// Use this to continue receiving updates after a connection was dropped.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the background response to resume.
  ///   - apiKey: API key for authentication.
  ///   - startingAfter: The sequence number to resume from.
  ///   - update: Callback invoked with each streamed response chunk.
  /// - Returns: The final generation response.
  public func resumeBackgroundStream(
    responseId: String,
    apiKey: String?,
    startingAfter: Int,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    await MainActor.run {
      isGenerating = true
      activeBackgroundResponseId = responseId
      activeBackgroundResponseApiKey = apiKey
    }

    let task = Task<GenerationResponse, Error> {
      var finalContent: [Message.Content] = []
      var finalMetadata: GenerationResponse.Metadata?

      let (stream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
      let backgroundTask = Task {
        do {
          try await streamBackgroundResponse(
            responseId: responseId,
            apiKey: apiKey,
            continuation: continuation,
            startingAfter: startingAfter,
          )
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { @Sendable _ in
        backgroundTask.cancel()
      }

      for try await chunk in stream {
        try Task.checkCancellation()

        finalContent = chunk.content
        finalMetadata = chunk.metadata

        await MainActor.run {
          update(chunk)
        }
      }

      return .init(content: finalContent, metadata: finalMetadata)
    }

    await MainActor.run {
      currentTask = task
    }
    let result = await task.result
    await cleanUpGeneration()
    return try result.get()
  }
}

extension ResponsesClient {
  /// Reasoning effort level for models that support extended thinking.
  /// Maps to the `effort` value in the API's `reasoning` object.
  public enum ReasoningEffortLevel: String, CaseIterable, Identifiable, Sendable {
    /// Minimal reasoning effort for simple tasks.
    case minimal
    /// Low reasoning effort for straightforward tasks.
    case low
    /// Medium reasoning effort for balanced performance.
    case medium
    /// High reasoning effort for complex tasks.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Response verbosity level controlling output detail.
  public enum VerbosityLevel: String, CaseIterable, Identifiable, Sendable {
    /// Concise responses.
    case low
    /// Balanced detail level.
    case medium
    /// Detailed responses.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Context size for web search results.
  public enum WebSearchContextSize: String, CaseIterable, Identifiable, Sendable {
    /// Fewer search results, lower latency.
    case low
    /// Balanced search results.
    case medium
    /// More search results, higher latency.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// A server-side tool that runs on the provider's infrastructure.
  ///
  /// Different providers support different server-side tools. Use the
  /// static factory methods on `OpenAI` or `xAI` to create tools.
  public struct ServerSideTool: Sendable, Equatable {
    /// The raw tool definition dictionary sent to the API.
    public let definition: [String: any Sendable]

    /// Creates a server-side tool from a raw definition dictionary.
    ///
    /// - Parameter definition: The tool definition as expected by the API.
    public init(_ definition: [String: any Sendable]) {
      self.definition = definition
    }

    // MARK: - OpenAI Tools

    /// Factory methods for OpenAI server-side tools.
    public enum OpenAI {
      /// Web search tool with configurable context size.
      public static func webSearch(contextSize: WebSearchContextSize) -> ServerSideTool {
        ServerSideTool([
          "type": "web_search_preview",
          "search_context_size": contextSize.rawValue,
        ])
      }

      /// Code interpreter (Python execution environment).
      public static func codeInterpreter() -> ServerSideTool {
        ServerSideTool(["type": "code_interpreter", "container": ["type": "auto"]])
      }
    }

    // MARK: - xAI Tools

    /// Factory methods for xAI server-side tools.
    public enum xAI {
      /// Web search tool for searching the internet.
      public static func webSearch() -> ServerSideTool {
        ServerSideTool(["type": "web_search"])
      }

      /// X/Twitter search tool for searching posts, users, and threads.
      public static func xSearch() -> ServerSideTool {
        ServerSideTool(["type": "x_search"])
      }

      /// Code execution tool for running Python code.
      public static func codeExecution() -> ServerSideTool {
        ServerSideTool(["type": "code_interpreter"])
      }
    }

    // MARK: - Equatable

    public static func == (lhs: ServerSideTool, rhs: ServerSideTool) -> Bool {
      // Compare by serializing to JSON since [String: any Sendable] isn't directly Equatable
      guard let lhsData = try? JSONSerialization.data(withJSONObject: lhs.definition, options: .sortedKeys),
            let rhsData = try? JSONSerialization.data(withJSONObject: rhs.definition, options: .sortedKeys)
      else {
        return false
      }
      return lhsData == rhsData
    }
  }

  /// Configuration options for Responses API requests.
  public struct Configuration: Sendable {
    /// Reasoning effort level for extended thinking models.
    public var reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel?

    /// Response verbosity level.
    public var verbosityLevel: ResponsesClient.VerbosityLevel?

    /// Server-side tools to enable (web search, code interpreter, etc.).
    public var serverSideTools: [ServerSideTool]

    /// Enable background mode for long-running requests.
    public var backgroundMode: Bool

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - reasoningEffortLevel: Reasoning effort for thinking models.
    ///   - verbosityLevel: Response verbosity level.
    ///   - serverSideTools: Server-side tools to enable.
    ///   - backgroundMode: Enable background mode for long requests.
    public init(
      reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel? = nil,
      verbosityLevel: ResponsesClient.VerbosityLevel? = nil,
      serverSideTools: [ServerSideTool] = [],
      backgroundMode: Bool = false,
    ) {
      self.reasoningEffortLevel = reasoningEffortLevel
      self.verbosityLevel = verbosityLevel
      self.serverSideTools = serverSideTools
      self.backgroundMode = backgroundMode
    }
  }

  /// Status of a background response request.
  public enum BackgroundResponseStatus: String, CaseIterable, Sendable {
    /// Request is waiting to be processed.
    case queued
    /// Request is currently being processed.
    case in_progress
    /// Request completed successfully.
    case completed
    /// Request failed with an error.
    case failed
    /// Request was cancelled.
    case cancelled
  }

  /// Information about a background response request.
  public struct BackgroundResponse: Sendable {
    /// The unique identifier for this response.
    public let id: String
    /// The current status of the response.
    public let status: BackgroundResponseStatus
    /// The generation response if completed, nil otherwise.
    public let response: GenerationResponse?
    /// Error message if the request failed.
    public let error: String?
  }

  // MARK: - Streaming Event Types

  struct StreamEvent: Decodable {
    let type: String?
    let sequenceNumber: Int?
    let delta: String?
    let itemId: String?
    let outputIndex: Int?
    let arguments: String?
    let item: OutputItem?
    let response: ResponseObject?
    let error: ErrorObject?

    enum CodingKeys: String, CodingKey {
      case type
      case sequenceNumber = "sequence_number"
      case delta
      case itemId = "item_id"
      case outputIndex = "output_index"
      case arguments
      case item
      case response
      case error
    }
  }

  struct OutputItem: Decodable {
    let type: String?
    let name: String?
    let callId: String?
    let summary: [SummaryItem]?

    enum CodingKeys: String, CodingKey {
      case type, name, summary
      case callId = "call_id"
    }
  }

  struct SummaryItem: Decodable {
    let type: String?
    let text: String?
  }

  struct Usage: Decodable {
    let inputTokens: Int?
    let outputTokens: Int?
    let totalTokens: Int?
    let inputTokensDetails: InputTokensDetails?
    let outputTokensDetails: OutputTokensDetails?

    enum CodingKeys: String, CodingKey {
      case inputTokens = "input_tokens"
      case outputTokens = "output_tokens"
      case totalTokens = "total_tokens"
      case inputTokensDetails = "input_tokens_details"
      case outputTokensDetails = "output_tokens_details"
    }
  }

  struct InputTokensDetails: Decodable {
    let cachedTokens: Int?

    enum CodingKeys: String, CodingKey {
      case cachedTokens = "cached_tokens"
    }
  }

  struct OutputTokensDetails: Decodable {
    let reasoningTokens: Int?

    enum CodingKeys: String, CodingKey {
      case reasoningTokens = "reasoning_tokens"
    }
  }

  struct IncompleteDetails: Decodable {
    let reason: String?
  }

  struct ResponseObject: Decodable {
    let id: String?
    let status: String?
    let model: String?
    let createdAt: Int?
    let output: [ResponseOutputItem]?
    let outputText: String?
    let usage: Usage?
    let error: ErrorObject?
    let incompleteDetails: IncompleteDetails?

    enum CodingKeys: String, CodingKey {
      case id, status, model, output, usage, error
      case createdAt = "created_at"
      case outputText = "output_text"
      case incompleteDetails = "incomplete_details"
    }

    /// Converts the response object to a GenerationResponse
    func toGenerationResponse() -> GenerationResponse {
      var content: [Message.Content] = []
      var hasRefusal = false

      if let outputArray = output, !outputArray.isEmpty {
        for item in outputArray {
          guard let itemType = item.type else { continue }

          switch itemType {
            case OutputItemType.message:
              if let contentArray = item.content {
                for contentItem in contentArray {
                  if contentItem.type == OutputItemType.outputText, let text = contentItem.text, !text.isEmpty {
                    content.append(.text(text))
                  } else if contentItem.type == OutputItemType.refusal, let refusal = contentItem.refusal, !refusal.isEmpty {
                    content.append(.text(refusal))
                    hasRefusal = true
                  }
                }
              }
              if let phase = item.phase {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: "phase",
                  content: phase,
                )))
              }
            case OutputItemType.reasoning:
              // Prefer reasoning text from content array (reasoning_text items),
              // fall back to summary text
              let reasoningContentText = item.content?
                .filter { $0.type == "reasoning_text" }
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
              let summaryText = item.summary?
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
              let reasoningText = (reasoningContentText?.isEmpty == false) ? reasoningContentText : summaryText
              if let reasoningText, !reasoningText.isEmpty {
                content.append(.thinking(text: reasoningText, signature: nil))
              }
              // Preserve the full reasoning item for round-tripping via the Responses API
              if let itemId = item.id {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: "reasoning",
                  content: summaryText,
                  signature: itemId,
                  data: item.encryptedContent,
                )))
              }
            case OutputItemType.functionCall:
              if let name = item.name,
                 let callId = item.callId,
                 let argumentsString = item.arguments
              {
                var parameters: [String: Value] = [:]
                if let argumentsData = argumentsString.data(using: .utf8),
                   let parsedArgs = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
                {
                  parameters = parsedArgs
                }
                content.append(.toolCall(ToolCall(
                  name: name,
                  id: callId,
                  parameters: parameters,
                )))
              }
            default:
              break
          }
        }
      } else if let outputText, !outputText.isEmpty {
        content.append(.text(outputText))
      }

      // Build metadata from response
      let toolCallCount = content.reduce(into: 0) { count, item in
        if case .toolCall = item {
          count += 1
        }
      }
      let finishReason: GenerationResponse.FinishReason? = if let status {
        switch status {
          case "completed":
            if hasRefusal { .refusal }
            else if toolCallCount > 0 { .toolUse }
            else { .stop }
          case "incomplete":
            // Check the reason for incomplete status
            switch incompleteDetails?.reason {
              case "max_output_tokens": .maxTokens
              case "content_filter": .contentFilter
              default: .other
            }
          case "failed", "cancelled": .other
          default: nil
        }
      } else {
        nil
      }

      var createdAtDate: Date?
      if let createdAt {
        createdAtDate = Date(timeIntervalSince1970: TimeInterval(createdAt))
      }

      let metadata = GenerationResponse.Metadata(
        responseId: id,
        model: model,
        createdAt: createdAtDate,
        finishReason: finishReason,
        inputTokens: usage?.inputTokens,
        outputTokens: usage?.outputTokens,
        totalTokens: usage?.totalTokens,
        cacheReadInputTokens: usage?.inputTokensDetails?.cachedTokens,
        reasoningTokens: usage?.outputTokensDetails?.reasoningTokens,
      )

      return GenerationResponse(content: content, metadata: metadata)
    }
  }

  struct ResponseOutputItem: Decodable {
    let id: String?
    let type: String?
    let content: [ContentItem]?
    let name: String?
    let callId: String?
    let arguments: String?
    let summary: [SummaryItem]?
    let encryptedContent: String?
    let phase: String?

    enum CodingKeys: String, CodingKey {
      case id, type, content, name, arguments, summary, phase
      case callId = "call_id"
      case encryptedContent = "encrypted_content"
    }
  }

  struct ContentItem: Decodable {
    let type: String?
    let text: String?
    let refusal: String?
    let annotations: [AnnotationItem]?
  }

  struct AnnotationItem: Decodable {
    let type: String?
    let url: String?
    let title: String?
  }

  struct ErrorObject: Decodable {
    let message: String?
    let code: String?
  }

  /// Checks whether a model supports the reasoning effort parameter.
  /// Defaults to true (latest paradigm), returning false only for known older models.
  static func supportsReasoning(_ modelId: String) -> Bool {
    if modelId.hasPrefix("gpt-3") || modelId.hasPrefix("gpt-4") { return false }
    if modelId.hasPrefix("chatgpt-") { return false }
    if modelId == "o1-mini" || modelId.hasPrefix("o1-mini-") { return false }
    if modelId == "o1-preview" || modelId.hasPrefix("o1-preview-") { return false }
    if modelId.hasPrefix("grok-") {
      return modelId.hasPrefix("grok-3-mini")
    }
    return true
  }
}

/// Response format options for the responses endpoint
enum ResponseFormat {
  case text
  case jsonObject
  case jsonSchema(schema: [String: any Sendable], name: String? = nil, description: String? = nil)
}

private let openAIResponsesLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "ResponsesClient")
