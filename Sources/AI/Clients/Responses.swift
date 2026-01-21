// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log

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
/// print(response.texts.response ?? "")
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

  // MARK: - Streaming Event Types

  private enum StreamEventType {
    static let outputTextDelta = "response.output_text.delta"
    static let reasoningDelta = "response.reasoning.delta"
    static let reasoningSummaryDelta = "response.reasoning_summary_text.delta"
    static let outputItemAdded = "response.output_item.added"
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
    tools: [Tool] = []
  ) async throws -> AsyncThrowingStream<GenerationResponse, Error> {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = 600.0 // 10 minutes - suitable for o3's long response times
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    var inputContent: [[String: any Sendable]] = []
    for message in input {
      switch message.role {
        case .user:
          var contentItems: [[String: any Sendable]] = []
          // Add text content if present
          if let content = message.content, !content.isEmpty {
            contentItems.append([
              "type": ContentType.inputText,
              "text": content,
            ])
          }
          // Process attachments
          if !message.attachments.isEmpty {
            for attachment in message.attachments {
              switch attachment.kind {
                case let .image(data, mimeType):
                  do {
                    let processedImageData = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
                    contentItems.append([
                      "type": ContentType.inputImage,
                      "detail": "auto",
                      "image_url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: mimeType),
                    ])
                  } catch {
                    openAIResponsesLogger.error("Failed to process image: \(error.localizedDescription)")
                    throw error
                  }
                case let .document(data, mimeType):
                  var contentItem = [
                    "type": ContentType.inputFile,
                    "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
                  ]
                  if let fileName = attachment.filename {
                    contentItem["filename"] = fileName
                  }
                  contentItems.append(contentItem)
                case .video, .audio:
                  // Not supported yet
                  break
              }
            }
          }
          // Only add user message if it has content
          if !contentItems.isEmpty {
            inputContent.append([
              "type": ContentType.message,
              "role": "user",
              "content": contentItems,
            ])
          }

        case .assistant:
          var contentItems: [[String: any Sendable]] = []
          // Add text output if present
          if let content = message.content, !content.isEmpty {
            contentItems.append([
              "type": ContentType.outputText,
              "text": content,
            ])
          }
          // Only add assistant message if it has content
          if !contentItems.isEmpty {
            inputContent.append([
              "type": ContentType.message,
              "role": "assistant",
              "content": contentItems,
            ])
          }

          // If the assistant message contained function calls that were executed,
          // add them now as separate top-level items before the function results.
          // This represents the model's turn where it decided to call functions.
          if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            for toolCall in toolCalls {
              let callId = "call_" + toolCall.id
              // Convert parameters to Foundation types before serializing
              let foundationParams = Self.convertValueToSendable(toolCall.parameters)
              let argumentsData = try JSONSerialization.data(withJSONObject: foundationParams, options: [])
              guard let argumentsString = String(data: argumentsData, encoding: .utf8) else {
                throw AIError.invalidRequest(message: "Failed to serialize function call arguments to JSON string")
              }
              inputContent.append([
                "type": ContentType.functionCall,
                "call_id": callId,
                "name": toolCall.name,
                "arguments": argumentsString,
              ])
            }
          }

        case .tool:
          // Handle function results. These become top-level items.
          // TODO: Verify OpenAI Responses API support for multi-content tool results.
          // Current approach: text as string output, images as data URLs, files as input_file.
          // Audio falls back to description text. Need to test with actual multi-content
          // responses and confirm the output format is correct per API docs.
          if let toolResults = message.toolResults, !toolResults.isEmpty {
            for toolResult in toolResults {
              // Make sure the ID has the correct "call_" prefix for the output
              // The API expects the call_id to match the original function call's ID
              let callId = "call_" + toolResult.id
              // The output can be a string or an array of content items
              let resultOutput: any Sendable

              // Handle error results
              if toolResult.isError == true {
                let errorText = toolResult.content.compactMap { content -> String? in
                  if case let .text(text) = content { return text }
                  return nil
                }.joined(separator: "\n")
                resultOutput = "{\"error\": \"\((errorText.isEmpty ? "Unknown error" : errorText).replacingOccurrences(of: "\"", with: "\\\""))\"}"
              } else {
                // Process content items
                var outputItems: [[String: any Sendable]] = []
                var textOutput: String? = nil

                for content in toolResult.content {
                  switch content {
                    case let .text(text):
                      textOutput = text
                    case let .image(data, mimeType):
                      let mediaType = mimeType ?? "image/png"
                      let dataUrl = "data:\(mediaType);base64,\(data.base64EncodedString())"
                      outputItems.append([
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": dataUrl,
                      ])
                    case let .audio(data, mimeType):
                      openAIResponsesLogger.warning("Tool '\(toolResult.name)' returned audio, which is not supported by Responses API. Using fallback text.")
                      textOutput = ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription
                    case let .file(data, mimeType, filename):
                      if mimeType.hasPrefix("image/") {
                        let dataUrl = "data:\(mimeType);base64,\(data.base64EncodedString())"
                        outputItems.append([
                          "type": "input_image",
                          "detail": "auto",
                          "image_url": dataUrl,
                        ])
                      } else {
                        var fileItem: [String: any Sendable] = [
                          "type": "input_file",
                          "file_data": data.base64EncodedString(),
                        ]
                        if let name = filename {
                          fileItem["filename"] = name
                        }
                        outputItems.append(fileItem)
                      }
                  }
                }

                // Use text if only text, otherwise use output items array
                if let text = textOutput, outputItems.isEmpty {
                  resultOutput = text
                } else if !outputItems.isEmpty {
                  resultOutput = outputItems
                } else {
                  resultOutput = textOutput ?? ""
                }
              }

              inputContent.append([
                "type": ContentType.functionCallOutput,
                "call_id": callId,
                "output": resultOutput,
              ])
            }
          }

        case .system, .developer:
          // System/Developer roles are handled by the 'instructions' parameter, not in the input array.
          // If they appear here, it might be an issue with how history is managed.
          openAIResponsesLogger.warning("System/Developer message found in input array for ResponsesClient. Ignoring, use 'instructions' parameter instead.")
      }
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
          Self.convertSchemaForStrictMode(tool.rawInputSchema)
        } else {
          Self.convertValueToSendable(tool.rawInputSchema)
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

//    // Debugging request body
//    do {
//      let jsonData = try JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys])
//      if let jsonString = String(data: jsonData, encoding: .utf8) {
//        print("Request Body JSON:\n\(jsonString)")
//      }
//    } catch {
//      openAIResponsesLogger.error("Failed to serialize request body for debugging: \(error)")
//    }

    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let finalRequest = request

    return AsyncThrowingStream { continuation in
      let streamTask = Task { @Sendable in
        let request = finalRequest
        do {
          if backgroundMode, stream {
            openAIResponsesLogger.log("Initiating background mode response with streaming in OpenAI Responses client")
            // For background mode with streaming, stream directly but with proper retry logic
            try await streamBackgroundResponseDirect(
              request: request,
              apiKey: apiKey,
              continuation: continuation
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
            }
            // Poll for completion
            try await pollBackgroundResponse(responseId: responseId, apiKey: apiKey, continuation: continuation)
          } else if stream {
            openAIResponsesLogger.log("Initiating standard streamed response in OpenAI Responses client")
            try await performSSEStream(
              request: request,
              continuation: continuation,
              logPrefix: "Standard Stream",
              isBackground: false,
              apiKey: apiKey
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
    }
  }

  private func handleErrorResponse(_ httpResponse: HTTPURLResponse, data: Data) throws {
    if let errorJson = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable] {
      openAIResponsesLogger.warning("Error: \(errorJson)")
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
    await MainActor.run {
      isGenerating = true
    }

    let task = Task<GenerationResponse, Error> {
      defer {
        Task { @MainActor in
          isGenerating = false
          currentTask = nil
          activeBackgroundResponseId = nil
        }
      }

      var finalReasoningText: String? = nil
      var finalResponseText: String? = nil
      var finalEndnotesText: String? = nil
      var finalFunctionCalls: [GenerationResponse.ToolCall] = []
      var finalMetadata: GenerationResponse.Metadata? = nil

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
          tools: tools
        )

        for try await chunk in stream {
          try Task.checkCancellation()

          // Update final state from the latest yielded response
          finalReasoningText = chunk.texts.reasoning
          finalResponseText = chunk.texts.response
          finalEndnotesText = chunk.texts.notes
          finalFunctionCalls = chunk.toolCalls // This now includes completed and streaming calls
          finalMetadata = chunk.metadata

          // Create copies for the MainActor update closure
          let reasoningCopy = finalReasoningText
          let responseCopy = finalResponseText
          let notesCopy = finalEndnotesText
          let toolCallsCopy = finalFunctionCalls

          await MainActor.run {
            update(.init(texts: .init(
              reasoning: reasoningCopy, // Pass nil if empty handled by GenerationResponse init
              response: responseCopy,
              notes: notesCopy
            ), toolCalls: toolCallsCopy))
          }
        }

        // If cancelled, return the state as it was when cancellation was detected
        if Task.isCancelled {
          openAIResponsesLogger.log("Generation task returning cancelled state")
          return .init(texts: .init(
            reasoning: finalReasoningText,
            response: finalResponseText,
            notes: finalEndnotesText
          ), toolCalls: finalFunctionCalls, metadata: finalMetadata)
        }

        // Stream finished normally, return the final state
        return .init(texts: .init(
          reasoning: finalReasoningText,
          response: finalResponseText,
          notes: finalEndnotesText
        ), toolCalls: finalFunctionCalls.filter { !$0.parameters.keys.contains("_parseError") }, metadata: finalMetadata) // Filter out calls that failed final parsing

      } catch {
        // Handle cancellation error specifically if it bubbles up
        if error is CancellationError || (error as NSError).code == NSURLErrorCancelled {
          openAIResponsesLogger.log("Generation task caught cancellation error.")
          return .init(texts: .init(
            reasoning: finalReasoningText,
            response: finalResponseText,
            notes: finalEndnotesText
          ), toolCalls: finalFunctionCalls, metadata: finalMetadata)
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

    return try await task.value
  }

  /// Cancels any ongoing generation task and active background response.
  @MainActor
  public func stop() {
    openAIResponsesLogger.log("Stop called - cancelling current task")
    currentTask?.cancel()
    // Also cancel background response if one is active
    if let backgroundResponseId = activeBackgroundResponseId {
      openAIResponsesLogger.log("Stop called - cancelling background response \(backgroundResponseId)")
      Task {
        try? await cancelBackgroundResponse(responseId: backgroundResponseId, apiKey: nil)
      }
    }
  }

  // MARK: - Shared Event Processing

  private func processStreamingEvent(
    event: StreamEvent,
    reasoningText: inout String,
    responseText: inout String,
    streamingFunctionCalls: inout [String: GenerationResponse.ToolCall],
    streamingArgumentsStrings: inout [String: String],
    completedFunctionCalls: inout [GenerationResponse.ToolCall],
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation
  ) throws {
    if let errorMessage = event.error?.message {
      throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
    }

    guard let eventType = event.type else { return }

    switch eventType {
      case StreamEventType.outputTextDelta:
        if let delta = event.delta {
          responseText += delta
          continuation.yield(GenerationResponse(texts: .init(
            reasoning: reasoningText.isEmpty ? nil : reasoningText,
            response: responseText,
            notes: nil
          ), toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values)))
        }

      case StreamEventType.reasoningDelta, StreamEventType.reasoningSummaryDelta:
        if let delta = event.delta {
          reasoningText += delta
          continuation.yield(GenerationResponse(texts: .init(
            reasoning: reasoningText,
            response: responseText.isEmpty ? nil : responseText,
            notes: nil
          ), toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values)))
        }

      case StreamEventType.outputItemAdded:
        if let item = event.item, let itemType = item.type {
          switch itemType {
            case OutputItemType.functionCall:
              if let name = item.name, let callId = item.callId {
                streamingFunctionCalls[callId] = GenerationResponse.ToolCall(
                  name: name,
                  id: callId,
                  parameters: [:]
                )
                streamingArgumentsStrings[callId] = ""
                continuation.yield(GenerationResponse(texts: .init(
                  reasoning: reasoningText.isEmpty ? nil : reasoningText,
                  response: responseText.isEmpty ? nil : responseText,
                  notes: nil
                ), toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values)))
              }
            case OutputItemType.reasoning:
              if let summaryArray = item.summary {
                for summaryItem in summaryArray {
                  if let text = summaryItem.text {
                    reasoningText = reasoningText + text + "\n"
                  }
                }
                continuation.yield(GenerationResponse(texts: .init(
                  reasoning: reasoningText.isEmpty ? nil : reasoningText,
                  response: responseText.isEmpty ? nil : responseText,
                  notes: nil
                ), toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values)))
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
           let callId = event.itemId,
           var currentCall = streamingFunctionCalls[callId]
        {
          let existingArgsString = streamingArgumentsStrings[callId] ?? ""
          let newArgsString = existingArgsString + delta
          streamingArgumentsStrings[callId] = newArgsString

          if let argsData = newArgsString.data(using: .utf8),
             let partialArgs = try? JSONDecoder().decode([String: Value].self, from: argsData)
          {
            currentCall.parameters = partialArgs
            streamingFunctionCalls[callId] = currentCall
          }

          continuation.yield(GenerationResponse(texts: .init(
            reasoning: reasoningText.isEmpty ? nil : reasoningText,
            response: responseText.isEmpty ? nil : responseText,
            notes: nil
          ), toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values)))
        }

      case StreamEventType.functionCallArgumentsDone:
        if let argumentsString = event.arguments,
           let callId = event.itemId,
           var completedCall = streamingFunctionCalls.removeValue(forKey: callId)
        {
          streamingArgumentsStrings.removeValue(forKey: callId)

          if let argumentsData = argumentsString.data(using: .utf8),
             let parsedArguments = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
          {
            completedCall.parameters = parsedArguments
          } else {
            openAIResponsesLogger.error("Failed to parse final function call arguments for call ID \(callId): \(argumentsString)")
            completedCall.parameters = ["_parseError": .string("Failed to parse arguments JSON")]
          }
          completedFunctionCalls.append(completedCall)

          continuation.yield(GenerationResponse(texts: .init(
            reasoning: reasoningText.isEmpty ? nil : reasoningText,
            response: responseText.isEmpty ? nil : responseText,
            notes: nil
          ), toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values)))
        }

      case StreamEventType.completed:
        // Fallback: Extract thinking text from completion if it wasn't streamed
        if let outputArray = event.response?.output {
          for item in outputArray {
            if item.type == OutputItemType.reasoning, let summaryArray = item.summary {
              for summaryItem in summaryArray {
                if let text = summaryItem.text, reasoningText.isEmpty {
                  reasoningText = text
                }
              }
            }
          }
        }

        // Build and yield final response with metadata
        if let response = event.response {
          let generationResponse = response.toGenerationResponse()
          continuation.yield(GenerationResponse(
            texts: .init(
              reasoning: reasoningText.isEmpty ? nil : reasoningText,
              response: responseText.isEmpty ? generationResponse.texts.response : responseText,
              notes: nil
            ),
            toolCalls: completedFunctionCalls + Array(streamingFunctionCalls.values),
            metadata: generationResponse.metadata
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
    maxRetries: Int = 3
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
      isDirect: true
    )
  }

  private func streamBackgroundResponse(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    startingAfter: Int? = nil,
    retryCount: Int = 0,
    maxRetries: Int = 3
  ) async throws {
    openAIResponsesLogger.log("Background Stream: Starting for response \(responseId), attempt \(retryCount + 1)/\(maxRetries + 1), startingAfter: \(startingAfter ?? 0)")

    let streamUrl = endpoint.appendingPathComponent(responseId)
    var urlComponents = URLComponents(url: streamUrl, resolvingAgainstBaseURL: false)!
    urlComponents.queryItems = [
      URLQueryItem(name: "stream", value: "true"),
    ]
    if let startingAfter {
      urlComponents.queryItems?.append(URLQueryItem(name: "starting_after", value: String(startingAfter)))
    }

    var request = URLRequest(url: urlComponents.url!)
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
      isDirect: false
    )
  }

  // Shared streaming logic for both direct and resumption modes
  private func performBackgroundStream(
    request: URLRequest,
    responseId: String?,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    startingAfter: Int,
    retryCount: Int,
    maxRetries: Int,
    isDirect: Bool
  ) async throws {
    let logPrefix = isDirect ? "Background Stream Direct" : "Background Stream"
    var lastSequenceNumber: Int = startingAfter
    var currentResponseId: String? = responseId

    do {
      try await performSSEStream(
        request: request,
        continuation: continuation,
        logPrefix: logPrefix,
        isBackground: true,
        apiKey: apiKey,
        responseIdHandler: isDirect ? { id in
          guard !Task.isCancelled else { return }
          currentResponseId = id
          openAIResponsesLogger.log("\(logPrefix): Got response ID: \(id)")
          await MainActor.run { [weak self] in
            self?.activeBackgroundResponseId = id
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
        }
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
              maxRetries: maxRetries
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
              logPrefix: logPrefix
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
              maxRetries: maxRetries
            )
          } else {
            openAIResponsesLogger.log("\(logPrefix): Retrying from sequence \(lastSequenceNumber)...")
            try await streamBackgroundResponse(
              responseId: responseId!,
              apiKey: apiKey,
              continuation: continuation,
              startingAfter: lastSequenceNumber,
              retryCount: retryCount + 1,
              maxRetries: maxRetries
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

  // Shared SSE (Server-Sent Events) streaming logic for all stream types
  private func performSSEStream(
    request: URLRequest,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    logPrefix: String,
    isBackground _: Bool,
    apiKey _: String? = nil,
    responseIdHandler: ((String) async -> Void)? = nil,
    sequenceHandler: ((Int) -> Void)? = nil
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

    var reasoningText = ""
    var responseText = ""
    var streamingFunctionCalls: [String: GenerationResponse.ToolCall] = [:]
    var streamingArgumentsStrings: [String: String] = [:]
    var completedFunctionCalls: [GenerationResponse.ToolCall] = []

    for try await jsonString in SSEParser.dataPayloads(from: result) {
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
          reasoningText: &reasoningText,
          responseText: &responseText,
          streamingFunctionCalls: &streamingFunctionCalls,
          streamingArgumentsStrings: &streamingArgumentsStrings,
          completedFunctionCalls: &completedFunctionCalls,
          continuation: continuation
        )
      } catch let error as AIError {
        throw error
      } catch {
        openAIResponsesLogger.error("\(logPrefix) parsing error for JSON: \(jsonString). Error: \(error)")
        throw AIError.parsing(message: "Failed to parse streamed JSON: \(jsonString)")
      }
    }
  }

  // Helper method to check response status and handle completion/failure
  private func checkResponseStatusAndHandle(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    logPrefix: String
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

  // Helper method to parse completed response
  private func parseCompletedResponse(
    _ response: ResponseObject,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation
  ) {
    continuation.yield(response.toGenerationResponse())
  }

  private func pollBackgroundResponse(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation
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
      error: response.error?.message
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
        try handleHTTPError(httpResponse.statusCode, message: nil)
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
      try handleHTTPError(httpResponse.statusCode, message: nil)
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
    update: @Sendable @escaping (GenerationResponse) -> Void
  ) async throws -> GenerationResponse {
    await MainActor.run {
      isGenerating = true
      activeBackgroundResponseId = responseId
    }

    let task = Task<GenerationResponse, Error> {
      defer {
        Task { @MainActor in
          isGenerating = false
          currentTask = nil
          activeBackgroundResponseId = nil
        }
      }

      var finalReasoningText: String? = nil
      var finalResponseText: String? = nil
      var finalEndnotesText: String? = nil
      var finalFunctionCalls: [GenerationResponse.ToolCall] = []
      var finalMetadata: GenerationResponse.Metadata? = nil

      let stream = AsyncThrowingStream<GenerationResponse, Error> { continuation in
        Task {
          do {
            try await streamBackgroundResponse(
              responseId: responseId,
              apiKey: apiKey,
              continuation: continuation,
              startingAfter: startingAfter
            )
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
      }

      for try await chunk in stream {
        try Task.checkCancellation()

        finalReasoningText = chunk.texts.reasoning
        finalResponseText = chunk.texts.response
        finalEndnotesText = chunk.texts.notes
        finalFunctionCalls = chunk.toolCalls
        finalMetadata = chunk.metadata

        let reasoningCopy = finalReasoningText
        let responseCopy = finalResponseText
        let notesCopy = finalEndnotesText
        let toolCallsCopy = finalFunctionCalls

        await MainActor.run {
          update(.init(texts: .init(
            reasoning: reasoningCopy,
            response: responseCopy,
            notes: notesCopy
          ), toolCalls: toolCallsCopy))
        }
      }

      return .init(texts: .init(
        reasoning: finalReasoningText,
        response: finalResponseText,
        notes: finalEndnotesText
      ), toolCalls: finalFunctionCalls.filter { !$0.parameters.keys.contains("_parseError") }, metadata: finalMetadata)
    }

    await MainActor.run {
      currentTask = task
    }

    return try await task.value
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
        result["required"] = propertyNames.sorted() // Sort for consistency
      }
    }

    return result
  }
}

extension ResponsesClient {
  /// Reasoning effort level for models that support extended thinking.
  public enum ReasoningEffortLevel: String, CaseIterable, Identifiable, Sendable {
    /// The default reasoning effort level.
    public static let `default`: ReasoningEffortLevel = .medium

    /// Minimal reasoning effort for simple tasks.
    case minimal
    /// Low reasoning effort for straightforward tasks.
    case low
    /// Medium reasoning effort for balanced performance.
    case medium
    /// High reasoning effort for complex tasks.
    case high

    /// The raw value identifier.
    public var id: String { rawValue }
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
    public var id: String { rawValue }
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
    public var id: String { rawValue }
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

    /// A configuration with all features disabled.
    public static let disabled = Configuration()

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
      backgroundMode: Bool = false
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
    let arguments: String?
    let item: OutputItem?
    let response: ResponseObject?
    let error: ErrorObject?

    enum CodingKeys: String, CodingKey {
      case type
      case sequenceNumber = "sequence_number"
      case delta
      case itemId = "item_id"
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
      var responseText: String? = outputText
      var reasoningText: String? = nil
      var toolCalls: [GenerationResponse.ToolCall] = []

      if let outputArray = output {
        for item in outputArray {
          guard let itemType = item.type else { continue }

          switch itemType {
            case OutputItemType.message:
              if let contentArray = item.content {
                for contentItem in contentArray {
                  if contentItem.type == OutputItemType.outputText, let text = contentItem.text {
                    if responseText == nil { responseText = "" }
                    responseText! += text
                  }
                }
              }
            case OutputItemType.reasoning:
              if let summaryArray = item.summary {
                for summaryItem in summaryArray {
                  if let text = summaryItem.text {
                    reasoningText = (reasoningText ?? "") + text + "\n"
                  }
                }
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
                toolCalls.append(GenerationResponse.ToolCall(
                  name: name,
                  id: callId,
                  parameters: parameters
                ))
              }
            default:
              break
          }
        }
      }

      reasoningText = reasoningText?.trimmingCharacters(in: .whitespacesAndNewlines)

      // Build metadata from response
      let finishReason: GenerationResponse.FinishReason? = if let status {
        switch status {
          case "completed":
            // If there are function calls, the finish reason is tool use
            toolCalls.isEmpty ? .stop : .toolUse
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

      var createdAtDate: Date? = nil
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
        reasoningTokens: usage?.outputTokensDetails?.reasoningTokens
      )

      return GenerationResponse(
        texts: .init(reasoning: reasoningText, response: responseText, notes: nil),
        toolCalls: toolCalls,
        metadata: metadata
      )
    }
  }

  struct ResponseOutputItem: Decodable {
    let type: String?
    let content: [ContentItem]?
    let name: String?
    let callId: String?
    let arguments: String?
    let summary: [SummaryItem]?

    enum CodingKeys: String, CodingKey {
      case type, content, name, arguments, summary
      case callId = "call_id"
    }
  }

  struct ContentItem: Decodable {
    let type: String?
    let text: String?
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
}

// Response format options for the responses endpoint
enum ResponseFormat {
  case text
  case jsonObject
  case jsonSchema(schema: [String: any Sendable], name: String? = nil, description: String? = nil)
}

private let openAIResponsesLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "ResponsesClient")
