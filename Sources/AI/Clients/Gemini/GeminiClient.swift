// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log

/// A client for the Google Gemini API.
///
/// Supports Gemini models with features like tool use, streaming, multimodal inputs,
/// and thinking mode for reasoning models.
///
/// ## Example
///
/// ```swift
/// let client = GeminiClient()
/// let response = try await client.generateText(
///   modelId: "gemini-2.0-flash",
///   prompt: "Hello, Gemini!",
///   apiKey: "your-api-key"
/// )
/// print(response.content)
/// ```
@Observable
public final class GeminiClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image, .audio, .file]

  private static let defaultModelsEndpoint = URL(string: "https://generativelanguage.googleapis.com/v1beta/models")!

  private let modelsEndpoint: URL

  /// URLSession with no timeout. Gemini thinking requests can go over a minute between
  /// streaming updates, and URLSession's `timeoutIntervalForRequest` applies between every
  /// data packet (not just the initial connection), so a finite value would kill the stream.
  /// Callers that need a timeout can pass their own URLSession.
  public static let defaultSession: URLSession = {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = .infinity
    config.timeoutIntervalForResource = .infinity
    return URLSession(configuration: config)
  }()

  private let session: URLSession
  let retryHandler: RetryHandler

  struct GeminiError: LocalizedError {
    let message: String
    let response: GenerateContentResponse?

    var errorDescription: String? {
      message
    }

    init(message: String, response: GenerateContentResponse? = nil) {
      self.message = message
      self.response = response
    }
  }

  struct GenerateContentResponse: Codable {
    var candidates: [Candidate]?
    let promptFeedback: PromptFeedback?
    var usageMetadata: UsageMetadata?
  }

  struct Candidate: Codable {
    var content: Content?
    var finishReason: FinishReason?
    var safetyRatings: [SafetyRating]?
    var citationMetadata: CitationMetadata?
    let tokenCount: Int?
    let avgLogprobs: Double?
    let index: Int?
    var groundingMetadata: GroundingMetadata?
  }

  struct Content: Codable {
    var parts: [Part]?
    let role: String?
  }

  struct Part: Codable {
    let text: String?
  }

  enum FinishReason: String, Codable {
    case stop = "STOP"
    case maxTokens = "MAX_TOKENS"
    case safety = "SAFETY"
    case recitation = "RECITATION"
    case language = "LANGUAGE"
    case blocklist = "BLOCKLIST"
    case prohibitedContent = "PROHIBITED_CONTENT"
    case spii = "SPII"
    case malformedFunctionCall = "MALFORMED_FUNCTION_CALL"
    case imageSafety = "IMAGE_SAFETY"
    case unexpectedToolCall = "UNEXPECTED_TOOL_CALL"
    case imageProhibitedContent = "IMAGE_PROHIBITED_CONTENT"
    case imageRecitation = "IMAGE_RECITATION"
    case imageOther = "IMAGE_OTHER"
    case noImage = "NO_IMAGE"
    case other = "OTHER"
    case unspecified = "FINISH_REASON_UNSPECIFIED"
  }

  struct SafetyRating: Codable {
    let category: String
    let probability: String
  }

  struct CitationMetadata: Codable {
    // Add relevant fields
  }

  struct PromptFeedback: Codable {
    // Add relevant fields
  }

  struct UsageMetadata: Codable {
    let promptTokenCount: Int?
    let candidatesTokenCount: Int?
    let totalTokenCount: Int?
    let cachedContentTokenCount: Int?
    let thoughtsTokenCount: Int?
  }

  /// Code execution
  struct ExecutableCode: Codable {
    let code: String?
    let language: String?
  }

  struct CodeExecutionResult: Codable {
    let outcome: String?
    let output: String?
  }

  /// Grounding metadata structures
  struct GroundingMetadata: Codable {
    let webSearchQueries: [String]?
    let groundingChunks: [GroundingChunk]?
    let groundingSupports: [GroundingSupport]?
    let searchEntryPoint: SearchEntryPoint?
  }

  struct SearchEntryPoint: Codable {
    let renderedContent: String?
  }

  struct GroundingSupport: Codable {
    let segment: Segment?
    let groundingChunkIndices: [Int]?
    let confidenceScores: [Double]?
  }

  struct Segment: Codable {
    let startIndex: Int?
    let endIndex: Int?
    let text: String?
  }

  struct GroundingChunk: Codable {
    let web: WebSource?
  }

  struct WebSource: Codable {
    let uri: String
    let title: String
  }

  @MainActor public private(set) var isGenerating: Bool = false

  @MainActor private var currentTask: Task<GenerationResponse, Error>?

  /// Creates a new Gemini client.
  ///
  /// - Parameters:
  ///   - maxRetries: Maximum number of retry attempts for failed requests.
  ///   - session: URLSession to use for requests. Defaults to a session with no timeout.
  ///   - modelsEndpoint: Custom endpoint URL for the models API.
  public init(maxRetries: Int = 4, session: URLSession = GeminiClient.defaultSession, modelsEndpoint: URL? = nil) {
    retryHandler = RetryHandler(maxRetries: maxRetries)
    self.session = session
    self.modelsEndpoint = modelsEndpoint ?? GeminiClient.defaultModelsEndpoint
  }

  private static func assistantContent(
    reasoningText: String? = nil,
    reasoningSignature: String? = nil,
    responseText: String? = nil,
    notesText: String? = nil,
    toolCalls: [ToolCall] = [],
  ) -> [Message.Content] {
    // When a Gemini thought signature is present, store it as a provider-scoped opaque block
    // rather than in the generic .thinking(signature:) slot, which Anthropic treats as its own.
    if let reasoningText, !reasoningText.isEmpty, let reasoningSignature {
      var content: [Message.Content] = [
        .providerOpaque(OpaqueBlock(provider: "gemini", type: "thinking", content: reasoningText, signature: reasoningSignature)),
      ]
      if let responseText, !responseText.isEmpty {
        content.append(.text(responseText))
      }
      if let notesText, !notesText.isEmpty {
        content.append(.endnotes(notesText))
      }
      content.append(contentsOf: toolCalls.map(Message.Content.toolCall))
      return content
    }
    return Message.assistantContent(reasoningText: reasoningText, responseText: responseText, notesText: notesText, toolCalls: toolCalls)
  }

  static func finalizedContent(
    orderedContent: [Message.Content],
    notesText: String?,
    metadataOpaqueBlocks: [OpaqueBlock],
  ) -> [Message.Content] {
    var content = orderedContent
    if let notesText, !notesText.isEmpty {
      content.append(.endnotes(notesText))
    }
    content.append(contentsOf: metadataOpaqueBlocks.map(Message.Content.providerOpaque))
    return content
  }

  static func systemInstructionParts(for message: Message) -> [[String: any Sendable]] {
    message.replayableTextSegmentsWithAttachmentFallback().map { ["text": $0] }
  }

  func requestParts(for message: Message, apiKey: String) async throws -> [[String: any Sendable]] {
    var parts: [[String: any Sendable]] = []

    for block in message.content {
      switch block {
        case let .toolCall(toolCall):
          var nativeArgs: [String: any Sendable] = [:]
          for (key, value) in toolCall.parameters {
            nativeArgs[key] = value.toAny()
          }
          var toolCallDict: [String: any Sendable] = [
            "name": toolCall.name,
            "args": nativeArgs,
          ]
          if !toolCall.id.isEmpty {
            toolCallDict["id"] = toolCall.id
          }
          var partDict: [String: any Sendable] = [
            "functionCall": toolCallDict,
          ]
          if let thoughtSignature = toolCall.providerMetadata?["thoughtSignature"] {
            partDict["thoughtSignature"] = thoughtSignature
          }
          parts.append(partDict)

        case let .toolResult(toolResult):
          var functionResponse: [String: any Sendable] = [
            "name": toolResult.name,
            "id": toolResult.id,
          ]

          if toolResult.isError == true {
            let errorText = toolResult.content.compactMap { content -> String? in
              if case let .text(text) = content { return text }
              return nil
            }.joined(separator: "\n")
            functionResponse["response"] = ["error": errorText.isEmpty ? "Unknown error" : errorText] as [String: any Sendable]
          } else {
            var inlineDataParts: [[String: any Sendable]] = []
            var textOutputs: [String] = []

            for content in toolResult.content {
              switch content {
                case let .text(text):
                  textOutputs.append(text)
                case let .image(data, mimeType):
                  let mediaType = mimeType ?? "image/png"
                  inlineDataParts.append([
                    "inlineData": [
                      "mimeType": mediaType,
                      "data": data.base64EncodedString(),
                    ] as [String: any Sendable],
                  ])
                case let .audio(data, mimeType):
                  inlineDataParts.append([
                    "inlineData": [
                      "mimeType": mimeType,
                      "data": data.base64EncodedString(),
                    ] as [String: any Sendable],
                  ])
                case let .file(data, mimeType, _):
                  inlineDataParts.append([
                    "inlineData": [
                      "mimeType": mimeType,
                      "data": data.base64EncodedString(),
                    ] as [String: any Sendable],
                  ])
              }
            }

            let joinedText = textOutputs.joined(separator: "\n")
            if !joinedText.isEmpty {
              functionResponse["response"] = ["output": joinedText] as [String: any Sendable]
            } else {
              functionResponse["response"] = [:] as [String: any Sendable]
            }
            if !inlineDataParts.isEmpty {
              functionResponse["parts"] = inlineDataParts
            }
          }

          parts.append(["functionResponse": functionResponse])

        case let .attachment(attachment):
          switch attachment.kind {
            case let .image(data, mimeType):
              let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
              parts.append([
                "inlineData": [
                  "mimeType": processedMimeType,
                  "data": processedImageData.base64EncodedString(),
                ],
              ])
            case let .video(data, mimeType):
              let fileUri = try await uploadFile(
                data: data,
                mimeType: mimeType,
                displayName: attachment.filename ?? "Video",
                apiKey: apiKey,
              )
              parts.append([
                "fileData": [
                  "mimeType": mimeType,
                  "fileUri": fileUri,
                ],
              ])
            case let .audio(data, mimeType):
              let fileUri = try await uploadFile(
                data: data,
                mimeType: mimeType,
                displayName: attachment.filename ?? "Audio",
                apiKey: apiKey,
              )
              parts.append([
                "fileData": [
                  "mimeType": mimeType,
                  "fileUri": fileUri,
                ],
              ])
            case let .document(data, mimeType):
              let mimeTypeForGemini = switch mimeType {
                case "net.daringfireball.markdown", "text/x-markdown": "text/md"
                default: mimeType
              }
              if data.count < 20_000_000 {
                parts.append([
                  "inlineData": [
                    "mimeType": mimeTypeForGemini,
                    "data": data.base64EncodedString(),
                  ],
                ])
              } else {
                let fileUri = try await uploadFile(
                  data: data,
                  mimeType: mimeTypeForGemini,
                  displayName: attachment.filename ?? "Document",
                  apiKey: apiKey,
                )
                parts.append([
                  "fileData": [
                    "mimeType": mimeTypeForGemini,
                    "fileUri": fileUri,
                  ],
                ])
              }
          }

        case let .text(text) where !text.isEmpty:
          parts.append(["text": text])

        case let .thinking(text, _):
          parts.append([
            "text": text,
            "thought": true,
          ])

        case let .endnotes(text) where !text.isEmpty:
          parts.append(["text": text])

        case let .providerOpaque(opaque) where opaque.isGeminiThinking:
          var part: [String: any Sendable] = [
            "text": opaque.content ?? "",
            "thought": true,
          ]
          if let signature = opaque.signature {
            part["thoughtSignature"] = signature
          }
          parts.append(part)

        case let .providerOpaque(opaque) where opaque.isGeminiRoundTrippablePart:
          if let jsonObject = Self.geminiJSONObject(from: opaque) {
            parts.append([opaque.type: jsonObject])
          } else if let text = opaque.replayDowngradeText(for: .gemini) {
            // If a manually constructed Gemini opaque block is missing raw JSON,
            // preserve its visible output rather than dropping the history item.
            parts.append(["text": text])
          }

        case let .providerOpaque(opaque) where opaque.isGeminiURLContextMetadata:
          // urlContextMetadata is candidate-level output metadata, not a request Part.
          break

        case let .providerOpaque(opaque):
          // Non-Gemini providers store some visible output only in opaque blocks.
          // Downgrade that text so provider switches preserve the assistant transcript.
          if let text = opaque.replayDowngradeText(for: .gemini) {
            parts.append(["text": text])
          }

        default:
          break
      }
    }

    return parts
  }

  /// Sends a request to the Gemini API.
  ///
  /// When `streaming` is true, uses `:streamGenerateContent?alt=sse` and returns
  /// incremental SSE chunks. When false, uses `:generateContent` and returns the
  /// complete response as a single-element stream, matching the Gemini TS SDK's
  /// endpoint split.
  private func streamResponse(
    messages: [Message],
    systemPrompt: String?,
    modelId: String,
    apiKey: String,
    maxTokens: Int?,
    temperature: Float?,
    configuration: Configuration,
    tools: [Tool] = [],
    streaming: Bool = true,
  ) async throws -> AsyncThrowingStream<GeminiStreamChunk, Error> {
    let request = try await GeminiRequestEncoder.makeRequest(
      modelsEndpoint: modelsEndpoint,
      modelId: modelId,
      apiKey: apiKey,
      messages: messages,
      systemPrompt: systemPrompt,
      maxTokens: maxTokens,
      temperature: temperature,
      configuration: configuration,
      tools: tools,
      streaming: streaming,
      requestParts: { [self] message, apiKey in
        try await requestParts(for: message, apiKey: apiKey)
      },
    )

    var retriesRemaining = retryHandler.maxRetries
    while true {
      do {
        if streaming {
          let (result, response) = try await session.bytes(for: request)
          // Consume the stream to check for HTTP errors before returning
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          if !(200 ... 299).contains(httpResponse.statusCode) {
            var errorData = Data()
            for try await byte in result {
              try Task.checkCancellation()
              errorData.append(byte)
            }
            let errorMessage = GeminiStreamTransport.parseGeminiErrorMessage(from: errorData)
            throw Self.geminiHTTPError(
              statusCode: httpResponse.statusCode,
              message: errorMessage,
              retryAfter: AIError.parseRetryAfter(from: httpResponse),
            )
          }
          return GeminiStreamTransport.processStreamBytes(result: result, response: response)
        } else {
          let (data, response) = try await session.data(for: request)
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          if !(200 ... 299).contains(httpResponse.statusCode) {
            let errorMessage = GeminiStreamTransport.parseGeminiErrorMessage(from: data)
            throw Self.geminiHTTPError(
              statusCode: httpResponse.statusCode,
              message: errorMessage,
              retryAfter: AIError.parseRetryAfter(from: httpResponse),
            )
          }
          return GeminiStreamTransport.processBufferedResponse(data: data, response: response)
        }
      } catch {
        let aiError = (error as? AIError) ?? .network(underlying: error)
        if retriesRemaining > 0, retryHandler.shouldRetry(aiError) {
          let delay = retryHandler.retryDelay(retriesRemaining: retriesRemaining)
          try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
          retriesRemaining -= 1
          continue
        }
        throw aiError
      }
    }
  }

  /// Extract grounding information from metadata
  private func formatGroundingInfo(from metadata: GroundingMetadata) async -> String? {
    var notes = [String]()
    // Add sources
    if let chunks = metadata.groundingChunks, !chunks.isEmpty {
      struct SourceReference {
        let index: Int
        let title: String
        let url: String
      }
      var resolvedSources = [SourceReference]() // Use the new struct
      // Resolve all URLs in parallel
      await withTaskGroup(of: SourceReference.self) { group in
        for (index, chunk) in chunks.enumerated() {
          if let webSource = chunk.web {
            group.addTask {
              let resolvedURL = await self.resolveRedirectURL(webSource.uri)
              return SourceReference(
                index: index,
                title: webSource.title,
                url: resolvedURL,
              )
            }
          }
        }
        // Collect results
        for await result in group {
          resolvedSources.append(result)
        }
      }
      // Sort by original index and format
      resolvedSources.sorted { $0.index < $1.index }.forEach { source in
        if source.url.starts(with: "https://vertexaisearch.cloud.google.com") {
          notes.append("- [\(source.title)](\(source.url))")
        } else {
          notes.append("- \(source.url)")
        }
      }
    }

    return notes.isEmpty ? nil : notes.joined(separator: "\n")
  }

  /// Generates a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Gemini model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature (0.0-2.0).
  ///   - apiKey: Gemini API key.
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
      configuration: configuration,
      update: { _ in },
    )
  }

  /// Streams a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Gemini model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature (0.0-2.0).
  ///   - apiKey: Gemini API key.
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
        let finalResponse = try await _generate(
          modelId: modelId,
          tools: tools,
          systemPrompt: systemPrompt,
          messages: messages,
          maxTokens: maxTokens,
          temperature: temperature,
          apiKey: apiKey,
          configuration: configuration,
          streaming: true,
          update: { response in
            continuation.yield(response)
          },
        )
        continuation.yield(finalResponse)
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
    configuration: Configuration,
    streaming: Bool = false,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    guard let apiKey else {
      throw AIError.authentication(message: "Missing API key")
    }
    await MainActor.run {
      isGenerating = true
    }
    let taskHolder = OSAllocatedUnfairLock(initialState: Task<GenerationResponse, Error>?.none)

    do {
      let result = try await withTaskCancellationHandler {
        try Task.checkCancellation()

        let task = Task<GenerationResponse, Error> {
          var assembler = GeminiResponseAssembler()

          do {
            let stream = try await streamResponse(
              messages: messages,
              systemPrompt: systemPrompt,
              modelId: modelId,
              apiKey: apiKey,
              maxTokens: maxTokens,
              temperature: temperature,
              configuration: configuration,
              tools: tools,
              streaming: streaming,
            )

            let sendUpdate = {
              let response = assembler.response()
              await MainActor.run {
                update(response)
              }
            }

            for try await chunk in stream {
              try Task.checkCancellation()

              if await assembler.consume(
                chunk,
                formatGroundingInfo: { [self] metadata in
                  await formatGroundingInfo(from: metadata)
                },
              ) {
                await sendUpdate()
              }
            }

            // Yield final state with complete metadata (usage and finish reason may arrive
            // in chunks that don't contain content, so the last content-triggered update
            // may lack them)
            await sendUpdate()

            return assembler.response()
          } catch let error as GeminiError {
            // Check if the task was cancelled
            if Task.isCancelled {
              return assembler.partialResponse()
            }

            geminiLogger.warning("Gemini error: \(error.message)")
            let errorMessage = if error.message.contains("SAFETY") {
              "Response blocked due to safety filters"
            } else if error.message.contains("RECITATION") {
              "Response blocked due to content recitation"
            } else {
              error.message
            }
            throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
          } catch {
            // Handle cancellation
            if error is CancellationError || Task.isCancelled {
              return assembler.partialResponse()
            } else {
              throw error
            }
          }
        }

        taskHolder.withLock { $0 = task }
        if Task.isCancelled {
          task.cancel()
        }
        await MainActor.run {
          currentTask = task
        }
        return await task.result
      } onCancel: {
        taskHolder.withLock { $0?.cancel() }
      }

      taskHolder.withLock { $0 = nil }
      await cleanUpGeneration()
      return try result.get()
    } catch {
      taskHolder.withLock { $0 = nil }
      await cleanUpGeneration()
      throw error
    }
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

  private func queryItemsPreservingCustomParameters(
    _ queryItems: [URLQueryItem]?,
    apiKey: String,
  ) -> [URLQueryItem] {
    var preservedQueryItems = queryItems ?? []
    preservedQueryItems.removeAll { $0.name == "key" }
    preservedQueryItems.append(URLQueryItem(name: "key", value: apiKey))
    return preservedQueryItems
  }

  private func uploadFile(data: Data, mimeType: String, displayName: String, apiKey: String) async throws -> String {
    // Derive upload URL from the configured models endpoint so custom endpoints (proxies, mocks) work.
    // modelsEndpoint path is e.g. "/v1beta/models" or "/prefix/v1beta/models";
    // replace the version+models suffix with the upload path, preserving any proxy prefix.
    var uploadComponents = URLComponents(url: modelsEndpoint, resolvingAgainstBaseURL: true)!
    let path = uploadComponents.path
    if let range = path.range(of: "/v1beta/models", options: .backwards) {
      uploadComponents.path = String(path[..<range.lowerBound]) + "/upload/v1beta/files"
    } else {
      uploadComponents.path = "/upload/v1beta/files"
    }
    uploadComponents.queryItems = queryItemsPreservingCustomParameters(uploadComponents.queryItems, apiKey: apiKey)
    let uploadURL = uploadComponents.url!
    // Start resumable upload
    var request = URLRequest(url: uploadURL)
    request.httpMethod = "POST"
    request.setValue("resumable", forHTTPHeaderField: "X-Goog-Upload-Protocol")
    request.setValue("start", forHTTPHeaderField: "X-Goog-Upload-Command")
    request.setValue("\(data.count)", forHTTPHeaderField: "X-Goog-Upload-Header-Content-Length")
    request.setValue(mimeType, forHTTPHeaderField: "X-Goog-Upload-Header-Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    let metadata = ["file": ["displayName": displayName]]
    request.httpBody = try JSONSerialization.data(withJSONObject: metadata)
    let (responseData, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      let errorMessage = String(data: responseData, encoding: .utf8) ?? "Unknown error"
      throw AIError.fromHTTPStatusCode(httpResponse.statusCode, message: errorMessage)
    }
    guard let uploadUrl = httpResponse.value(forHTTPHeaderField: "X-Goog-Upload-URL") else {
      throw AIError.parsing(message: "Failed to get upload URL from response headers")
    }
    // Upload the actual file
    guard let uploadURL = URL(string: uploadUrl) else {
      throw AIError.parsing(message: "Invalid upload URL from server: \(uploadUrl)")
    }
    var uploadRequest = URLRequest(url: uploadURL)
    uploadRequest.httpMethod = "POST"
    uploadRequest.setValue("\(data.count)", forHTTPHeaderField: "Content-Length")
    uploadRequest.setValue("0", forHTTPHeaderField: "X-Goog-Upload-Offset")
    uploadRequest.setValue("upload, finalize", forHTTPHeaderField: "X-Goog-Upload-Command")
    uploadRequest.httpBody = data
    let (uploadResponseData, uploadResponse) = try await session.data(for: uploadRequest)
    if let uploadHttpResponse = uploadResponse as? HTTPURLResponse, !(200 ... 299).contains(uploadHttpResponse.statusCode) {
      let errorMessage = String(data: uploadResponseData, encoding: .utf8) ?? "Unknown error"
      throw AIError.fromHTTPStatusCode(uploadHttpResponse.statusCode, message: errorMessage)
    }
    do {
      struct UploadErrorInfo: Codable {
        let code: Int?
        let message: String?
      }

      // Response structure for the upload
      struct FileResponse: Codable {
        struct File: Codable {
          let uri: String?
          let state: String?
          let error: UploadErrorInfo?
        }

        let file: File
      }

      // Status check response structure
      struct StatusResponse: Codable {
        let uri: String?
        let state: String?
        let error: UploadErrorInfo?
      }

      func failedUploadError(message: String? = nil) -> AIError {
        AIError.serverError(
          statusCode: 0,
          message: message ?? "File processing failed with unknown error",
          context: nil,
        )
      }

      func resolvedState(_ state: String?, error: UploadErrorInfo?) -> String? {
        if let state, !state.isEmpty {
          return state
        }
        return error == nil ? nil : "FAILED"
      }

      let fileResponse = try JSONDecoder().decode(FileResponse.self, from: uploadResponseData)
      let initialState = resolvedState(fileResponse.file.state, error: fileResponse.file.error)
      if initialState == "FAILED" {
        throw failedUploadError(message: fileResponse.file.error?.message)
      }
      guard let fileUri = fileResponse.file.uri, !fileUri.isEmpty else {
        throw AIError.parsing(message: "Upload response omitted file URI")
      }
      guard let initialState else {
        throw AIError.parsing(message: "Upload response omitted file state")
      }
      var fileState = initialState
      // Wait for video processing to complete
      while fileState == "PROCESSING" {
        try Task.checkCancellation()
        try await Task.sleep(for: .seconds(2))
        // Use the full URI from the response
        guard let checkURL = URL(string: fileUri),
              var checkComponents = URLComponents(url: checkURL, resolvingAgainstBaseURL: true)
        else {
          throw AIError.parsing(message: "Invalid file URI from server: \(fileUri)")
        }
        checkComponents.queryItems = queryItemsPreservingCustomParameters(checkComponents.queryItems, apiKey: apiKey)
        guard let checkRequestURL = checkComponents.url else {
          throw AIError.parsing(message: "Failed to construct status check URL for file: \(fileUri)")
        }
        let statusRequest = URLRequest(url: checkRequestURL)
        let (checkData, checkResponse) = try await session.data(for: statusRequest)
        guard let checkHTTPResponse = checkResponse as? HTTPURLResponse else {
          throw AIError.network(underlying: URLError(.badServerResponse))
        }
        if !(200 ... 299).contains(checkHTTPResponse.statusCode) {
          let errorMessage = GeminiStreamTransport.parseGeminiErrorMessage(from: checkData)
          throw Self.geminiHTTPError(
            statusCode: checkHTTPResponse.statusCode,
            message: errorMessage,
            retryAfter: AIError.parseRetryAfter(from: checkHTTPResponse),
          )
        }
        // Decode the status check response
        let statusResponse = try JSONDecoder().decode(StatusResponse.self, from: checkData)
        let statusState = resolvedState(statusResponse.state, error: statusResponse.error)
        // Check for processing failure
        if statusState == "FAILED" {
          throw failedUploadError(message: statusResponse.error?.message)
        }
        guard let statusState else {
          throw AIError.parsing(message: "Status response omitted file state")
        }
        fileState = statusState
      }
      guard fileState == "ACTIVE" else {
        throw AIError.parsing(message: "Unexpected file state: \(fileState)")
      }
      // Return the complete URI for use with the Gemini API
      return fileUri
    } catch let error as AIError {
      throw error
    } catch {
      geminiLogger.error("Decoding error: \(error)")
      // Try to decode as an error response format

      struct ErrorResponse: Codable {
        let error: ErrorDetail

        struct ErrorDetail: Codable {
          let code: Int
          let message: String
          let status: String
        }
      }
      if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: uploadResponseData) {
        throw AIError.serverError(statusCode: errorResponse.error.code, message: errorResponse.error.message, context: nil)
      }
      throw AIError.parsing(message: "Failed to decode response: \(error.localizedDescription)")
    }
  }

  /// Used to get actual URL in search results references instead of Google tracking URL
  private func resolveRedirectURL(_ url: String) async -> String {
    guard let originalURL = URL(string: url) else { return url }
    // Create a HEAD request to follow redirects without downloading content
    var request = URLRequest(url: originalURL)
    request.httpMethod = "HEAD"
    request.timeoutInterval = 5
    do {
      let (_, response) = try await session.data(for: request)
      if let httpResponse = response as? HTTPURLResponse, let finalURL = httpResponse.url {
        return finalURL.absoluteString
      }
    } catch {
      // In the case of ATS errors on http-only URLs, try to extract the blocked URL from the error
      if let urlError = error as? URLError, case .appTransportSecurityRequiresSecureConnection = urlError.code {
        if let blockedURL = extractURLFromATSError(error) {
          return blockedURL
        } else {
          geminiLogger.warning("Could not extract URL from ATS error")
        }
      } else {
        geminiLogger.warning("Failed to resolve redirect URL '\(originalURL.absoluteString)' for reference in response: \(error.localizedDescription)")
      }
    }
    // Return the original URL if resolution fails
    return url
  }

  /// Extract the blocked URL from ATS error
  private func extractURLFromATSError(_ error: Error) -> String? {
    let nsError = error as NSError
    // Check the error's userInfo dictionary for URL information
    if let failingURL = nsError.userInfo[NSURLErrorFailingURLErrorKey] as? URL {
      return failingURL.absoluteString
    }
    // For URLError, check the failureURLString property
    if let urlError = error as? URLError {
      if let failureURLString = urlError.failingURL?.absoluteString {
        return failureURLString
      }
    }
    return nil
  }
}

extension GeminiClient {
  /// Maps an HTTP status code and optional error message to an `AIError`.
  static func geminiHTTPError(statusCode: Int, message: String?, retryAfter: TimeInterval? = nil) -> AIError {
    let providerInfo = AIError.ProviderErrorInfo.gemini(status: "\(statusCode)", message: message)
    let context = AIError.ErrorContext(providerInfo: providerInfo)
    return switch statusCode {
      case 400: .invalidRequest(message: message ?? "There was a problem with the request body.")
      case 403: .authentication(message: "Ensure your API key is set correctly and has the right access.")
      case 404: .invalidRequest(message: message.map { "Not found: \($0)" } ?? "The requested resource wasn't found.")
      case 429: .rateLimit(retryAfter: retryAfter)
      case 500: .serverError(statusCode: 500, message: message ?? "An unexpected error occurred. Try reducing your input context, switching to another model temporarily, or retry after a short wait.", context: context)
      case 503: .serverError(statusCode: 503, message: message ?? "The service may be temporarily overloaded. Try switching to another model temporarily or retry after a short wait.", context: context)
      case 504: .timeout
      default: .serverError(statusCode: statusCode, message: message ?? "HTTP error \(statusCode)", context: context)
    }
  }
}

private let geminiLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "GeminiClient")

// MARK: - Configuration

public extension GeminiClient {
  /// Configuration options for Gemini API requests.
  struct Configuration: Sendable {
    /// Content safety filtering threshold.
    public var safetyThreshold: SafetyThreshold

    /// Enables Google Search grounding for factual responses.
    public var searchGrounding: Bool

    /// Enables web content fetching for retrieving page content from URLs.
    public var webContent: Bool

    /// Enables code execution in a sandboxed Python environment.
    public var codeExecution: Bool

    /// Token budget for extended thinking (Gemini 2.5 models).
    /// Use `thinkingLevel` for Gemini 3 models instead.
    public var thinkingBudget: Int?

    /// Thinking level for reasoning depth (Gemini 3 models).
    /// Use `thinkingBudget` for Gemini 2.5 models instead.
    public var thinkingLevel: ThinkingLevel?

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - safetyThreshold: Content safety filtering level.
    ///   - searchGrounding: Enable Google Search grounding.
    ///   - webContent: Enable web content fetching.
    ///   - codeExecution: Enable sandboxed code execution.
    ///   - thinkingBudget: Token budget for thinking (Gemini 2.5).
    ///   - thinkingLevel: Thinking level (Gemini 3).
    public init(
      safetyThreshold: SafetyThreshold = .none,
      searchGrounding: Bool = false,
      webContent: Bool = false,
      codeExecution: Bool = false,
      thinkingBudget: Int? = nil,
      thinkingLevel: ThinkingLevel? = nil,
    ) {
      self.safetyThreshold = safetyThreshold
      self.searchGrounding = searchGrounding
      self.webContent = webContent
      self.codeExecution = codeExecution
      self.thinkingBudget = thinkingBudget
      self.thinkingLevel = thinkingLevel
    }
  }

  /// Thinking level for Gemini 3 models.
  enum ThinkingLevel: String, CaseIterable, Identifiable, Codable, Equatable, Hashable, Sendable {
    /// Matches "no thinking" for most queries. Flash-only.
    case minimal
    /// Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput applications.
    case low
    /// Balanced thinking for most tasks. Flash-only.
    case medium
    /// Maximizes reasoning depth. Supported by both Pro and Flash.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Content safety filtering threshold levels.
  enum SafetyThreshold: String, CaseIterable, Identifiable, Sendable {
    /// No content filtering.
    case none = "BLOCK_NONE"
    /// Block only high-probability harmful content.
    case high = "BLOCK_ONLY_HIGH"
    /// Block medium and high probability harmful content.
    case medium = "BLOCK_MEDIUM_AND_ABOVE"
    /// Block low, medium, and high probability harmful content.
    case low = "BLOCK_LOW_AND_ABOVE"

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Returns the appropriate thinking configuration for a Gemini model.
  /// Defaults to `thinkingLevel` (Gemini 3+ paradigm). Uses `thinkingBudget` for
  /// Gemini 2.5 models, and disables thinking for older models.
  static func thinkingConfig(
    for modelId: String,
    reasoning: Bool,
  ) -> (thinkingLevel: ThinkingLevel?, thinkingBudget: Int?) {
    guard reasoning, modelId.hasPrefix("gemini-") else {
      return (nil, nil)
    }
    if modelId.hasPrefix("gemini-2.0") || modelId.hasPrefix("gemini-1") {
      return (nil, nil)
    }
    if modelId.hasPrefix("gemini-2.5") {
      return (nil, -1)
    }
    return (.high, nil)
  }
}
