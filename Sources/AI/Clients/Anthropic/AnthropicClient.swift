// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log

/// A client for the Anthropic Messages API.
///
/// Supports Claude models with features like tool use, streaming, prompt caching,
/// extended thinking, and web search (via computer use beta).
///
/// ## Example
///
/// ```swift
/// let client = AnthropicClient()
/// let response = try await client.generateText(
///   modelId: "claude-sonnet-4-20250514",
///   prompt: "Hello, Claude!",
///   apiKey: "your-api-key"
/// )
/// print(response.content)
/// ```
@Observable
public final class AnthropicClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image, .file]

  private let baseURL = URL(string: "https://api.anthropic.com/v1")!
  let messagesEndpoint: URL

  let version = "2023-06-01"
  let retryHandler: RetryHandler
  let timeout: TimeInterval
  let session: URLSession

  @MainActor public private(set) var isGenerating: Bool = false
  @MainActor private var currentTask: Task<(GenerationResponse, Bool), Error>?

  struct ThinkingConfig: Encodable {
    enum EnabledSetting: String, Encodable {
      case enabled, disabled, adaptive
    }

    let type: EnabledSetting
    let budgetTokens: Int?

    enum CodingKeys: String, CodingKey {
      case type
      case budgetTokens = "budget_tokens"
    }
  }

  /// Controls how much effort the model spends on thinking.
  /// Maps to the `effort` value in the API's `output_config`.
  public enum EffortLevel: String, CaseIterable, Identifiable, Sendable {
    case low, medium, high, max
    public var id: String {
      rawValue
    }
  }

  /// Configuration options for Anthropic API requests.
  public struct Configuration: Sendable {
    /// Maximum tokens for extended thinking. Set to enable thinking mode.
    /// The minimum value supported by Anthropic is 1024.
    public var maxThinkingTokens: Int?

    /// Controls how much effort the model spends on adaptive thinking.
    /// Setting this enables adaptive thinking (the model decides how much to think).
    /// Mutually exclusive with `maxThinkingTokens`.
    public var effort: EffortLevel?

    /// Enables web search tool for retrieving information from the internet.
    public var webSearch: Bool

    /// Enables web content fetching for retrieving full page content.
    public var webContent: Bool

    /// Enables code execution in a sandboxed environment.
    public var codeExecution: Bool

    /// Returns the thinking config adjusted for the given maxTokens.
    /// Returns nil if thinking should be skipped (e.g., budget would fall below 1024).
    func effectiveThinkingConfig(maxTokens: Int?) -> ThinkingConfig? {
      if effort != nil {
        return ThinkingConfig(type: .adaptive, budgetTokens: nil)
      }
      guard let maxThinkingTokens, maxThinkingTokens > 0 else {
        return nil
      }
      var budgetTokens = maxThinkingTokens
      // budget_tokens must be less than max_tokens
      if let maxTokens, budgetTokens >= maxTokens {
        budgetTokens = min(budgetTokens - 1, maxTokens - 1)
      }
      // budget_tokens must be at least 1024
      if budgetTokens < 1024 {
        return nil
      }
      return ThinkingConfig(type: .enabled, budgetTokens: budgetTokens)
    }

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - maxThinkingTokens: Maximum tokens for extended thinking. Minimum is 1024.
    ///   - effort: Effort level for adaptive thinking. Setting this enables adaptive thinking.
    ///   - webSearch: Enable web search tool.
    ///   - webContent: Enable web content fetching.
    ///   - codeExecution: Enable sandboxed code execution.
    public init(maxThinkingTokens: Int? = nil, effort: EffortLevel? = nil, webSearch: Bool = false, webContent: Bool = false, codeExecution: Bool = false) {
      self.maxThinkingTokens = maxThinkingTokens
      self.effort = effort
      self.webSearch = webSearch
      self.webContent = webContent
      self.codeExecution = codeExecution
    }
  }

  /// Creates a new Anthropic client.
  ///
  /// - Parameters:
  ///   - maxRetries: Maximum number of retry attempts for failed requests.
  ///   - timeout: Request timeout in seconds.
  ///   - session: URLSession to use for requests.
  ///   - messagesEndpoint: Custom endpoint URL for the messages API.
  public init(maxRetries: Int = 2, timeout: TimeInterval = 600, session: URLSession = .shared, messagesEndpoint: URL? = nil) {
    retryHandler = RetryHandler(maxRetries: maxRetries)
    self.timeout = timeout
    self.session = session
    self.messagesEndpoint = messagesEndpoint ?? baseURL.appendingPathComponent("messages")
  }

  /// Cancels any ongoing generation task.
  @MainActor
  public func stop() {
    currentTask?.cancel()
  }

  /// Helper method to make requests with retries
  private func makeRequest<T: Decodable>(
    endpoint: URL,
    method: String,
    apiKey: String,
    body: [String: any Sendable]? = nil,
    retries: Int? = nil,
  ) async throws -> T {
    let retriesRemaining = retries ?? retryHandler.maxRetries
    var lastResponseHeaders: [AnyHashable: Any]?
    do {
      var request = URLRequest(url: endpoint)
      request.httpMethod = method
      request.timeoutInterval = timeout
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.setValue(version, forHTTPHeaderField: "anthropic-version")
      request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
      if let body {
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
      }
      let (data, response) = try await session.data(for: request)
      guard let httpResponse = response as? HTTPURLResponse else {
        throw AIError.network(underlying: URLError(.badServerResponse))
      }
      lastResponseHeaders = httpResponse.allHeaderFields
      if !(200 ... 299).contains(httpResponse.statusCode) {
        throw Self.aiErrorFromHTTPResponse(httpResponse: httpResponse, data: data)
      }
      return try JSONDecoder().decode(T.self, from: data)
    } catch {
      if retriesRemaining > 0, retryHandler.shouldRetry(error, responseHeaders: lastResponseHeaders) {
        let delay = retryHandler.retryDelay(retriesRemaining: retriesRemaining, responseHeaders: lastResponseHeaders)
        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        return try await makeRequest(
          endpoint: endpoint,
          method: method,
          apiKey: apiKey,
          body: body,
          retries: retriesRemaining - 1,
        )
      }
      throw error
    }
  }
}

public extension AnthropicClient {
  /// Generates a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Anthropic model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
  ///   - configuration: Additional configuration options.
  /// - Returns: The generation response with text and metadata.
  func generateText(
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
  ///   - modelId: The Anthropic model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
  ///   - configuration: Additional configuration options.
  /// - Returns: An async stream of generation responses as they arrive.
  func streamText(
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
  func generateText(
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
  func streamText(
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
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    guard let apiKey else {
      throw AIError.authentication(message: "Missing API key")
    }
    await MainActor.run { isGenerating = true }
    let taskHolder = OSAllocatedUnfairLock(initialState: Task<(GenerationResponse, Bool), Error>?.none)
    let streamHolder = OSAllocatedUnfairLock(initialState: MessageStream?.none)

    do {
      let taskResult = try await withTaskCancellationHandler {
        try Task.checkCancellation()

        // Create a task that can be canceled and returns a result even when cancelled
        let task = Task<(GenerationResponse, Bool), Error> {
          var wasCancelled = false
          var latestSnapshot: APIMessage?
          var finalMessage: APIMessage?
          // Compute the effective thinking config, accounting for maxTokens and budget minimum.
          // Use the model default when the caller doesn't specify maxTokens, since
          // buildMessagesRequest will inject it and budget_tokens must be less than max_tokens.
          let effectiveMaxTokens = maxTokens ?? Self.defaultMaxTokens(for: modelId)
          let effectiveThinking = configuration.effectiveThinkingConfig(maxTokens: effectiveMaxTokens)
          let replayPlan = try await AnthropicReplayNormalizer.normalize(
            messages,
            thinkingEnabled: effectiveThinking != nil,
          )
          // Combine the explicit systemPrompt with any system/developer messages from history
          let combinedSystemPrompt: String? = {
            var parts: [String] = []
            if let systemPrompt, !systemPrompt.isEmpty {
              parts.append(systemPrompt)
            }
            parts.append(contentsOf: replayPlan.systemTexts)
            return parts.isEmpty ? nil : parts.joined(separator: "\n")
          }()
          // Temperature must be set to 1 when thinking is enabled.
          let adjustedTemperature = effectiveThinking != nil ? 1.0 : temperature

          // Create parameters
          var params = MessageCreateParams(
            model: modelId,
            messages: replayPlan.messages,
            maxTokens: maxTokens,
            system: combinedSystemPrompt,
            temperature: adjustedTemperature,
            thinking: effectiveThinking,
            effort: configuration.effort,
          )
          for tool in tools {
            if let baseSchemaBuildErrorMessage = tool.baseSchemaBuildErrorMessage {
              throw AIError.invalidRequest(
                message: "Tool '\(tool.name)' has an invalid input schema: \(baseSchemaBuildErrorMessage)",
              )
            }
          }
          // Tools - rawInputSchema is always populated (either explicit or generated from parameters)
          var anthropicTools = tools.map { tool -> AnthropicClient.APITool in
            APITool.rawCustom(
              name: tool.name,
              description: tool.description,
              rawInputSchema: tool.rawInputSchema,
            )
          }
          // Web search
          if configuration.webSearch {
            anthropicTools.append(.webSearch)
          }

          // Web fetch
          if configuration.webContent {
            anthropicTools.append(.webFetch)
          }

          // Code execution
          if configuration.codeExecution {
            anthropicTools.append(.codeExecution)
          }

          // Include tools if custom tools or web search tool are present
          if !anthropicTools.isEmpty {
            params.tools = anthropicTools
            params.toolChoice = .auto
          }
          // Create message stream using the provided API key
          let stream = await createMessageStream(params: params, apiKey: apiKey)
          streamHolder.withLock { $0 = stream }
          if Task.isCancelled {
            await stream.abort()
          }
          // Use AsyncStream for events
          let events = await stream.events()
          do {
            for await event in events {
              try Task.checkCancellation()
              switch event {
                case let .streamEvent(streamEvent, snapshot):
                  latestSnapshot = snapshot
                  switch streamEvent.type {
                    case .messageStart, .ping:
                      break
                    case .messageDelta, .contentBlockStart, .contentBlockDelta, .contentBlockStop, .messageStop:
                      await MainActor.run {
                        update(Self.generationResponse(from: snapshot))
                      }
                      if streamEvent.type == .messageStop {
                        finalMessage = snapshot
                      }
                    case .error:
                      break
                  }
                case let .finalMessage(message):
                  finalMessage = message
                  latestSnapshot = message
                case let .error(error):
                  throw error
                case let .abort(error):
                  wasCancelled = true
                  throw error
                case .end:
                  let finalSnapshot = finalMessage ?? latestSnapshot
                  return (finalSnapshot.map(Self.generationResponse(from:)) ?? GenerationResponse(content: []), wasCancelled)
                default:
                  continue
              }
            }
          } catch {
            // Check if the error is due to user cancellation
            if error is CancellationError {
              // Abort the stream to stop the background processing task
              await stream.abort()
              wasCancelled = true
            } else if let aiError = error as? AIError, case .cancelled = aiError {
              // Don't show error in UI
              wasCancelled = true
            } else if let aiError = error as? AIError {
              throw aiError
            } else {
              throw AIError.network(underlying: error)
            }
          }
          let finalSnapshot = finalMessage ?? latestSnapshot
          let result = finalSnapshot.map(Self.generationResponse(from:)) ?? GenerationResponse(content: [])
          return (result, wasCancelled)
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
        if let stream = streamHolder.withLock({ $0 }) {
          Task {
            await stream.abort()
          }
        }
      }

      taskHolder.withLock { $0 = nil }
      streamHolder.withLock { $0 = nil }
      await cleanUpGeneration()
      switch taskResult {
        case let .success((result, _)):
          return result
        case let .failure(error):
          if let aiError = error as? AIError {
            throw aiError
          } else if error is CancellationError {
            return .init(content: [])
          } else {
            throw AIError.network(underlying: error)
          }
      }
    } catch {
      taskHolder.withLock { $0 = nil }
      streamHolder.withLock { $0 = nil }
      await cleanUpGeneration()
      throw error
    }
  }

  @MainActor
  private func cleanUpGeneration() {
    isGenerating = false
    currentTask = nil
  }
}

// Model defaults

public extension AnthropicClient {
  /// Returns the default max_tokens for a given model ID.
  /// Newer models default to 64000; older models use their documented limits.
  internal static func defaultMaxTokens(for modelId: String) -> Int {
    if modelId.contains("claude-3-5-haiku") || modelId.contains("claude-3-5-sonnet") {
      8192
    } else if modelId.contains("claude-3-haiku") || modelId.contains("claude-3-sonnet")
      || modelId.contains("claude-3-opus")
    {
      4096
    } else if modelId.contains("claude-opus-4-1") {
      32000
    } else {
      64000
    }
  }

  /// Whether the given model supports extended thinking.
  /// Thinking was introduced with Claude 3.7 Sonnet; older models reject thinking parameters.
  /// Defaults to true for unrecognized models, since new Anthropic models generally support thinking
  /// and a blocklist of old models is more forward-compatible than an allowlist of known models.
  static func supportsThinking(_ modelId: String) -> Bool {
    if modelId.contains("claude-3-haiku") || modelId.contains("claude-3-opus")
      || modelId.contains("claude-3-sonnet") || modelId.contains("claude-3-5-")
      || modelId.contains("claude-2") || modelId.contains("claude-1")
      || modelId.contains("claude-instant")
    {
      return false
    }
    return true
  }

  /// Returns the maximum thinking budget for a given model ID.
  /// Thinking budget must be less than max_tokens.
  static func maxThinkingBudget(for modelId: String) -> Int {
    defaultMaxTokens(for: modelId) - 1
  }
}

extension AnthropicClient {
  static func aiErrorFromHTTPResponse(httpResponse: HTTPURLResponse, data: Data) -> AIError {
    let retryAfter = AIError.parseRetryAfter(from: httpResponse)
    let responseHeaders = httpResponse.allHeaderFields.reduce(into: [String: String]()) { result, pair in
      result["\(pair.key)"] = "\(pair.value)"
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    if let errorResponse = try? decoder.decode(ErrorResponse.self, from: data) {
      let message = errorResponse.error.message ?? "Unknown error"
      let errorType = errorResponse.error.type ?? "unknown"
      let context = AIError.ErrorContext(
        url: httpResponse.url,
        responseHeaders: responseHeaders,
        responseBody: data,
        providerInfo: .anthropic(type: errorType, message: errorResponse.error.message),
      )
      switch httpResponse.statusCode {
        case 400:
          return .invalidRequest(message: message)
        case 401:
          return .authentication(message: "There's an issue with your API key")
        case 403:
          return .authentication(message: "Your API key does not have permission to use the specified resource")
        case 404:
          return .invalidRequest(message: "Not found: \(message)")
        case 408, 409:
          return .serverError(statusCode: httpResponse.statusCode, message: message, context: context)
        case 429:
          return .rateLimit(retryAfter: retryAfter)
        case 500 ... 599:
          return .serverError(statusCode: httpResponse.statusCode, message: message, context: context)
        default:
          return .invalidRequest(message: message)
      }
    }

    let context = AIError.ErrorContext(
      url: httpResponse.url,
      responseHeaders: responseHeaders,
      responseBody: data,
    )
    return .serverError(statusCode: httpResponse.statusCode, message: "Unknown error", context: context)
  }

  private struct ErrorResponse: Codable {
    let error: ErrorDetails

    struct ErrorDetails: Codable {
      let type: String?
      let message: String?
      let param: String?
      let code: String?
    }
  }
}

let anthropicLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "AnthropicClient")
