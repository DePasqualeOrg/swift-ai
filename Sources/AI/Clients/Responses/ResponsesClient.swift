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

  @MainActor public internal(set) var isGenerating: Bool = false
  @MainActor var currentTask: Task<GenerationResponse, Error>?
  /// The ID of the currently active background response, if any
  /// This can be used to manually interrupt and resume background streams
  @MainActor public internal(set) var activeBackgroundResponseId: String?
  /// The API key associated with the active background response, used for authenticated cancellation
  @MainActor var activeBackgroundResponseApiKey: String?

  let session: URLSession

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
    stream: Bool,
    configuration: Configuration,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    await MainActor.run {
      isGenerating = true
    }
    let taskHolder = OSAllocatedUnfairLock(initialState: Task<GenerationResponse, Error>?.none)

    do {
      let result = try await withTaskCancellationHandler {
        try Task.checkCancellation()

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
              provider: configuration.provider,
              tools: tools,
              enableStrictModeForTools: configuration.enableStrictModeForTools,
            )

            for try await chunk in stream {
              try Task.checkCancellation()

              finalContent = chunk.content
              finalMetadata = chunk.metadata

              await MainActor.run {
                update(chunk)
              }
            }

            if Task.isCancelled {
              openAIResponsesLogger.log("Generation task returning cancelled state")
              return .init(content: finalContent, metadata: finalMetadata)
            }

            return .init(content: finalContent, metadata: finalMetadata)
          } catch {
            if error is CancellationError || (error as NSError).code == NSURLErrorCancelled {
              openAIResponsesLogger.log("Generation task caught cancellation error.")
              return .init(content: finalContent, metadata: finalMetadata)
            } else {
              openAIResponsesLogger.error("Generation failed: \(error)")
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
  func cleanUpGeneration() {
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
    if let backgroundResponseId = activeBackgroundResponseId {
      let apiKey = activeBackgroundResponseApiKey
      openAIResponsesLogger.log("Stop called - cancelling background response \(backgroundResponseId)")
      Task {
        try? await cancelBackgroundResponse(responseId: backgroundResponseId, apiKey: apiKey)
      }
    }
  }
}

public extension ResponsesClient {
  /// Reasoning effort level for models that support extended thinking.
  /// Maps to the `effort` value in the API's `reasoning` object.
  enum ReasoningEffortLevel: String, CaseIterable, Identifiable, Sendable {
    /// No reasoning. Default for gpt-5.1.
    case none
    /// Minimal reasoning effort for simple tasks.
    case minimal
    /// Low reasoning effort for straightforward tasks.
    case low
    /// Medium reasoning effort for balanced performance.
    case medium
    /// High reasoning effort for complex tasks.
    case high
    /// Maximum reasoning effort. Supported for models after gpt-5.1-codex-max.
    case xhigh

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Response verbosity level controlling output detail.
  enum VerbosityLevel: String, CaseIterable, Identifiable, Sendable {
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
  enum WebSearchContextSize: String, CaseIterable, Identifiable, Sendable {
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
  struct ServerSideTool: Sendable, Equatable {
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
      guard let lhsData = try? JSONSerialization.data(withJSONObject: lhs.definition, options: .sortedKeys),
            let rhsData = try? JSONSerialization.data(withJSONObject: rhs.definition, options: .sortedKeys)
      else {
        return false
      }
      return lhsData == rhsData
    }
  }

  /// Configuration options for Responses API requests.
  struct Configuration: Sendable {
    /// Reasoning effort level for extended thinking models.
    public var reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel?

    /// Response verbosity level.
    public var verbosityLevel: ResponsesClient.VerbosityLevel?

    /// Server-side tools to enable (web search, code interpreter, etc.).
    public var serverSideTools: [ServerSideTool]

    /// Enable background mode for long-running requests.
    public var backgroundMode: Bool

    /// The provider family for custom Responses-compatible endpoints.
    ///
    /// Built-in OpenAI and xAI endpoints infer this automatically. Set it when using
    /// a custom endpoint and you need provider-specific request behavior, such as
    /// reasoning replay capture for OpenAI-compatible backends.
    public var provider: ResponsesProvider?

    /// When true, tool schemas are rewritten for OpenAI strict mode compliance and sent
    /// with `strict: true`. This ensures the model's output exactly matches the schema,
    /// but requires all properties to be in `required` and optional properties to be nullable.
    /// Defaults to true. Disable for tools with optional non-nullable parameters or for
    /// compatible endpoints that don't support strict mode.
    public var enableStrictModeForTools: Bool

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - reasoningEffortLevel: Reasoning effort for thinking models.
    ///   - verbosityLevel: Response verbosity level.
    ///   - serverSideTools: Server-side tools to enable.
    ///   - backgroundMode: Enable background mode for long requests.
    ///   - provider: Provider family for custom Responses-compatible endpoints.
    ///   - enableStrictModeForTools: Rewrite tool schemas for strict mode compliance.
    public init(
      reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel? = nil,
      verbosityLevel: ResponsesClient.VerbosityLevel? = nil,
      serverSideTools: [ServerSideTool] = [],
      backgroundMode: Bool = false,
      provider: ResponsesProvider?,
      enableStrictModeForTools: Bool = true,
    ) {
      self.reasoningEffortLevel = reasoningEffortLevel
      self.verbosityLevel = verbosityLevel
      self.serverSideTools = serverSideTools
      self.backgroundMode = backgroundMode
      self.provider = provider
      self.enableStrictModeForTools = enableStrictModeForTools
    }

    public init(
      reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel? = nil,
      verbosityLevel: ResponsesClient.VerbosityLevel? = nil,
      serverSideTools: [ServerSideTool] = [],
      backgroundMode: Bool = false,
      enableStrictModeForTools: Bool = true,
    ) {
      self.init(
        reasoningEffortLevel: reasoningEffortLevel,
        verbosityLevel: verbosityLevel,
        serverSideTools: serverSideTools,
        backgroundMode: backgroundMode,
        provider: nil,
        enableStrictModeForTools: enableStrictModeForTools,
      )
    }
  }

  /// Status of a background response request.
  enum BackgroundResponseStatus: String, CaseIterable, Sendable {
    /// Request is waiting to be processed.
    case queued
    /// Request is currently being processed.
    case in_progress
    /// Request completed successfully.
    case completed
    /// Request ended early (e.g., max_output_tokens or content filtering) but the response payload is usable.
    case incomplete
    /// Request failed with an error.
    case failed
    /// Request was cancelled.
    case cancelled
  }

  /// Information about a background response request.
  struct BackgroundResponse: Sendable {
    /// The unique identifier for this response.
    public let id: String
    /// The current status of the response.
    public let status: BackgroundResponseStatus
    /// The generation response if completed, nil otherwise.
    public let response: GenerationResponse?
    /// Error message if the request failed.
    public let error: String?
  }
}
