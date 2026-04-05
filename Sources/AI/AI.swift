// Copyright © Anthony DePasquale

import Foundation

/// Represents a model from a specific provider.
///
/// Each case corresponds to a different LLM provider's API. The associated `String`
/// is the model identifier (e.g., `"claude-sonnet-4-20250514"`, `"gemini-2.0-flash"`).
public enum Model: Sendable {
  /// An Anthropic Claude model via the Messages API.
  case anthropic(String)
  /// A Google Gemini model via the Generative Language API.
  case gemini(String)
  /// A model using the OpenAI Chat Completions API format (OpenAI, xAI, or compatible).
  case chatCompletions(String, endpoint: URL = ChatCompletionsClient.Endpoint.openAI.url)
  /// A model using the OpenAI Responses API format (OpenAI, xAI, or compatible).
  case responses(String, endpoint: URL = ResponsesClient.Endpoint.openAI.url)

  /// The model identifier string.
  public var modelId: String {
    switch self {
      case let .anthropic(id): id
      case let .gemini(id): id
      case let .chatCompletions(id, _): id
      case let .responses(id, _): id
    }
  }
}

/// The provider family behind a Responses API-compatible endpoint.
///
/// This is used by the top-level `generateText` and `streamText` helpers when built-in
/// server-side tools like `webSearch` need provider-specific wire formats for custom
/// Responses endpoints.
public enum ResponsesProvider: Sendable, Equatable {
  case openAI
  case xAI
}

/// Provider-specific configuration for use with top-level generation functions.
///
/// Each case wraps the corresponding provider client's `Configuration` type.
/// When passed to ``generateText(model:tools:systemPrompt:messages:maxTokens:temperature:apiKey:webSearch:reasoning:configuration:)``
/// or ``streamText(model:tools:systemPrompt:messages:maxTokens:temperature:apiKey:webSearch:reasoning:configuration:)``,
/// the explicit configuration takes precedence and the `reasoning` and `webSearch`
/// parameters are ignored.
public enum ProviderConfiguration: Sendable {
  /// Configuration for Anthropic Claude models.
  case anthropic(AnthropicClient.Configuration)
  /// Configuration for Google Gemini models.
  case gemini(GeminiClient.Configuration)
  /// Configuration for OpenAI Chat Completions API models.
  case chatCompletions(ChatCompletionsClient.Configuration)
  /// Configuration for OpenAI Responses API models.
  case responses(ResponsesClient.Configuration)
}

// MARK: - Top-Level Generation Functions

/// Generate a text response from an LLM without streaming.
///
/// This function creates a new client for each call. For cancellation support and generation
/// state observation, use the provider-specific client directly (e.g., ``AnthropicClient``).
///
/// Example usage:
/// ```swift
/// let response = try await generateText(
///     model: .anthropic("claude-sonnet-4-20250514"),
///     messages: [Message(role: .user, content: "Hello!")],
///     apiKey: "sk-..."
/// )
/// print(response.content)
/// ```
///
/// - Parameters:
///   - model: The model to use, including provider and model ID.
///   - tools: Tools available for the model to use. Defaults to empty.
///   - systemPrompt: Optional system prompt.
///   - messages: The conversation messages.
///   - maxTokens: Maximum tokens to generate. Pass nil to use provider default.
///   - temperature: Sampling temperature. Pass nil to use model default.
///   - apiKey: The API key for authentication. Can be nil for local endpoints.
///   - webSearch: Enable web search if supported by the provider. Defaults to false.
///     Ignored when an explicit `configuration` is provided.
///   - responsesProvider: The provider family for custom Responses endpoints when using
///     built-in Responses tools such as `webSearch`. Ignored for non-Responses models and
///     when an explicit `configuration` is provided.
///   - reasoning: Enable reasoning/thinking for Anthropic and Gemini models. Defaults to true.
///     Ignored when an explicit `configuration` is provided.
///   - configuration: Optional provider-specific configuration. When provided, takes precedence
///     over `reasoning` and `webSearch`. Must match the provider specified in `model`.
/// - Returns: The generation response.
public func generateText(
  model: Model,
  tools: some Collection<AI.Tool> = [],
  systemPrompt: String? = nil,
  messages: [Message],
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  responsesProvider: ResponsesProvider? = nil,
  reasoning: Bool = true,
  configuration: ProviderConfiguration? = nil,
) async throws -> GenerationResponse {
  switch model {
    case let .anthropic(modelId):
      let client = AnthropicClient()
      let config: AnthropicClient.Configuration = try extractConfiguration(configuration, expected: .anthropic) {
        let enableThinking = reasoning && AnthropicClient.supportsThinking(modelId)
        return AnthropicClient.Configuration(
          effort: enableThinking ? .high : nil,
          webSearch: webSearch,
        )
      }
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )

    case let .gemini(modelId):
      let client = GeminiClient()
      let config: GeminiClient.Configuration = try extractConfiguration(configuration, expected: .gemini) {
        let (thinkingLevel, thinkingBudget) = GeminiClient.thinkingConfig(for: modelId, reasoning: reasoning)
        return GeminiClient.Configuration(
          searchGrounding: webSearch,
          thinkingBudget: thinkingBudget,
          thinkingLevel: thinkingLevel,
        )
      }
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )

    case let .chatCompletions(modelId, endpoint):
      let client = ChatCompletionsClient(endpoint: endpoint)
      let config: ChatCompletionsClient.Configuration = try extractConfiguration(configuration, expected: .chatCompletions) {
        ChatCompletionsClient.Configuration()
      }
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )

    case let .responses(modelId, endpoint):
      let client = ResponsesClient(endpoint: endpoint)
      let config: ResponsesClient.Configuration = try extractConfiguration(configuration, expected: .responses) {
        try defaultResponsesConfiguration(
          webSearch: webSearch,
          endpoint: endpoint,
          provider: responsesProvider,
        )
      }
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )
  }
}

/// Generate a text response from an LLM with streaming.
///
/// This function creates a new client for each call. For cancellation support and generation
/// state observation, use the provider-specific client directly (e.g., ``AnthropicClient``).
///
/// - Parameters:
///   - model: The model to use, including provider and model ID.
///   - tools: Tools available for the model to use. Defaults to empty.
///   - systemPrompt: Optional system prompt.
///   - messages: The conversation messages.
///   - maxTokens: Maximum tokens to generate. Pass nil to use provider default.
///   - temperature: Sampling temperature. Pass nil to use model default.
///   - apiKey: The API key for authentication. Can be nil for local endpoints.
///   - webSearch: Enable web search if supported by the provider. Defaults to false.
///     Ignored when an explicit `configuration` is provided.
///   - responsesProvider: The provider family for custom Responses endpoints when using
///     built-in Responses tools such as `webSearch`. Ignored for non-Responses models and
///     when an explicit `configuration` is provided.
///   - reasoning: Enable reasoning/thinking for Anthropic and Gemini models. Defaults to true.
///     Ignored when an explicit `configuration` is provided.
///   - configuration: Optional provider-specific configuration. When provided, takes precedence
///     over `reasoning` and `webSearch`. Must match the provider specified in `model`.
/// - Returns: An async stream of generation responses.
public func streamText(
  model: Model,
  tools: some Collection<AI.Tool> = [],
  systemPrompt: String? = nil,
  messages: [Message],
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  responsesProvider: ResponsesProvider? = nil,
  reasoning: Bool = true,
  configuration: ProviderConfiguration? = nil,
) -> AsyncThrowingStream<GenerationResponse, Error> {
  switch model {
    case let .anthropic(modelId):
      let client = AnthropicClient()
      let config: AnthropicClient.Configuration
      do {
        config = try extractConfiguration(configuration, expected: .anthropic) {
          let enableThinking = reasoning && AnthropicClient.supportsThinking(modelId)
          return AnthropicClient.Configuration(
            effort: enableThinking ? .high : nil,
            webSearch: webSearch,
          )
        }
      } catch {
        return AsyncThrowingStream { $0.finish(throwing: error) }
      }
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )

    case let .gemini(modelId):
      let client = GeminiClient()
      let config: GeminiClient.Configuration
      do {
        config = try extractConfiguration(configuration, expected: .gemini) {
          let (thinkingLevel, thinkingBudget) = GeminiClient.thinkingConfig(for: modelId, reasoning: reasoning)
          return GeminiClient.Configuration(
            searchGrounding: webSearch,
            thinkingBudget: thinkingBudget,
            thinkingLevel: thinkingLevel,
          )
        }
      } catch {
        return AsyncThrowingStream { $0.finish(throwing: error) }
      }
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )

    case let .chatCompletions(modelId, endpoint):
      let client = ChatCompletionsClient(endpoint: endpoint)
      let config: ChatCompletionsClient.Configuration
      do {
        config = try extractConfiguration(configuration, expected: .chatCompletions) {
          ChatCompletionsClient.Configuration()
        }
      } catch {
        return AsyncThrowingStream { $0.finish(throwing: error) }
      }
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )

    case let .responses(modelId, endpoint):
      let client = ResponsesClient(endpoint: endpoint)
      let config: ResponsesClient.Configuration
      do {
        config = try extractConfiguration(configuration, expected: .responses) {
          try defaultResponsesConfiguration(
            webSearch: webSearch,
            endpoint: endpoint,
            provider: responsesProvider,
          )
        }
      } catch {
        return AsyncThrowingStream { $0.finish(throwing: error) }
      }
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: config,
      )
  }
}

// MARK: - Prompt Convenience Overloads

/// Generate a text response from an LLM without streaming, using a simple prompt string.
///
/// This is a convenience overload for single-turn interactions where you don't need
/// conversation history.
///
/// Example usage:
/// ```swift
/// let response = try await generateText(
///     model: .anthropic("claude-sonnet-4-20250514"),
///     prompt: "What is the capital of France?",
///     apiKey: "sk-..."
/// )
/// print(response.content)
/// ```
///
/// - Parameters:
///   - model: The model to use, including provider and model ID.
///   - tools: Tools available for the model to use. Defaults to empty.
///   - systemPrompt: Optional system prompt.
///   - prompt: The user prompt to send.
///   - maxTokens: Maximum tokens to generate. Pass nil to use provider default.
///   - temperature: Sampling temperature. Pass nil to use model default.
///   - apiKey: The API key for authentication. Can be nil for local endpoints.
///   - webSearch: Enable web search if supported by the provider. Defaults to false.
///     Ignored when an explicit `configuration` is provided.
///   - responsesProvider: The provider family for custom Responses endpoints when using
///     built-in Responses tools such as `webSearch`. Ignored for non-Responses models and
///     when an explicit `configuration` is provided.
///   - reasoning: Enable reasoning/thinking for Anthropic and Gemini models. Defaults to true.
///     Ignored when an explicit `configuration` is provided.
///   - configuration: Optional provider-specific configuration. When provided, takes precedence
///     over `reasoning` and `webSearch`. Must match the provider specified in `model`.
/// - Returns: The generation response.
public func generateText(
  model: Model,
  tools: some Collection<AI.Tool> = [],
  systemPrompt: String? = nil,
  prompt: String,
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  responsesProvider: ResponsesProvider? = nil,
  reasoning: Bool = true,
  configuration: ProviderConfiguration? = nil,
) async throws -> GenerationResponse {
  try await generateText(
    model: model,
    tools: tools,
    systemPrompt: systemPrompt,
    messages: [Message(role: .user, content: prompt)],
    maxTokens: maxTokens,
    temperature: temperature,
    apiKey: apiKey,
    webSearch: webSearch,
    responsesProvider: responsesProvider,
    reasoning: reasoning,
    configuration: configuration,
  )
}

/// Generate a text response from an LLM with streaming, using a simple prompt string.
///
/// This is a convenience overload for single-turn interactions where you don't need
/// conversation history.
///
/// - Parameters:
///   - model: The model to use, including provider and model ID.
///   - tools: Tools available for the model to use. Defaults to empty.
///   - systemPrompt: Optional system prompt.
///   - prompt: The user prompt to send.
///   - maxTokens: Maximum tokens to generate. Pass nil to use provider default.
///   - temperature: Sampling temperature. Pass nil to use model default.
///   - apiKey: The API key for authentication. Can be nil for local endpoints.
///   - webSearch: Enable web search if supported by the provider. Defaults to false.
///     Ignored when an explicit `configuration` is provided.
///   - responsesProvider: The provider family for custom Responses endpoints when using
///     built-in Responses tools such as `webSearch`. Ignored for non-Responses models and
///     when an explicit `configuration` is provided.
///   - reasoning: Enable reasoning/thinking for Anthropic and Gemini models. Defaults to true.
///     Ignored when an explicit `configuration` is provided.
///   - configuration: Optional provider-specific configuration. When provided, takes precedence
///     over `reasoning` and `webSearch`. Must match the provider specified in `model`.
/// - Returns: An async stream of generation responses.
public func streamText(
  model: Model,
  tools: some Collection<AI.Tool> = [],
  systemPrompt: String? = nil,
  prompt: String,
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  responsesProvider: ResponsesProvider? = nil,
  reasoning: Bool = true,
  configuration: ProviderConfiguration? = nil,
) -> AsyncThrowingStream<GenerationResponse, Error> {
  streamText(
    model: model,
    tools: tools,
    systemPrompt: systemPrompt,
    messages: [Message(role: .user, content: prompt)],
    maxTokens: maxTokens,
    temperature: temperature,
    apiKey: apiKey,
    webSearch: webSearch,
    responsesProvider: responsesProvider,
    reasoning: reasoning,
    configuration: configuration,
  )
}

// MARK: - Helpers

func defaultResponsesConfiguration(
  webSearch: Bool,
  endpoint: URL,
  provider: ResponsesProvider?,
) throws -> ResponsesClient.Configuration {
  try ResponsesClient.Configuration(
    serverSideTools: webSearch ? responsesWebSearchTools(endpoint: endpoint, provider: provider) : [],
  )
}

/// Returns the appropriate web search server-side tools for a Responses API model.
/// Custom endpoints must provide an explicit provider because the OpenAI and xAI
/// Responses APIs use different tool formats.
func responsesWebSearchTools(
  endpoint: URL,
  provider: ResponsesProvider?,
) throws -> [ResponsesClient.ServerSideTool] {
  let resolvedProvider = try resolvedResponsesProvider(for: endpoint, provider: provider)

  return switch resolvedProvider {
    case .openAI:
      [ResponsesClient.ServerSideTool.OpenAI.webSearch(contextSize: .medium)]
    case .xAI:
      [ResponsesClient.ServerSideTool.xAI.webSearch()]
  }
}

private func resolvedResponsesProvider(
  for endpoint: URL,
  provider: ResponsesProvider?,
) throws -> ResponsesProvider {
  if let inferredProvider = inferredResponsesProvider(for: endpoint) {
    if let provider, provider != inferredProvider {
      throw AIError.invalidRequest(message:
        "responsesProvider \(provider.argumentName) conflicts with the built-in Responses endpoint for " +
          "\(inferredProvider.argumentName). Omit responsesProvider for known OpenAI/xAI endpoints, " +
          "or set it to \(inferredProvider.argumentName).")
    }
    return inferredProvider
  }

  guard let provider else {
    throw AIError.invalidRequest(message:
      "Custom Responses endpoints require an explicit responsesProvider when using webSearch. " +
        "Pass `.openAI` or `.xAI`, or provide an explicit `.responses` configuration.")
  }
  return provider
}

private func inferredResponsesProvider(for endpoint: URL) -> ResponsesProvider? {
  switch endpoint.host {
    case ResponsesClient.Endpoint.openAI.url.host:
      .openAI
    case ResponsesClient.Endpoint.xAI.url.host:
      .xAI
    default:
      nil
  }
}

private extension ResponsesProvider {
  var argumentName: String {
    switch self {
      case .openAI:
        "`.openAI`"
      case .xAI:
        "`.xAI`"
    }
  }
}

/// The provider family, used for configuration mismatch error messages.
private enum Provider: String {
  case anthropic
  case gemini
  case chatCompletions
  case responses
}

/// Extracts the provider-specific configuration from a `ProviderConfiguration`, or falls back
/// to building a default configuration from the generic parameters.
///
/// Throws ``AIError/invalidRequest(message:)`` if the configuration doesn't match the expected provider.
private func extractConfiguration<T>(
  _ configuration: ProviderConfiguration?,
  expected: Provider,
  default defaultConfig: () throws -> T,
) throws -> T {
  guard let configuration else {
    return try defaultConfig()
  }

  switch (configuration, expected) {
    case let (.anthropic(config), .anthropic):
      return config as! T
    case let (.gemini(config), .gemini):
      return config as! T
    case let (.chatCompletions(config), .chatCompletions):
      return config as! T
    case let (.responses(config), .responses):
      return config as! T
    default:
      let actual = switch configuration {
        case .anthropic: "anthropic"
        case .gemini: "gemini"
        case .chatCompletions: "chatCompletions"
        case .responses: "responses"
      }
      throw AIError.invalidRequest(message: "Configuration mismatch: expected \(expected.rawValue) configuration for \(expected.rawValue) model, but got \(actual) configuration")
  }
}
