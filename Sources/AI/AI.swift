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

// MARK: - Top-Level Generation Functions

/// Generate a text response from an LLM without streaming.
///
/// Example usage:
/// ```swift
/// let response = try await generateText(
///     model: .anthropic("claude-sonnet-4-20250514"),
///     messages: [Message(role: .user, content: "Hello!")],
///     apiKey: "sk-..."
/// )
/// print(response.texts.response ?? "No response")
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
///   - reasoning: Enable reasoning mode if supported. Defaults to true.
/// - Returns: The generation response.
public func generateText(
  model: Model,
  tools: [AI.Tool] = [],
  systemPrompt: String? = nil,
  messages: [Message],
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  reasoning: Bool = true
) async throws -> GenerationResponse {
  switch model {
    case let .anthropic(modelId):
      let client = AnthropicClient()
      let configuration = AnthropicClient.Configuration(
        maxThinkingTokens: reasoning ? AnthropicClient.maxThinkingBudget(for: modelId) : nil,
        webSearch: webSearch
      )
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )

    case let .gemini(modelId):
      let client = GeminiClient()
      let (thinkingLevel, thinkingBudget) = GeminiClient.thinkingConfig(for: modelId, reasoning: reasoning)
      let configuration = GeminiClient.Configuration(
        searchGrounding: webSearch,
        thinkingBudget: thinkingBudget,
        thinkingLevel: thinkingLevel
      )
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )

    case let .chatCompletions(modelId, endpoint):
      let client = ChatCompletionsClient(endpoint: endpoint)
      let configuration = ChatCompletionsClient.Configuration()
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )

    case let .responses(modelId, endpoint):
      let client = ResponsesClient(endpoint: endpoint)
      let configuration = ResponsesClient.Configuration(
        reasoningEffortLevel: reasoning && ResponsesClient.supportsReasoning(modelId) ? .medium : nil,
        serverSideTools: webSearch ? responsesWebSearchTools(modelId: modelId) : []
      )
      return try await client.generateText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )
  }
}

/// Generate a text response from an LLM with streaming.
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
///   - reasoning: Enable reasoning mode if supported. Defaults to true.
/// - Returns: An async stream of generation responses.
public func streamText(
  model: Model,
  tools: [AI.Tool] = [],
  systemPrompt: String? = nil,
  messages: [Message],
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  reasoning: Bool = true
) -> AsyncThrowingStream<GenerationResponse, Error> {
  switch model {
    case let .anthropic(modelId):
      let client = AnthropicClient()
      let configuration = AnthropicClient.Configuration(
        maxThinkingTokens: reasoning ? AnthropicClient.maxThinkingBudget(for: modelId) : nil,
        webSearch: webSearch
      )
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )

    case let .gemini(modelId):
      let client = GeminiClient()
      let (thinkingLevel, thinkingBudget) = GeminiClient.thinkingConfig(for: modelId, reasoning: reasoning)
      let configuration = GeminiClient.Configuration(
        searchGrounding: webSearch,
        thinkingBudget: thinkingBudget,
        thinkingLevel: thinkingLevel
      )
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )

    case let .chatCompletions(modelId, endpoint):
      let client = ChatCompletionsClient(endpoint: endpoint)
      let configuration = ChatCompletionsClient.Configuration()
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
      )

    case let .responses(modelId, endpoint):
      let client = ResponsesClient(endpoint: endpoint)
      let configuration = ResponsesClient.Configuration(
        reasoningEffortLevel: reasoning && ResponsesClient.supportsReasoning(modelId) ? .medium : nil,
        serverSideTools: webSearch ? responsesWebSearchTools(modelId: modelId) : []
      )
      return client.streamText(
        modelId: modelId,
        tools: tools,
        systemPrompt: systemPrompt,
        messages: messages,
        maxTokens: maxTokens,
        temperature: temperature,
        apiKey: apiKey,
        configuration: configuration
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
/// print(response.texts.response ?? "No response")
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
///   - reasoning: Enable reasoning mode if supported. Defaults to true.
/// - Returns: The generation response.
public func generateText(
  model: Model,
  tools: [AI.Tool] = [],
  systemPrompt: String? = nil,
  prompt: String,
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  reasoning: Bool = true
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
    reasoning: reasoning
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
///   - reasoning: Enable reasoning mode if supported. Defaults to true.
/// - Returns: An async stream of generation responses.
public func streamText(
  model: Model,
  tools: [AI.Tool] = [],
  systemPrompt: String? = nil,
  prompt: String,
  maxTokens: Int? = nil,
  temperature: Float? = nil,
  apiKey: String?,
  webSearch: Bool = false,
  reasoning: Bool = true
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
    reasoning: reasoning
  )
}

// MARK: - Helpers

/// Returns the appropriate web search server-side tools for a Responses API model.
/// xAI (Grok) uses a different tool format than OpenAI.
private func responsesWebSearchTools(modelId: String) -> [ResponsesClient.ServerSideTool] {
  if modelId.hasPrefix("grok-") {
    [.xAI.webSearch()]
  } else {
    [.OpenAI.webSearch(contextSize: .medium)]
  }
}
