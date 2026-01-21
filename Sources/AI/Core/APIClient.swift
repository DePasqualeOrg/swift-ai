// Copyright © Anthony DePasquale

import Foundation

/// A protocol that defines the interface for LLM API clients.
///
/// Conforming types provide generation capabilities for specific LLM providers,
/// handling both streaming and non-streaming responses.
public protocol APIClient: Sendable {
  associatedtype Configuration: Sendable = Void

  /// The types of tool result values this client supports.
  /// Used for filtering tools and providing fallback messages.
  static var supportedResultTypes: Set<ToolResult.ValueType> { get }

  /// Whether a generation request is currently in progress.
  @MainActor var isGenerating: Bool { get }

  /// Cancels any ongoing generation request.
  @MainActor func stop()

  /// Generate a text response without streaming.
  func generateText(
    modelId: String,
    tools: [Tool],
    systemPrompt: String?,
    messages: [Message],
    maxTokens: Int?,
    temperature: Float?,
    apiKey: String?,
    configuration: Configuration
  ) async throws -> GenerationResponse

  /// Generate a text response with streaming.
  func streamText(
    modelId: String,
    tools: [Tool],
    systemPrompt: String?,
    messages: [Message],
    maxTokens: Int?,
    temperature: Float?,
    apiKey: String?,
    configuration: Configuration
  ) -> AsyncThrowingStream<GenerationResponse, Error>
}
