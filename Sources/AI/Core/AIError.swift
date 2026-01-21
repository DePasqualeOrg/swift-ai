// Copyright © Anthony DePasquale

import Foundation

/// Unified error type for LLM API clients.
public enum AIError: Error, Sendable {
  /// A network connectivity error occurred.
  case network(underlying: any Error)

  /// Authentication failed, typically due to an invalid or missing API key.
  case authentication(message: String)

  /// The API rate limit was exceeded.
  case rateLimit(retryAfter: TimeInterval?)

  /// The server returned an error response.
  case serverError(statusCode: Int, message: String, context: ErrorContext?)

  /// The request was invalid or malformed.
  case invalidRequest(message: String)

  /// Failed to parse the API response.
  case parsing(message: String)

  /// The request was cancelled by the user.
  case cancelled

  /// The request timed out.
  case timeout

  /// Additional context about an error for debugging.
  public struct ErrorContext: Sendable, Hashable {
    /// The URL that was requested.
    public var url: URL?

    /// The HTTP response headers, if available.
    public var responseHeaders: [String: String]?

    /// The raw response body, if available.
    public var responseBody: Data?

    /// Provider-specific error information.
    public var providerInfo: ProviderErrorInfo?

    /// Creates a new error context.
    ///
    /// - Parameters:
    ///   - url: The URL that was requested.
    ///   - responseHeaders: The HTTP response headers.
    ///   - responseBody: The raw response body.
    ///   - providerInfo: Provider-specific error information.
    public init(
      url: URL? = nil,
      responseHeaders: [String: String]? = nil,
      responseBody: Data? = nil,
      providerInfo: ProviderErrorInfo? = nil
    ) {
      self.url = url
      self.responseHeaders = responseHeaders
      self.responseBody = responseBody
      self.providerInfo = providerInfo
    }
  }

  /// Provider-specific error details for debugging and special handling.
  public enum ProviderErrorInfo: Sendable, Hashable {
    /// Error from the Anthropic API.
    case anthropic(type: String, message: String?)

    /// Error from the OpenAI Chat Completions API.
    case openAI(type: String, code: String?)

    /// Error from the Google Gemini API.
    case gemini(status: String, message: String?)

    /// Error from the OpenAI Responses API.
    case openAIResponses(type: String, code: String?)
  }

  /// Whether this error is potentially retryable.
  public var isRetryable: Bool {
    switch self {
      case .rateLimit, .serverError, .network, .timeout: true
      default: false
    }
  }
}

extension AIError: LocalizedError {
  public var errorDescription: String? {
    switch self {
      case let .network(underlying):
        "Network error: \(underlying.localizedDescription)"
      case let .authentication(message):
        "Authentication error: \(message)"
      case let .rateLimit(retryAfter):
        if let retryAfter {
          "Rate limited. Retry after \(Int(retryAfter)) seconds."
        } else {
          "Rate limited. Please try again later."
        }
      case let .serverError(statusCode, message, _):
        "Server error (\(statusCode)): \(message)"
      case let .invalidRequest(message):
        "Invalid request: \(message)"
      case let .parsing(message):
        "Parsing error: \(message)"
      case .cancelled:
        "Request was cancelled."
      case .timeout:
        "Request timed out."
    }
  }
}
