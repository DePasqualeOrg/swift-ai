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
      providerInfo: ProviderErrorInfo? = nil,
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
      case .rateLimit, .network, .timeout: true
      case let .serverError(statusCode, _, _):
        statusCode == 408 || statusCode == 409 || statusCode >= 500
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

// MARK: - OpenAI-Format HTTP Error Handling

import os.log

extension AIError {
  /// Parses an HTTP error response in OpenAI-compatible format (also Fireworks, Mistral)
  /// and throws the appropriate `AIError`.
  static func throwOpenAIHTTPError(
    _ httpResponse: HTTPURLResponse,
    data: Data,
    logger: Logger,
  ) throws -> Never {
    let retryAfter = parseRetryAfter(from: httpResponse)
    let responseHeaders = httpResponse.allHeaderFields.reduce(into: [String: String]()) { result, pair in
      result["\(pair.key)"] = "\(pair.value)"
    }
    let context = ErrorContext(
      url: httpResponse.url,
      responseHeaders: responseHeaders,
      responseBody: data,
    )
    if let errorJson = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable] {
      logger.warning("Error: \(errorJson)")
      // OpenAI nested format
      if let error = errorJson["error"] as? [String: any Sendable], let message = error["message"] as? String {
        let type = error["type"] as? String
        let code = error["code"] as? String
        let providerContext = ErrorContext(
          url: context.url,
          responseHeaders: context.responseHeaders,
          responseBody: context.responseBody,
          providerInfo: .openAI(type: type ?? "unknown", code: code),
        )
        throw fromHTTPStatusCode(httpResponse.statusCode, message: message, retryAfter: retryAfter, context: providerContext)
      }
      // Fireworks format
      if let message = errorJson["error"] as? String {
        throw fromHTTPStatusCode(httpResponse.statusCode, message: message, retryAfter: retryAfter, context: context)
      }
      // Mistral format
      if let message = errorJson["message"] as? String {
        throw fromHTTPStatusCode(httpResponse.statusCode, message: message, retryAfter: retryAfter, context: context)
      }
    }
    throw fromHTTPStatusCode(httpResponse.statusCode, message: nil, retryAfter: retryAfter, context: context)
  }

  static func fromHTTPStatusCode(_ statusCode: Int, message: String? = nil, retryAfter: TimeInterval? = nil, context: ErrorContext? = nil) -> AIError {
    let errorMessage = message ?? "HTTP error \(statusCode)"
    return switch statusCode {
      case 401: .authentication(message: "Ensure the correct API key is being used.")
      case 403: .authentication(message: "You may be accessing the API from an unsupported country, region, or territory.")
      case 408, 409: .serverError(statusCode: statusCode, message: errorMessage, context: context)
      case 429: .rateLimit(retryAfter: retryAfter)
      case 500 ... 599: .serverError(statusCode: statusCode, message: errorMessage, context: context)
      default: .invalidRequest(message: errorMessage)
    }
  }

  /// Parses the `retry-after-ms` (custom, milliseconds) or `Retry-After`
  /// (standard, seconds or HTTP-date) headers from an HTTP response.
  static func parseRetryAfter(from response: HTTPURLResponse) -> TimeInterval? {
    // Custom header (milliseconds)
    if let retryAfterMs = response.value(forHTTPHeaderField: "retry-after-ms"),
       let ms = Double(retryAfterMs)
    {
      return ms / 1000.0
    }
    // Standard Retry-After header (seconds or HTTP-date)
    if let retryAfter = response.value(forHTTPHeaderField: "Retry-After") {
      if let seconds = Double(retryAfter) {
        return seconds
      }
      let formatter = DateFormatter()
      formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
      formatter.locale = Locale(identifier: "en_US_POSIX")
      if let date = formatter.date(from: retryAfter) {
        return max(0, date.timeIntervalSinceNow)
      }
    }
    return nil
  }
}
