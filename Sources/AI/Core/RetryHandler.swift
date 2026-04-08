// Copyright © Anthony DePasquale

import Foundation

/// Shared retry logic for API clients.
///
/// Handles exponential backoff with jitter, `Retry-After` / `retry-after-ms` header parsing,
/// and the `x-should-retry` header used by Anthropic and OpenAI.
struct RetryHandler {
  let maxRetries: Int

  private static let initialRetryDelay = 0.5
  private static let maxRetryDelay = 8.0

  init(maxRetries: Int = 2) {
    self.maxRetries = maxRetries
  }

  /// Determines whether a failed request should be retried, incorporating the
  /// `x-should-retry` response header when available.
  func shouldRetry(_ error: Error, responseHeaders: [AnyHashable: Any]? = nil) -> Bool {
    // Check x-should-retry header first — the server's explicit directive takes priority
    if let headers = responseHeaders {
      if let shouldRetry = headers["x-should-retry"] as? String ?? headers["X-Should-Retry"] as? String {
        if shouldRetry == "true" { return true }
        if shouldRetry == "false" { return false }
      }
    }

    if let urlError = error as? URLError {
      return urlError.code != .cancelled
    }
    if let aiError = error as? AIError {
      return aiError.isRetryable
    }
    return false
  }

  /// Calculates the retry delay, honoring server-provided `Retry-After` headers
  /// and falling back to exponential backoff with jitter.
  func retryDelay(retriesRemaining: Int, responseHeaders: [AnyHashable: Any]? = nil) -> TimeInterval {
    // Check server-provided retry delay
    if let headers = responseHeaders {
      if let delay = parseRetryAfter(headers: headers) {
        return delay
      }
    }
    return calculateBackoff(retriesRemaining: retriesRemaining)
  }

  /// Parses retry delay from `retry-after-ms` (custom, milliseconds) or
  /// `Retry-After` (standard, seconds or HTTP-date) response headers.
  private func parseRetryAfter(headers: [AnyHashable: Any]) -> TimeInterval? {
    // Custom header (milliseconds) — not standard, but supported by Anthropic and OpenAI
    if let retryAfterMs = headers["retry-after-ms"] as? String ?? headers["Retry-After-Ms"] as? String,
       let ms = Double(retryAfterMs)
    {
      return ms / 1000.0
    }

    // Standard Retry-After header (seconds or HTTP-date)
    if let retryAfter = headers["Retry-After"] as? String ?? headers["retry-after"] as? String {
      if let seconds = Double(retryAfter) {
        return seconds
      }
      // Try parsing as HTTP-date
      let formatter = DateFormatter()
      formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
      formatter.locale = Locale(identifier: "en_US_POSIX")
      if let date = formatter.date(from: retryAfter) {
        return max(0, date.timeIntervalSinceNow)
      }
    }

    return nil
  }

  /// Exponential backoff with jitter, matching the Anthropic/OpenAI TS SDKs.
  /// Base delay of 0.5s, doubling each retry, capped at 8s, with up to 25% jitter.
  private func calculateBackoff(retriesRemaining: Int) -> TimeInterval {
    let numRetries = maxRetries - retriesRemaining
    let sleepSeconds = min(Self.initialRetryDelay * pow(2.0, Double(numRetries)), Self.maxRetryDelay)
    let jitter = 1.0 - Double.random(in: 0.0 ... 0.25)
    return sleepSeconds * jitter
  }
}
