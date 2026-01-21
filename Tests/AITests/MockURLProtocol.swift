// Copyright © Anthony DePasquale

import Foundation

/// A mock URL protocol for testing HTTP requests without network access.
///
/// Uses URL-based routing to isolate tests that run concurrently.
///
/// Usage:
/// ```swift
/// let testId = UUID().uuidString
/// MockURLProtocol.setHandler(for: testId) { request in
///     let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
///     let data = "data: {\"foo\": true}\n\n".data(using: .utf8)!
///     return (response, data)
/// }
/// // Use URL: https://mock.test/\(testId)
/// ```
final class MockURLProtocol: URLProtocol, @unchecked Sendable {
  /// Lock for thread-safe access to static properties.
  private static let lock = NSLock()

  /// URL-keyed handlers for test isolation.
  /// The key is extracted from the URL path.
  /// Thread-safety is guaranteed by the lock.
  private nonisolated(unsafe) static var _handlers: [String: (URLRequest) throws -> (HTTPURLResponse, Data)] = [:]

  /// Legacy global handler (for backwards compatibility).
  /// Thread-safety is guaranteed by the lock.
  private nonisolated(unsafe) static var _requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

  /// Handler for streaming responses that yields data chunks over time.
  /// Thread-safety is guaranteed by the lock.
  private nonisolated(unsafe) static var _streamHandler: ((URLRequest) async throws -> (HTTPURLResponse, AsyncStream<Data>))?

  /// Thread-safe accessor for requestHandler.
  static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))? {
    get {
      lock.lock()
      defer { lock.unlock() }
      return _requestHandler
    }
    set {
      lock.lock()
      defer { lock.unlock() }
      _requestHandler = newValue
    }
  }

  /// Thread-safe accessor for streamHandler.
  static var streamHandler: ((URLRequest) async throws -> (HTTPURLResponse, AsyncStream<Data>))? {
    get {
      lock.lock()
      defer { lock.unlock() }
      return _streamHandler
    }
    set {
      lock.lock()
      defer { lock.unlock() }
      _streamHandler = newValue
    }
  }

  /// Sets a handler for a specific test ID.
  static func setHandler(for testId: String, handler: @escaping (URLRequest) throws -> (HTTPURLResponse, Data)) {
    lock.lock()
    defer { lock.unlock() }
    _handlers[testId] = handler
  }

  /// Removes a handler for a specific test ID.
  static func removeHandler(for testId: String) {
    lock.lock()
    defer { lock.unlock() }
    _handlers.removeValue(forKey: testId)
  }

  override class func canInit(with _: URLRequest) -> Bool {
    true
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest {
    request
  }

  /// Finds a matching handler for the given request by checking if the path starts with any registered test ID.
  private func findHandler(for request: URLRequest) -> ((URLRequest) throws -> (HTTPURLResponse, Data))? {
    guard let path = request.url?.path else { return nil }
    let cleanPath = path.hasPrefix("/") ? String(path.dropFirst()) : path

    MockURLProtocol.lock.lock()
    defer { MockURLProtocol.lock.unlock() }

    // First try exact match
    if let handler = MockURLProtocol._handlers[cleanPath] {
      return handler
    }

    // Then try prefix match (for cases like "/testId/modelId:action")
    for (testId, handler) in MockURLProtocol._handlers {
      if cleanPath.hasPrefix(testId) {
        return handler
      }
    }

    return nil
  }

  override func startLoading() {
    // Try streaming handler first
    if let streamHandler = MockURLProtocol.streamHandler {
      Task {
        do {
          let (response, dataStream) = try await streamHandler(request)
          client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
          for await chunk in dataStream {
            client?.urlProtocol(self, didLoad: chunk)
          }
          client?.urlProtocolDidFinishLoading(self)
        } catch {
          client?.urlProtocol(self, didFailWithError: error)
        }
      }
      return
    }

    // Try URL-specific handler (with prefix matching for nested paths)
    if let handler = findHandler(for: request) {
      do {
        let (response, data) = try handler(request)
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
      } catch {
        client?.urlProtocol(self, didFailWithError: error)
      }
      return
    }

    // Fall back to global handler
    guard let handler = MockURLProtocol.requestHandler else {
      client?.urlProtocol(self, didFailWithError: URLError(.unknown))
      return
    }

    do {
      let (response, data) = try handler(request)
      client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
      client?.urlProtocol(self, didLoad: data)
      client?.urlProtocolDidFinishLoading(self)
    } catch {
      client?.urlProtocol(self, didFailWithError: error)
    }
  }

  override func stopLoading() {}
}

/// Creates a URLSession configured to use MockURLProtocol.
func makeMockSession() -> URLSession {
  let config = URLSessionConfiguration.ephemeral
  config.protocolClasses = [MockURLProtocol.self]
  return URLSession(configuration: config)
}
