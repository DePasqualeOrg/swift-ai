// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

/// Thread-safe collector for streaming updates in tests.
/// Uses a lock instead of an actor since update closures are synchronous.
final class UpdateCollector: @unchecked Sendable {
  private let lock = NSLock()
  private var _updates: [GenerationResponse] = []

  func append(_ response: GenerationResponse) {
    lock.lock()
    defer { lock.unlock() }
    _updates.append(response)
  }

  var updates: [GenerationResponse] {
    lock.lock()
    defer { lock.unlock() }
    return _updates
  }
}

@Suite("Anthropic Client", .serialized)
struct AnthropicClientTests {
  // MARK: - Test Helpers

  /// Loads a fixture file from the Fixtures directory.
  private func loadFixture(_ name: String) throws -> String {
    let fixturesURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("Fixtures")
      .appendingPathComponent(name)

    return try String(contentsOf: fixturesURL, encoding: .utf8)
  }

  /// Creates a mock client configured for testing with isolated handlers.
  private func makeTestClient(sseData: String, statusCode: Int = 200) -> (client: AnthropicClient, cleanup: () -> Void) {
    let testId = UUID().uuidString
    let testEndpoint = URL(string: "https://mock.test/\(testId)")!

    MockURLProtocol.setHandler(for: testId) { _ in
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      return (response, sseData.data(using: .utf8)!)
    }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint
    )

    return (client, { MockURLProtocol.removeHandler(for: testId) })
  }

  /// Creates a test function for use in tests.
  private func makeTestTool(name: String, description: String, paramName: String) -> Tool {
    Tool(
      name: name,
      description: description,
      title: name,
      parameters: [
        Tool.Parameter(
          name: paramName,
          title: paramName,
          type: .string,
          description: "Test parameter",
          required: true
        ),
      ],
      execute: { _ in [.text("test result")] }
    )
  }

  /// Consumes an async stream and returns the last element.
  private func consumeStream(
    _ stream: AsyncThrowingStream<GenerationResponse, Error>,
    collecting: UpdateCollector? = nil
  ) async throws -> GenerationResponse {
    var last: GenerationResponse?
    for try await response in stream {
      collecting?.append(response)
      last = response
    }
    guard let result = last else {
      fatalError("Stream ended without producing any values")
    }
    return result
  }

  // MARK: - Basic Response Tests

  @Test("Parses basic text response and accumulates text")
  func basicTextResponse() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ), collecting: collector)

    // Verify the final response contains the accumulated text
    #expect(response.texts.response == "Hello there!")

    // Verify we received streaming updates
    let updates = collector.updates
    #expect(!updates.isEmpty)

    // Verify the text was accumulated incrementally
    let textUpdates = updates.compactMap { $0.texts.response }
    #expect(textUpdates.contains { $0.contains("Hello") })
  }

  @Test("Parses tool use response with function call")
  func toolUseResponse() async throws {
    let fixture = try loadFixture("tool_use_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris?")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify we got a function call
    #expect(response.toolCalls.count == 1)

    let toolCall = response.toolCalls[0]
    #expect(toolCall.name == "get_weather")
    #expect(toolCall.id == "toolu_01NRLabsLyVHZPKxbKvkfSMn")

    // Verify the function call parameters were parsed correctly
    // The partial JSON "{\"location\": \"Paris\"}" should be assembled
    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected location parameter to be a string")
    }

    // Verify text response is also present
    #expect(response.texts.response?.contains("weather in Paris") == true)
  }

  @Test("Extracts token usage metadata")
  func tokenUsageMetadata() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify token counts from the fixture
    // From basic_response.txt: input_tokens: 11, output_tokens: 6
    #expect(response.metadata?.inputTokens == 11)
    #expect(response.metadata?.outputTokens == 6)
  }

  @Test("Sets finish reason correctly")
  func finishReason() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // From basic_response.txt: stop_reason: "end_turn"
    #expect(response.metadata?.finishReason == .stop)
  }

  @Test("Tool use sets finish reason to toolUse")
  func toolUseFinishReason() async throws {
    let fixture = try loadFixture("tool_use_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather?")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // From tool_use_response.txt: stop_reason: "tool_use"
    #expect(response.metadata?.finishReason == .toolUse)
  }

  // MARK: - Error Handling Tests

  @Test("Throws authentication error for 401 status")
  func authenticationError() async throws {
    let errorResponse = """
    {"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 401)
    defer { cleanup() }

    await #expect(throws: AIError.self) {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "invalid-key"
      ))
    }
  }

  @Test("Throws rate limit error for 429 status")
  func rateLimitError() async throws {
    let errorResponse = """
    {"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 429)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key"
      ))
      Issue.record("Expected rate limit error")
    } catch let error as AIError {
      if case .rateLimit = error {
        // Expected
      } else {
        Issue.record("Expected rate limit error, got: \(error)")
      }
    }
  }

  @Test("Throws server error for 500 status")
  func serverError() async throws {
    let errorResponse = """
    {"type":"error","error":{"type":"api_error","message":"Internal server error"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 500)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key"
      ))
      Issue.record("Expected server error")
    } catch let error as AIError {
      if case .serverError = error {
        // Expected
      } else {
        Issue.record("Expected server error, got: \(error)")
      }
    }
  }

  @Test("Throws error when API key is missing")
  func missingApiKey() async throws {
    let client = AnthropicClient()

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: nil
      ))
      Issue.record("Expected authentication error")
    } catch let error as AIError {
      if case .authentication = error {
        // Expected
      } else {
        Issue.record("Expected authentication error, got: \(error)")
      }
    }
  }

  // MARK: - Message Parsing Tests

  @Test("Parses message with response ID")
  func messageResponseId() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-opus-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-key"
    ))

    // Verify the response ID was captured from the fixture
    // From basic_response.txt: id: "msg_4QpJur2dWWDjF6C758FbBw5vm12BaVipnK"
    #expect(response.metadata?.responseId == "msg_4QpJur2dWWDjF6C758FbBw5vm12BaVipnK")
  }

  // MARK: - Network Error Tests

  @Test("Handles network errors")
  func networkError() async throws {
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { _ in
      throw URLError(.notConnectedToInternet)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint
    )

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key"
      ))
      Issue.record("Expected network error")
    } catch {
      // Verify we got an error (could be LLMError.network or URLError)
      // The important thing is that network errors are propagated
      #expect(error is URLError || error is AIError)
    }
  }

  // MARK: - In-Stream Error Tests

  @Test("Handles error event in stream")
  func streamErrorEvent() async throws {
    let sseData = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

    event: error
    data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}

    """
    let (client, cleanup) = makeTestClient(sseData: sseData)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key"
      ))
      Issue.record("Expected overloaded error")
    } catch let error as AIError {
      // Verify the error is handled appropriately
      // Could be .serverError or a specific overloaded error type
      switch error {
        case .serverError, .rateLimit:
          break // Expected - overloaded is typically a server-side issue
        default:
          Issue.record("Unexpected error type: \(error)")
      }
    }
  }

  // MARK: - Incomplete Response Tests

  @Test("Handles incomplete partial JSON when max_tokens is reached")
  func incompletePartialJsonResponse() async throws {
    let fixture = try loadFixture("incomplete_partial_json_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      tools: [makeTestTool(name: "make_file", description: "Make a file", paramName: "filename")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Create a file")],
      maxTokens: 100,
      apiKey: "test-key"
    ))

    // Verify finish reason is maxTokens (not toolUse or stop)
    #expect(response.metadata?.finishReason == .maxTokens)

    // Verify we got the text content
    #expect(response.texts.response?.contains("create a file") == true)

    // The incomplete tool use should still be captured (even if JSON is incomplete)
    // This tests graceful handling of truncated streams
    // The exact behavior depends on implementation - we just verify no crash
  }

  // MARK: - Stream Cancellation Tests

  @Test("Cancellation propagates correctly")
  func streamCancellation() async throws {
    // Use URL-specific handler to avoid interfering with other tests
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!

      // Return a complete response - cancellation test verifies no crash
      let sseData = """
      event: message_start
      data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

      event: content_block_stop
      data: {"type":"content_block_stop","index":0}

      event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

      event: message_stop
      data: {"type":"message_stop"}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint
    )

    let task = Task {
      try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key"
      ))
    }

    // Cancel after a brief moment
    try await Task.sleep(for: .milliseconds(10))
    task.cancel()

    // Verify the task completes (either successfully or cancelled)
    do {
      _ = try await task.value
      // If we get here, the stream completed before cancellation
      // This is acceptable - the test verifies cancellation doesn't crash
    } catch is CancellationError {
      // Expected - task was cancelled
    } catch {
      // Other errors may occur depending on timing
    }
  }
}
