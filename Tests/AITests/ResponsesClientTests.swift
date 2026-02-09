// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

@Suite("Responses Client", .serialized)
struct ResponsesClientTests {
  // MARK: - Test Helpers

  /// Loads a fixture file from the Fixtures directory.
  private func loadFixture(_ name: String) throws -> String {
    let fixturesURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("Fixtures")
      .appendingPathComponent(name)

    return try String(contentsOf: fixturesURL, encoding: .utf8)
  }

  /// Sets up a mock handler and returns test ID and endpoint for client creation.
  private func setupMockHandler(sseData: String, statusCode: Int = 200) -> (testId: String, endpoint: URL) {
    let testId = UUID().uuidString
    let testEndpoint = URL(string: "https://mock.test/\(testId)")!

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      return (response, sseData.data(using: .utf8)!)
    }

    return (testId, testEndpoint)
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

  /// Reads request body from either httpBody or httpBodyStream.
  private func readRequestBody(from request: URLRequest) -> Data? {
    if let body = request.httpBody {
      return body
    }
    if let stream = request.httpBodyStream {
      stream.open()
      defer { stream.close() }
      var data = Data()
      let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
      defer { buffer.deallocate() }
      while stream.hasBytesAvailable {
        let count = stream.read(buffer, maxLength: 4096)
        if count > 0 {
          data.append(buffer, count: count)
        } else {
          break
        }
      }
      return data.isEmpty ? nil : data
    }
    return nil
  }

  // MARK: - Basic Response Tests

  @Test("Parses basic text response and accumulates text")
  func basicTextResponse() async throws {
    let fixture = try loadFixture("responses_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
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

    // Verify metadata was captured
    #expect(response.metadata?.responseId == "resp_abc123")
    #expect(response.metadata?.model == "gpt-4o")
  }

  @Test("Parses function call response")
  func toolCallResponse() async throws {
    let fixture = try loadFixture("responses_function_call_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
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
    #expect(toolCall.id == "call_abc123")

    // Verify the function call parameters were parsed correctly
    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected location parameter to be a string")
    }

    // Verify text response is also present
    #expect(response.texts.response?.contains("check the weather") == true)
  }

  @Test("Parses multiple function calls in single response")
  func multipleToolCallsResponse() async throws {
    let fixture = try loadFixture("responses_multiple_function_calls.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris and London?")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify we got multiple function calls
    #expect(response.toolCalls.count == 2, "Expected 2 function calls for Paris and London")

    // Verify first function call
    let firstCall = response.toolCalls[0]
    #expect(firstCall.name == "get_weather")
    #expect(firstCall.id == "call_paris")
    if case let .string(location) = firstCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected location parameter to be 'Paris'")
    }

    // Verify second function call
    let secondCall = response.toolCalls[1]
    #expect(secondCall.name == "get_weather")
    #expect(secondCall.id == "call_london")
    if case let .string(location) = secondCall.parameters["location"] {
      #expect(location == "London")
    } else {
      Issue.record("Expected location parameter to be 'London'")
    }
  }

  @Test("Extracts token usage metadata")
  func tokenUsageMetadata() async throws {
    let fixture = try loadFixture("responses_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify token counts from the fixture
    // usage: input_tokens: 10, output_tokens: 3, total_tokens: 13
    #expect(response.metadata?.inputTokens == 10)
    #expect(response.metadata?.outputTokens == 3)
    #expect(response.metadata?.totalTokens == 13)
  }

  @Test("Extracts cache token metadata")
  func cacheTokenMetadata() async throws {
    let fixture = try loadFixture("responses_cache_tokens_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify cache token metadata
    // input_tokens_details: cached_tokens: 80
    #expect(response.metadata?.cacheReadInputTokens == 80)
    #expect(response.metadata?.inputTokens == 100)
    #expect(response.metadata?.outputTokens == 5)
  }

  // MARK: - Finish Reason Tests

  @Test("Sets finish reason correctly for normal stop")
  func finishReasonStop() async throws {
    let fixture = try loadFixture("responses_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    #expect(response.metadata?.finishReason == .stop)
  }

  @Test("Sets finish reason correctly for max tokens (incomplete)")
  func maxTokensFinishReason() async throws {
    let fixture = try loadFixture("responses_max_tokens_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Write a long story")],
      maxTokens: 15,
      apiKey: "test-api-key"
    ))

    // Verify the response was truncated
    #expect(response.texts.response?.isEmpty == false)

    // Verify finish reason is maxTokens
    #expect(response.metadata?.finishReason == .maxTokens)
  }

  @Test("Sets finish reason correctly for tool calls")
  func toolCallsFinishReason() async throws {
    let fixture = try loadFixture("responses_function_call_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather?")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify we got a function call
    #expect(!response.toolCalls.isEmpty)

    // Verify finish reason is toolUse
    #expect(response.metadata?.finishReason == .toolUse)
  }

  @Test("Sets finish reason correctly for content filter")
  func contentFilterFinishReason() async throws {
    let fixture = try loadFixture("responses_content_filter_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Some inappropriate request")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Verify finish reason is contentFilter
    #expect(response.metadata?.finishReason == .contentFilter)

    // Verify partial response was captured
    #expect(response.texts.response?.contains("can't") == true)
  }

  // MARK: - Non-Streaming Mode Tests

  @Test("Non-streaming response returns complete response at once")
  func nonStreamingResponse() async throws {
    let fixture = try loadFixture("responses_non_streaming_response.json")
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "application/json"]
      )!
      return (response, fixture.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())

    let response = try await client.generateText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    )

    // Verify the complete response is returned
    #expect(response.texts.response == "Hello! How can I help you today?")

    // Verify metadata is present
    #expect(response.metadata?.responseId == "resp_nonstream123")
    #expect(response.metadata?.model == "gpt-4o")
    #expect(response.metadata?.finishReason == .stop)

    // Verify token counts
    #expect(response.metadata?.inputTokens == 15)
    #expect(response.metadata?.outputTokens == 8)
    #expect(response.metadata?.totalTokens == 23)
  }

  // MARK: - Empty Response Tests

  @Test("Handles response with empty content gracefully")
  func emptyContentResponse() async throws {
    let fixture = try loadFixture("responses_empty_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ))

    // Response may be nil or empty string - both are valid for empty content
    let responseIsEmpty = response.texts.response == nil || response.texts.response?.isEmpty == true
    #expect(responseIsEmpty, "Expected empty or nil response text")

    // Verify finish reason is still captured
    #expect(response.metadata?.finishReason == .stop)

    // Verify token counts show 0 output tokens
    #expect(response.metadata?.outputTokens == 0)
  }

  // MARK: - Reasoning/Thinking Content Tests

  @Test("Handles reasoning content in response")
  func reasoningContentResponse() async throws {
    let fixture = try loadFixture("responses_reasoning_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "o3-mini",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What is the meaning of life?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
      configuration: .init(reasoningEffortLevel: .medium)
    ), collecting: collector)

    // Verify we got the final answer
    #expect(response.texts.response?.contains("42") == true)

    // Verify thinking content was captured
    let updates = collector.updates
    let reasoningUpdates = updates.filter { $0.texts.reasoning != nil }
    #expect(!reasoningUpdates.isEmpty, "Expected at least one update with thinking content")

    // Verify the thinking content contains the expected text
    let reasoningText = reasoningUpdates.compactMap { $0.texts.reasoning }.joined()
    #expect(reasoningText.contains("think"), "Thinking content should contain 'think'")
  }

  // MARK: - Error Handling Tests

  @Test("Throws error for 401 unauthorized")
  func unauthorizedError() async throws {
    let errorResponse = """
    {"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 401)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "invalid-key"
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

  @Test("Throws rate limit error for 429 status")
  func rateLimitError() async throws {
    let errorResponse = """
    {"error":{"message":"Rate limit exceeded","type":"rate_limit_error"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 429)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
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
    {"error":{"message":"Internal server error","type":"server_error"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 500)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
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

  // MARK: - Network Error Tests

  @Test("Handles network errors")
  func networkError() async throws {
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { _ in
      throw URLError(.notConnectedToInternet)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key"
      ))
      Issue.record("Expected network error")
    } catch {
      // Verify we got an error (could be LLMError.network or URLError)
      #expect(error is URLError || error is AIError)
    }
  }

  // MARK: - Invalid Response Handling Tests

  @Test("Handles malformed JSON response gracefully")
  func malformedJsonResponse() async throws {
    let malformedData = """
    data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

    data: {"type":"response.output_text.delta","delta":"Hello"}

    data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o - THIS IS MALFORMED JSON

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: malformedData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-api-key"
      ))
      Issue.record("Expected parsing error for malformed JSON")
    } catch let error as AIError {
      if case .parsing = error {
        // Expected - malformed JSON should cause parsing error
      } else {
        Issue.record("Expected parsing error, got: \(error)")
      }
    }
  }

  // MARK: - Stream Processing Tests

  @Test("Yields all chunks correctly")
  func yieldsAllChunks() async throws {
    let fixture = try loadFixture("responses_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let collector = UpdateCollector()
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key"
    ), collecting: collector)

    // Verify we received multiple streaming updates (at least 3 text chunks)
    let updates = collector.updates
    #expect(updates.count >= 3)
  }

  // MARK: - Request Body Validation Tests

  @Test("Request body includes instructions correctly")
  func instructionsInRequestBody() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { [self] request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      // Return a minimal valid response
      let sseData = """
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Hi"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Hi"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: "You are a helpful assistant",
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      temperature: 0.7,
      apiKey: "test-key"
    ))

    // Verify body was captured
    #expect(capturedBodyData != nil, "Request body should be available")

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    #expect(body != nil)

    // Verify instructions (system prompt) is included at top level
    let instructions = body?["instructions"] as? String
    #expect(instructions == "You are a helpful assistant", "Request should include instructions")

    // Verify model and other parameters
    #expect(body?["model"] as? String == "gpt-4o")
    #expect(body?["max_output_tokens"] as? Int == 1024)
    #expect(body?["stream"] as? Bool == true)

    // Verify input array exists
    let input = body?["input"] as? [[String: Any]]
    #expect(input != nil)
    #expect(try !#require(input?.isEmpty))
  }

  @Test("Request body includes tools correctly")
  func toolsInRequestBody() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { [self] request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      let sseData = """
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"I'll check."}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"I'll check."}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      tools: [makeTestTool(name: "get_weather", description: "Get current weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather?")],
      maxTokens: 1024,
      apiKey: "test-key"
    ))

    // Verify body was captured
    #expect(capturedBodyData != nil, "Request body should be available")

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    #expect(body != nil)

    // Verify tools are included in request
    let tools = body?["tools"] as? [[String: Any]]
    #expect(tools != nil, "Request should include tools")
    #expect(try !#require(tools?.isEmpty))

    // Verify tool structure (Responses API uses flat structure, not nested "function" object)
    let firstTool = tools?.first
    #expect(firstTool?["type"] as? String == "function")
    #expect(firstTool?["name"] as? String == "get_weather")
    #expect(firstTool?["description"] as? String == "Get current weather")
  }

  @Test("Request includes authorization header")
  func authorizationHeader() async throws {
    var capturedRequest: URLRequest?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedRequest = request
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      let sseData = """
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Hi"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "sk-test-api-key"
    ))

    // Verify authorization header
    #expect(capturedRequest != nil)
    let authHeader = capturedRequest?.value(forHTTPHeaderField: "Authorization")
    #expect(authHeader == "Bearer sk-test-api-key")
  }

  // MARK: - Cancellation Tests

  @Test("Cancellation propagates correctly")
  func cancellationPropagates() async throws {
    // Create a slow-responding fixture
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { _ in
      // Simulate a slow response
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      // Return incomplete response that would normally hang
      let sseData = """
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Hello"}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())

    let task = Task {
      try await consumeStream(client.streamText(
        modelId: "gpt-4o",
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
    } catch is CancellationError {
      // Expected - task was cancelled
    } catch {
      // Other errors may occur depending on timing
    }
  }
}
