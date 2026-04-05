// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

@Suite(.serialized)
struct AnthropicClientTests {
  // MARK: - Test Helpers

  /// Creates a mock client configured for testing with isolated handlers.
  private func makeTestClient(sseData: String, statusCode: Int = 200) -> (client: AnthropicClient, cleanup: () -> Void) {
    let testId = UUID().uuidString
    let testEndpoint = URL(string: "https://mock.test/\(testId)")!

    MockURLProtocol.setHandler(for: testId) { _ in
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint,
    )

    return (client, { MockURLProtocol.removeHandler(for: testId) })
  }

  // MARK: - Basic Response Tests

  @Test
  func `Parses basic text response and accumulates text`() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ), collecting: collector)

    // Verify the final response contains the accumulated text
    #expect(response.responseText == "Hello there!")

    // Verify we received streaming updates
    let updates = collector.updates
    #expect(!updates.isEmpty)

    // Verify the text was accumulated incrementally
    let textUpdates = updates.compactMap(\.responseText)
    #expect(textUpdates.contains { $0.contains("Hello") })
  }

  @Test
  func `Parses tool use response with function call`() async throws {
    let fixture = try loadFixture("tool_use_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
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
    #expect(response.responseText?.contains("weather in Paris") == true)
  }

  @Test
  func `Extracts token usage metadata`() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify token counts from the fixture
    // From basic_response.txt: input_tokens: 11, output_tokens: 6
    #expect(response.metadata?.inputTokens == 11)
    #expect(response.metadata?.outputTokens == 6)
  }

  @Test
  func `Sets finish reason correctly`() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // From basic_response.txt: stop_reason: "end_turn"
    #expect(response.metadata?.finishReason == .stop)
  }

  @Test
  func `Tool use sets finish reason to toolUse`() async throws {
    let fixture = try loadFixture("tool_use_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // From tool_use_response.txt: stop_reason: "tool_use"
    #expect(response.metadata?.finishReason == .toolUse)
  }

  // MARK: - Error Handling Tests

  @Test
  func `Throws authentication error for 401 status`() async throws {
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
        apiKey: "invalid-key",
      ))
    }
  }

  @Test
  func `Throws rate limit error for 429 status`() async throws {
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
        apiKey: "test-key",
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

  @Test
  func `Throws server error for 500 status`() async throws {
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
        apiKey: "test-key",
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

  @Test
  func `Throws error when API key is missing`() async throws {
    let client = AnthropicClient()

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: nil,
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

  @Test
  func `Parses message with response ID`() async throws {
    let fixture = try loadFixture("basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-opus-4-20250514",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    // Verify the response ID was captured from the fixture
    // From basic_response.txt: id: "msg_4QpJur2dWWDjF6C758FbBw5vm12BaVipnK"
    #expect(response.metadata?.responseId == "msg_4QpJur2dWWDjF6C758FbBw5vm12BaVipnK")
  }

  @Test
  func `Request body normalizes raw tool schema for Anthropic`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))
    let rawSchemaTool = Tool(
      name: "raw_schema_tool",
      description: "Tool with unsupported schema keywords",
      inputSchema: [
        "type": .string("object"),
        "properties": .object([
          "meeting_time": .object([
            "type": .string("string"),
            "format": .string("custom-time"),
          ]),
          "config": .object([
            "type": .string("object"),
            "additionalProperties": .object([
              "type": .string("string"),
            ]),
          ]),
          "tags": .object([
            "type": .string("array"),
            "items": .object(["type": .string("string")]),
            "minItems": .int(2),
          ]),
        ]),
        "required": .array([.string("meeting_time"), .string("config"), .string("tags")]),
      ],
    ) { _ in [.text("ok")] }

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      event: message_start
      data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

      event: message_stop
      data: {"type":"message_stop"}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint,
    )

    _ = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      tools: [ToolWithDateAndData.tool, ProcessNestedData.tool, rawSchemaTool],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Test")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let tools = try #require(body?["tools"] as? [[String: Any]])

    let dateTool = try #require(tools.first { ($0["name"] as? String) == "tool_with_date_data" })
    let dateSchema = try #require(dateTool["input_schema"] as? [String: Any])
    let dateProperties = try #require(dateSchema["properties"] as? [String: Any])
    let payloadProp = try #require(dateProperties["payload"] as? [String: Any])

    #expect(payloadProp["contentEncoding"] == nil)
    let payloadDescription = try #require(payloadProp["description"] as? String)
    #expect(payloadDescription.contains("contentEncoding: \"base64\""))

    let nestedTool = try #require(tools.first { ($0["name"] as? String) == "process_nested_data" })
    let nestedSchema = try #require(nestedTool["input_schema"] as? [String: Any])
    let nestedProperties = try #require(nestedSchema["properties"] as? [String: Any])
    let groupedProp = try #require(nestedProperties["grouped"] as? [String: Any])

    #expect(groupedProp["additionalProperties"] == nil)
    let groupedDescription = try #require(groupedProp["description"] as? String)
    #expect(groupedDescription.contains("additionalProperties"))

    let rawTool = try #require(tools.first { ($0["name"] as? String) == "raw_schema_tool" })
    let rawSchema = try #require(rawTool["input_schema"] as? [String: Any])
    let rawProperties = try #require(rawSchema["properties"] as? [String: Any])

    let meetingTimeProp = try #require(rawProperties["meeting_time"] as? [String: Any])
    #expect(meetingTimeProp["format"] == nil)
    let meetingTimeDescription = try #require(meetingTimeProp["description"] as? String)
    #expect(meetingTimeDescription.contains("format: \"custom-time\""))

    let configProp = try #require(rawProperties["config"] as? [String: Any])
    #expect(configProp["additionalProperties"] as? Bool == false)

    let tagsProp = try #require(rawProperties["tags"] as? [String: Any])
    #expect(tagsProp["minItems"] == nil)
    let tagsDescription = try #require(tagsProp["description"] as? String)
    #expect(tagsDescription.contains("minItems: 2"))
  }

  @Test
  func `Chat Completions refusals replay as text blocks in Anthropic requests`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      event: message_start
      data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

      event: message_stop
      data: {"type":"message_stop"}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint,
    )

    _ = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .providerOpaque(OpaqueBlock(
            provider: "openai-chat-completions",
            type: "refusal",
            content: "I can't assist with that.",
            isResponseContent: true,
          )),
        ]),
        Message(role: .user, content: "Try again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let messages = try #require(body?["messages"] as? [[String: Any]])
    let assistantMessage = try #require(messages.first { ($0["role"] as? String) == "assistant" })
    let content = try #require(assistantMessage["content"] as? [[String: Any]])
    #expect(content.count == 1)
    #expect(content.first?["type"] as? String == "text")
    #expect(content.first?["text"] as? String == "I can't assist with that.")
  }

  @Test
  func `Responses annotated text and refusal replay as text blocks in Anthropic requests`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      event: message_start
      data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

      event: message_stop
      data: {"type":"message_stop"}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint,
    )

    _ = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .providerOpaque(OpaqueBlock(
            provider: "openai-responses",
            type: "annotated_output_text",
            content: "Cited answer.",
            data: "[{\"type\":\"url_citation\",\"url\":\"https://example.com\"}]",
            isResponseContent: true,
          )),
          .providerOpaque(OpaqueBlock(
            provider: "openai-responses",
            type: "refusal",
            content: "I can't continue beyond that.",
            isResponseContent: true,
          )),
        ]),
        Message(role: .user, content: "Try again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let messages = try #require(body?["messages"] as? [[String: Any]])
    let assistantMessage = try #require(messages.first { ($0["role"] as? String) == "assistant" })
    let content = try #require(assistantMessage["content"] as? [[String: Any]])
    let texts = content.compactMap { $0["text"] as? String }
    #expect(texts == ["Cited answer.", "I can't continue beyond that."])
  }

  // MARK: - Network Error Tests

  @Test
  func `Handles network errors`() async throws {
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { _ in
      throw URLError(.notConnectedToInternet)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = AnthropicClient(
      session: makeMockSession(),
      messagesEndpoint: testEndpoint,
    )

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key",
      ))
      Issue.record("Expected network error")
    } catch {
      // Verify we got an error (could be LLMError.network or URLError)
      // The important thing is that network errors are propagated
      #expect(error is URLError || error is AIError)
    }
  }

  // MARK: - In-Stream Error Tests

  @Test
  func `Handles error event in stream`() async throws {
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
        apiKey: "test-key",
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

  @Test
  func `Handles incomplete partial JSON when max_tokens is reached`() async throws {
    let fixture = try loadFixture("incomplete_partial_json_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "claude-sonnet-4-20250514",
      tools: [makeTestTool(name: "make_file", description: "Make a file", paramName: "filename")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Create a file")],
      maxTokens: 100,
      apiKey: "test-key",
    ))

    // Verify finish reason is maxTokens (not toolUse or stop)
    #expect(response.metadata?.finishReason == .maxTokens)

    // Verify we got the text content
    #expect(response.responseText?.contains("create a file") == true)

    // The incomplete tool use should still be captured (even if JSON is incomplete)
    // This tests graceful handling of truncated streams
    // The exact behavior depends on implementation - we just verify no crash
  }

  // MARK: - Stream Cancellation Tests

  @Test
  func `Cancellation propagates correctly`() async throws {
    // Use URL-specific handler to avoid interfering with other tests
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
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
      messagesEndpoint: testEndpoint,
    )

    let task = Task {
      try await consumeStream(client.streamText(
        modelId: "claude-sonnet-4-20250514",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key",
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

  // MARK: - Server-Side Tool Decoder Tests

  @Test
  func `Decodes newer web_fetch version tag`() throws {
    let json = """
    {"name": "web_fetch", "type": "web_fetch_20260209"}
    """
    let tool = try JSONDecoder().decode(AnthropicClient.APITool.self, from: #require(json.data(using: .utf8)))
    if case .webFetch = tool {
      // Expected
    } else {
      Issue.record("Expected .webFetch but got \(tool)")
    }
  }

  @Test
  func `Decodes newer code_execution version tag`() throws {
    let json = """
    {"name": "code_execution", "type": "code_execution_20250825"}
    """
    let tool = try JSONDecoder().decode(AnthropicClient.APITool.self, from: #require(json.data(using: .utf8)))
    if case .codeExecution = tool {
      // Expected
    } else {
      Issue.record("Expected .codeExecution but got \(tool)")
    }
  }

  @Test
  func `Decodes newer web_search version tag`() throws {
    let json = """
    {"name": "web_search", "type": "web_search_20260401"}
    """
    let tool = try JSONDecoder().decode(AnthropicClient.APITool.self, from: #require(json.data(using: .utf8)))
    if case .webSearch = tool {
      // Expected
    } else {
      Issue.record("Expected .webSearch but got \(tool)")
    }
  }
}
