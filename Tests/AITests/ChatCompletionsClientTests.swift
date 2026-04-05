// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

@Suite(.serialized)
struct ChatCompletionsClientTests {
  // MARK: - Test Helpers

  /// Sets up a mock handler and returns test ID and endpoint for client creation.
  private func setupMockHandler(sseData: String, statusCode: Int = 200) -> (testId: String, endpoint: URL) {
    let testId = UUID().uuidString
    let testEndpoint = URL(string: "https://mock.test/\(testId)")!

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }

    return (testId, testEndpoint)
  }

  // MARK: - Basic Response Tests

  @Test
  func `Parses basic text response and accumulates text`() async throws {
    let fixture = try loadFixture("openai_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())
    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
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

    // Verify metadata was captured
    #expect(response.metadata?.responseId == "chatcmpl-abc123")
    #expect(response.metadata?.model == "gpt-4")
  }

  @Test
  func `Parses function call response`() async throws {
    let fixture = try loadFixture("openai_function_call_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
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
    #expect(toolCall.id == "call_abc123")

    // Verify the function call parameters were parsed correctly
    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected location parameter to be a string")
    }

    // Verify text response is also present
    #expect(response.responseText?.contains("check the weather") == true)

    // Verify finish reason
    #expect(response.metadata?.finishReason == .toolUse)
  }

  @Test
  func `Invalid strict schema tool fails before request serialization`() async {
    let (testId, endpoint) = setupMockHandler(sseData: "")
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())
    let invalidTool = Tool(
      name: "invalid_tool",
      description: "Invalid strict schema tool",
      inputSchema: [
        "type": "object",
        "properties": [:],
      ],
      schemaBuildErrorMessage: "Strict mode requires all properties to be required.",
    ) { _ in
      [.text("ok")]
    }

    await #expect(throws: AIError.self) {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
        tools: [invalidTool],
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hi")],
        maxTokens: 128,
        apiKey: "test-api-key",
      ))
    }
  }

  @Test
  func `Duplicate imperative parameter names fail before request serialization`() async {
    let (testId, endpoint) = setupMockHandler(sseData: "")
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())
    let invalidTool = Tool(
      name: "invalid_tool",
      description: "Invalid tool with duplicate parameters",
      parameters: [
        .string("query", description: "First query"),
        .string("query", description: "Second query"),
      ],
    ) { _ in
      [.text("ok")]
    }

    await #expect(throws: AIError.self) {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
        tools: [invalidTool],
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hi")],
        maxTokens: 128,
        apiKey: "test-api-key",
      ))
    }
  }

  @Test
  func `Parses multiple function calls in single response`() async throws {
    let fixture = try loadFixture("openai_multiple_function_calls_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris and London?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
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

  @Test
  func `Extracts token usage metadata`() async throws {
    let fixture = try loadFixture("openai_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify token counts from the fixture
    // usage: prompt_tokens: 10, completion_tokens: 3, total_tokens: 13
    #expect(response.metadata?.inputTokens == 10)
    #expect(response.metadata?.outputTokens == 3)
    #expect(response.metadata?.totalTokens == 13)
  }

  @Test
  func `Extracts cache token metadata`() async throws {
    let fixture = try loadFixture("openai_cache_tokens_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify cache token metadata
    // prompt_tokens_details: cached_tokens: 80
    #expect(response.metadata?.cacheReadInputTokens == 80)
    #expect(response.metadata?.inputTokens == 100)
    #expect(response.metadata?.outputTokens == 5)
  }

  // MARK: - Finish Reason Tests

  @Test
  func `Sets finish reason correctly for normal stop`() async throws {
    let fixture = try loadFixture("openai_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.metadata?.finishReason == .stop)
  }

  @Test
  func `Sets finish reason correctly for max tokens (length)`() async throws {
    let fixture = try loadFixture("openai_max_tokens_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Write a long story")],
      maxTokens: 15,
      apiKey: "test-api-key",
    ))

    // Verify the response was truncated
    #expect(response.responseText?.isEmpty == false)

    // Verify finish reason is maxTokens
    #expect(response.metadata?.finishReason == .maxTokens)
  }

  @Test
  func `Sets finish reason correctly for tool calls`() async throws {
    let fixture = try loadFixture("openai_function_call_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.metadata?.finishReason == .toolUse)
  }

  @Test
  func `Sets finish reason correctly for content filter`() async throws {
    let fixture = try loadFixture("openai_content_filter_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Some inappropriate request")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify finish reason is contentFilter
    #expect(response.metadata?.finishReason == .contentFilter)

    // Verify partial response was captured
    #expect(response.responseText?.contains("can't") == true)
  }

  @Test
  func `Streaming refusal is preserved as opaque response content`() async throws {
    let sseData = """
    data: {"id":"chatcmpl-refusal123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can't assist with that."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Tell me how to do something disallowed")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.responseText == "I can't assist with that.")
    #expect(response.metadata?.finishReason == .refusal)

    let refusalBlock = response.content.compactMap { item -> OpaqueBlock? in
      guard case let .providerOpaque(block) = item,
            block.provider == "openai-chat-completions",
            block.type == "refusal"
      else {
        return nil
      }
      return block
    }.first
    #expect(refusalBlock?.content == "I can't assist with that.")
  }

  // MARK: - Non-Streaming Mode Tests

  @Test
  func `Non-streaming response returns complete response at once`() async throws {
    let fixture = try loadFixture("openai_non_streaming_response.json")
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "application/json"],
      )!
      return (response, fixture.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())

    let response = try await client.generateText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    )

    // Verify the complete response is returned
    #expect(response.responseText == "Hello! How can I help you today?")

    // Verify metadata is present
    #expect(response.metadata?.responseId == "chatcmpl-nonstream123")
    #expect(response.metadata?.model == "gpt-4")
    #expect(response.metadata?.finishReason == .stop)

    // Verify token counts
    #expect(response.metadata?.inputTokens == 15)
    #expect(response.metadata?.outputTokens == 8)
    #expect(response.metadata?.totalTokens == 23)
  }

  @Test
  func `Non-streaming refusal is preserved as opaque response content`() async throws {
    let refusalResponse = """
    {
      "id": "chatcmpl-refusal456",
      "object": "chat.completion",
      "created": 1700000000,
      "model": "gpt-4",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": null,
          "refusal": "I can't assist with that."
        },
        "finish_reason": "stop"
      }],
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
      }
    }
    """
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "application/json"],
      )!
      return (response, refusalResponse.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())

    let response = try await client.generateText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Tell me how to do something disallowed")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    )

    #expect(response.responseText == "I can't assist with that.")
    #expect(response.metadata?.finishReason == .refusal)

    let refusalBlock = response.content.compactMap { item -> OpaqueBlock? in
      guard case let .providerOpaque(block) = item,
            block.provider == "openai-chat-completions",
            block.type == "refusal"
      else {
        return nil
      }
      return block
    }.first
    #expect(refusalBlock?.content == "I can't assist with that.")
  }

  @Test
  func `Non-streaming response parses web search annotations as endnotes`() async throws {
    let fixture = try loadFixture("openai_annotations_response.json")
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "application/json"],
      )!
      return (response, fixture.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    let response = try await client.generateText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What is Swift?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    )

    #expect(response.responseText == "Swift is a programming language developed by Apple.")

    // Verify annotations were parsed into endnotes
    let endnotes = response.content.compactMap { block -> String? in
      if case let .endnotes(text) = block { return text }
      return nil
    }.joined()
    #expect(endnotes.contains("Swift Programming Language"))
    #expect(endnotes.contains("https://developer.apple.com/swift/"))
    #expect(endnotes.contains("Swift on GitHub"))
    #expect(endnotes.contains("https://github.com/apple/swift"))

    let annotationBlock = response.content.compactMap { block -> OpaqueBlock? in
      guard case let .providerOpaque(opaque) = block,
            opaque.provider == "openai-chat-completions",
            opaque.type == "annotations"
      else {
        return nil
      }
      return opaque
    }.first
    let annotationData = try #require(annotationBlock?.data?.data(using: .utf8))
    let annotationObjects = try #require(JSONSerialization.jsonObject(with: annotationData) as? [[String: Any]])
    let annotationURLs = Set(annotationObjects.compactMap { annotation -> String? in
      (annotation["url_citation"] as? [String: Any])?["url"] as? String
    })
    #expect(annotationURLs == [
      "https://developer.apple.com/swift/",
      "https://github.com/apple/swift",
    ])
  }

  @Test
  func `Streaming response preserves citations from earlier chunks`() async throws {
    let sseData = """
    data: {"id":"chatcmpl-citations123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Swift is a programming language."},"finish_reason":null}],"citations":["https://developer.apple.com/swift/"]}

    data: {"id":"chatcmpl-citations123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

    data: [DONE]
    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4.1",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What is Swift?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.responseText == "Swift is a programming language.")

    let endnotes = response.content.compactMap { block -> String? in
      if case let .endnotes(text) = block { return text }
      return nil
    }.joined()
    #expect(endnotes.contains("https://developer.apple.com/swift/"))
  }

  // MARK: - Empty Response Tests

  @Test
  func `Handles response with empty content gracefully`() async throws {
    let fixture = try loadFixture("openai_empty_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Response may be nil or empty string - both are valid for empty content
    let responseIsEmpty = response.responseText == nil
    #expect(responseIsEmpty, "Expected empty or nil response text")

    // Verify finish reason is still captured
    #expect(response.metadata?.finishReason == .stop)

    // Verify token counts show 0 output tokens
    #expect(response.metadata?.outputTokens == 0)
  }

  // MARK: - Thinking Content Tests

  @Test
  func `Handles thinking content in response`() async throws {
    let fixture = try loadFixture("openai_thinking_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "o1-preview",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What is the meaning of life?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ), collecting: collector)

    // Verify we got the final answer
    #expect(response.responseText?.contains("42") == true)

    // Verify thinking content was captured
    let updates = collector.updates
    let reasoningUpdates = updates.filter { $0.reasoningText != nil }
    #expect(!reasoningUpdates.isEmpty, "Expected at least one update with thinking content")

    // Verify the thinking content contains the expected text
    let reasoningText = reasoningUpdates.compactMap { $0.reasoningText }.joined()
    #expect(reasoningText.contains("think"), "Thinking content should contain 'think'")

    // Verify thinking tokens were captured
    #expect(response.metadata?.reasoningTokens == 15)
  }

  // MARK: - Error Handling Tests

  @Test
  func `Throws error for 401 unauthorized`() async throws {
    let errorResponse = """
    {"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 401)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "invalid-key",
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

  @Test
  func `Throws error for 403 forbidden`() async throws {
    let errorResponse = """
    {"error":{"message":"Access denied","type":"invalid_request_error"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 403)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key",
      ))
      Issue.record("Expected forbidden error")
    } catch let error as AIError {
      if case .authentication = error {
        // Expected (403 maps to authentication)
      } else {
        Issue.record("Expected authentication error, got: \(error)")
      }
    }
  }

  @Test
  func `Throws rate limit error for 429 status`() async throws {
    let errorResponse = """
    {"error":{"message":"Rate limit exceeded","type":"rate_limit_error"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 429)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
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
    {"error":{"message":"Internal server error","type":"server_error"}}
    """
    let (testId, endpoint) = setupMockHandler(sseData: errorResponse, statusCode: 500)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
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

  // MARK: - Network Error Tests

  @Test
  func `Handles network errors`() async throws {
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { _ in
      throw URLError(.notConnectedToInternet)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key",
      ))
      Issue.record("Expected network error")
    } catch {
      // Verify we got an error (could be LLMError.network or URLError)
      #expect(error is URLError || error is AIError)
    }
  }

  // MARK: - Invalid Response Handling Tests

  @Test
  func `Handles malformed JSON response gracefully`() async throws {
    let malformedData = """
    data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

    data: {"id":"test","object":"chat.completion.chunk" - THIS IS MALFORMED JSON

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: malformedData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-api-key",
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

  @Test
  func `Yields all chunks correctly`() async throws {
    let fixture = try loadFixture("openai_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: endpoint, session: makeMockSession())

    let collector = UpdateCollector()
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ), collecting: collector)

    // Verify we received multiple streaming updates (at least 3 text chunks)
    let updates = collector.updates
    #expect(updates.count >= 3)
  }

  // MARK: - Request Body Validation Tests

  @Test
  func `Request body includes system message correctly`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: "You are a helpful assistant",
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      temperature: 0.7,
      apiKey: "test-key",
    ))

    // Verify body was captured
    #expect(capturedBodyData != nil, "Request body should be available")

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    #expect(body != nil)

    // Verify messages array includes system message
    let messages = body?["messages"] as? [[String: Any]]
    #expect(messages != nil)
    #expect(try #require(messages?.count) >= 2) // At least system + user

    // Verify first message is system message
    let systemMessage = messages?.first
    #expect(systemMessage?["role"] as? String == "system")
    #expect(systemMessage?["content"] as? String == "You are a helpful assistant")

    // Verify model and other parameters
    #expect(body?["model"] as? String == "gpt-4")
    #expect(body?["max_completion_tokens"] as? Int == 1024)
    #expect(body?["stream"] as? Bool == true)

    // Use approximate comparison for floating-point
    if let temp = body?["temperature"] as? Double {
      #expect(abs(temp - 0.7) < 0.01, "Temperature should be approximately 0.7")
    } else {
      Issue.record("temperature should be present")
    }
  }

  @Test
  func `Request body includes tools correctly`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      tools: [makeTestTool(name: "get_weather", description: "Get current weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather?")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    // Verify body was captured
    #expect(capturedBodyData != nil, "Request body should be available")

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    #expect(body != nil)

    // Verify tools are included in request
    let tools = body?["tools"] as? [[String: Any]]
    #expect(tools != nil, "Request should include tools")
    #expect(try #require(tools).isEmpty == false)

    // Verify tool structure
    let firstTool = tools?.first
    #expect(firstTool?["type"] as? String == "function")

    let function = firstTool?["function"] as? [String: Any]
    #expect(function?["name"] as? String == "get_weather")
    #expect(function?["description"] as? String == "Get current weather")
  }

  @Test
  func `Request body replays assistant refusals via refusal field`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .providerOpaque(OpaqueBlock(
            provider: "openai-chat-completions",
            type: "refusal",
            content: "I can't help with that.",
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
    let assistantMessage = try #require(messages.first(where: { $0["role"] as? String == "assistant" }))

    #expect(assistantMessage["refusal"] as? String == "I can't help with that.")
    #expect(assistantMessage["content"] == nil)
  }

  @Test
  func `Non-assistant Chat Completions refusals replay as text content`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [
        Message(role: .user, content: [
          .providerOpaque(OpaqueBlock(
            provider: "openai-chat-completions",
            type: "refusal",
            content: "I can't help with that.",
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
    let firstUserMessage = try #require(messages.first(where: {
      $0["role"] as? String == "user" && $0["content"] as? String == "I can't help with that."
    }))

    #expect(firstUserMessage["refusal"] == nil)
  }

  @Test
  func `Chat Completions refusals survive replay through Responses as text`() async throws {
    let refusalSSEData = """
    data: {"id":"chatcmpl-refusal-cross-client","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can't assist with that."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

    data: [DONE]

    """
    let (chatTestId, chatEndpoint) = setupMockHandler(sseData: refusalSSEData)

    var capturedResponsesBody: Data?
    let responsesTestId = UUID().uuidString
    let responsesEndpoint = try #require(URL(string: "https://mock.test/\(responsesTestId)"))
    MockURLProtocol.setHandler(for: responsesTestId) { request in
      capturedResponsesBody = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Ok"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }

    defer {
      MockURLProtocol.removeHandler(for: chatTestId)
      MockURLProtocol.removeHandler(for: responsesTestId)
    }

    let chatClient = ChatCompletionsClient(endpoint: chatEndpoint, session: makeMockSession())
    let refusalResponse = try await consumeStream(chatClient.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Tell me how to do something disallowed")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    let responsesClient = ResponsesClient(endpoint: responsesEndpoint, session: makeMockSession())
    _ = try await consumeStream(responsesClient.streamText(
      modelId: "gpt-4o",
      messages: [refusalResponse.message, Message(role: .user, content: "Try again")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedResponsesBody)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let assistantMsg = try #require(input.first(where: {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    }))
    let content = try #require(assistantMsg["content"] as? [[String: Any]])
    let refusalItem = try #require(content.first(where: { $0["text"] as? String == "I can't assist with that." }))
    #expect(refusalItem["type"] as? String == "input_text")
  }

  @Test
  func `Responses annotated text and refusal replay through Chat Completions`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
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
    let assistantMessage = try #require(messages.first(where: { $0["role"] as? String == "assistant" }))

    #expect(assistantMessage["content"] as? String == "Cited answer.")
    #expect(assistantMessage["refusal"] as? String == "I can't continue beyond that.")
  }

  @Test
  func `Chat Completions annotations replay through Chat Completions`() async throws {
    let fixture = try loadFixture("openai_annotations_response.json")

    let parseId = UUID().uuidString
    let parseEndpoint = try #require(URL(string: "https://mock.test/\(parseId)"))
    MockURLProtocol.setHandler(for: parseId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "application/json"],
      )!
      return (response, fixture.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: parseId) }

    let parsingClient = ChatCompletionsClient(endpoint: parseEndpoint, session: makeMockSession())
    let parsedResponse = try await parsingClient.generateText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What is Swift?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    )

    var capturedBodyData: Data?
    let replayId = UUID().uuidString
    let replayEndpoint = try #require(URL(string: "https://mock.test/\(replayId)"))
    MockURLProtocol.setHandler(for: replayId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: replayId) }

    let replayClient = ChatCompletionsClient(endpoint: replayEndpoint, session: makeMockSession())
    _ = try await consumeStream(replayClient.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [parsedResponse.message, Message(role: .user, content: "Tell me more")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let messages = try #require(body?["messages"] as? [[String: Any]])
    let assistantMessage = try #require(messages.first(where: { $0["role"] as? String == "assistant" }))

    let content = try #require(assistantMessage["content"] as? String)
    #expect(content.contains("Swift is a programming language developed by Apple."))
    #expect(content.contains("https://developer.apple.com/swift/"))
    #expect(content.contains("https://github.com/apple/swift"))
    #expect(assistantMessage["annotations"] == nil)
  }

  @Test
  func `Foreign response-content opaque blocks replay through Chat Completions`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .providerOpaque(OpaqueBlock(
            provider: "anthropic",
            type: "web_fetch_tool_result",
            content: "Fetched article body.",
            data: #"{"type":"web_fetch_tool_result"}"#,
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
    let assistantMessage = try #require(messages.first(where: { $0["role"] as? String == "assistant" }))
    #expect(assistantMessage["content"] as? String == "Fetched article body.")
  }

  @Test
  func `Request body downgrades assistant attachments to plain text`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let imageData = try #require(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR2QAAAAASUVORK5CYII="))
    let documentData = Data("Quarterly report".utf8)

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .attachment(Attachment(kind: .image(data: imageData, mimeType: "image/png"))),
          .attachment(Attachment(kind: .document(data: documentData, mimeType: "application/pdf"), filename: "report.pdf")),
        ]),
        Message(role: .user, content: "Try again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let messages = try #require(body?["messages"] as? [[String: Any]])
    let assistantMessage = try #require(messages.first(where: { $0["role"] as? String == "assistant" }))
    let assistantContent = try #require(assistantMessage["content"] as? String)

    #expect((assistantMessage["content"] as? [[String: Any]]) == nil)
    #expect(assistantContent.contains("image/png"))
    #expect(assistantContent.contains("report.pdf"))
    #expect(assistantContent.contains("application/pdf"))
  }

  @Test
  func `Request body downgrades system and developer attachments to plain text`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let imageData = try #require(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR2QAAAAASUVORK5CYII="))
    let documentData = Data("Write robust tests".utf8)

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [
        Message(role: .system, content: [
          .text("Follow the attached playbook."),
          .attachment(Attachment(kind: .document(data: documentData, mimeType: "text/plain"), filename: "playbook.txt")),
        ]),
        Message(role: .developer, content: [
          .text("Use the visual spec."),
          .attachment(Attachment(kind: .image(data: imageData, mimeType: "image/png"))),
        ]),
        Message(role: .user, content: "Hello"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let messages = try #require(body?["messages"] as? [[String: Any]])
    let systemMessage = try #require(messages.first(where: { $0["role"] as? String == "system" }))
    let developerMessage = try #require(messages.first(where: { $0["role"] as? String == "developer" }))
    let systemContent = try #require(systemMessage["content"] as? String)
    let developerContent = try #require(developerMessage["content"] as? String)

    #expect((systemMessage["content"] as? [[String: Any]]) == nil)
    #expect(systemContent.contains("Follow the attached playbook."))
    #expect(systemContent.contains("playbook.txt"))
    #expect(systemContent.contains("text/plain"))

    #expect((developerMessage["content"] as? [[String: Any]]) == nil)
    #expect(developerContent.contains("Use the visual spec."))
    #expect(developerContent.contains("image/png"))
  }

  @Test
  func `Request includes authorization header`() async throws {
    var capturedRequest: URLRequest?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      capturedRequest = request
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "sk-test-api-key",
    ))

    // Verify authorization header
    #expect(capturedRequest != nil)
    let authHeader = capturedRequest?.value(forHTTPHeaderField: "Authorization")
    #expect(authHeader == "Bearer sk-test-api-key")
  }

  // MARK: - Cancellation Tests

  @Test
  func `Cancellation propagates correctly`() async throws {
    // Create a slow-responding fixture
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { _ in
      // Simulate a slow response
      let response = HTTPURLResponse(
        url: testEndpoint,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      // Return incomplete response that would normally hang
      let sseData = """
      data: {"id":"test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ChatCompletionsClient(endpoint: testEndpoint, session: makeMockSession())

    let task = Task {
      try await consumeStream(client.streamText(
        modelId: "gpt-4",
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
    } catch is CancellationError {
      // Expected - task was cancelled
    } catch {
      // Other errors may occur depending on timing
    }
  }
}
