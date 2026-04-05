// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

@Suite(.serialized)
struct ResponsesClientTests {
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
      apiKey: "test-api-key",
    ), collecting: collector)

    // Verify the final response contains the accumulated text
    #expect(response.responseText == "Hello there!")

    // Verify we received streaming updates
    let updates = collector.updates
    #expect(!updates.isEmpty)

    // Verify metadata was captured
    #expect(response.metadata?.responseId == "resp_abc123")
    #expect(response.metadata?.model == "gpt-4o")
  }

  @Test
  func `Parses function call response`() async throws {
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
  }

  @Test
  func `Parses multiple function calls in single response`() async throws {
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
    let fixture = try loadFixture("responses_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify token counts from the fixture
    // usage: input_tokens: 10, output_tokens: 3, total_tokens: 13
    #expect(response.metadata?.inputTokens == 10)
    #expect(response.metadata?.outputTokens == 3)
    #expect(response.metadata?.totalTokens == 13)
  }

  @Test
  func `Extracts cache token metadata`() async throws {
    let fixture = try loadFixture("responses_cache_tokens_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify cache token metadata
    // input_tokens_details: cached_tokens: 80
    #expect(response.metadata?.cacheReadInputTokens == 80)
    #expect(response.metadata?.inputTokens == 100)
    #expect(response.metadata?.outputTokens == 5)
  }

  // MARK: - Finish Reason Tests

  @Test
  func `Sets finish reason correctly for normal stop`() async throws {
    let fixture = try loadFixture("responses_basic_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.metadata?.finishReason == .stop)
  }

  @Test
  func `Sets finish reason correctly for max tokens (incomplete)`() async throws {
    let fixture = try loadFixture("responses_max_tokens_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
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
      apiKey: "test-api-key",
    ))

    // Verify we got a function call
    #expect(!response.toolCalls.isEmpty)

    // Verify finish reason is toolUse
    #expect(response.metadata?.finishReason == .toolUse)
  }

  @Test
  func `Sets finish reason correctly for content filter`() async throws {
    let fixture = try loadFixture("responses_content_filter_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
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
  func `Handles response failed terminal event`() async throws {
    let fixture = try loadFixture("responses_failed_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.metadata?.finishReason == .other)
    #expect(response.metadata?.responseId == "resp_fail123")
  }

  @Test
  func `Handles response incomplete terminal event`() async throws {
    let fixture = try loadFixture("responses_incomplete_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Write a long story")],
      maxTokens: 15,
      apiKey: "test-api-key",
    ))

    #expect(response.responseText == "Truncated output")
    #expect(response.metadata?.finishReason == .maxTokens)
    #expect(response.metadata?.responseId == "resp_inc123")
  }

  // MARK: - Non-Streaming Mode Tests

  @Test
  func `Non-streaming response returns complete response at once`() async throws {
    let fixture = try loadFixture("responses_non_streaming_response.json")
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

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())

    let response = try await client.generateText(
      modelId: "gpt-4o",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    )

    // Verify the complete response is returned
    #expect(response.responseText == "Hello! How can I help you today?")

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

  @Test
  func `Handles response with empty content gracefully`() async throws {
    let fixture = try loadFixture("responses_empty_response.txt")
    let (testId, endpoint) = setupMockHandler(sseData: fixture)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
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

  // MARK: - Reasoning/Thinking Content Tests

  @Test
  func `Handles reasoning content in response`() async throws {
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
      configuration: .init(reasoningEffortLevel: .medium),
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
  }

  // MARK: - EOF Fallback Tests

  @Test
  func `Clean EOF finalizes text and tool calls without completed event`() async throws {
    let sseData = """
    event: response.created
    data: {"type":"response.created","response":{"id":"resp_eof_func","status":"in_progress","model":"gpt-4o"}}

    event: response.output_text.delta
    data: {"type":"response.output_text.delta","delta":"I'll check the "}

    event: response.output_text.delta
    data: {"type":"response.output_text.delta","delta":"weather."}

    event: response.output_item.added
    data: {"type":"response.output_item.added","item":{"type":"function_call","name":"get_weather","call_id":"call_eof_weather"}}

    event: response.function_call_arguments.delta
    data: {"type":"response.function_call_arguments.delta","item_id":"call_eof_weather","delta":"{\\"location\\":"}

    event: response.function_call_arguments.done
    data: {"type":"response.function_call_arguments.done","item_id":"call_eof_weather","arguments":"{\\"location\\":\\"Paris\\"}"}

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      messages: [Message(role: .user, content: "What's the weather in Paris?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.responseText == "I'll check the weather.")
    #expect(response.metadata?.responseId == "resp_eof_func")
    #expect(response.metadata?.model == "gpt-4o")
    #expect(response.toolCalls.count == 1)

    let toolCall = try #require(response.toolCalls.first)
    #expect(toolCall.name == "get_weather")
    #expect(toolCall.id == "call_eof_weather")
    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected location parameter to be a string")
    }
  }

  @Test
  func `Clean EOF finalizes reasoning without completed event`() async throws {
    let sseData = """
    event: response.created
    data: {"type":"response.created","response":{"id":"resp_eof_reasoning","status":"in_progress","model":"o3-mini"}}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_123","type":"reasoning","summary":[{"type":"text","text":""}]}}

    event: response.content_part.added
    data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":""}}

    event: response.reasoning_text.delta
    data: {"type":"response.reasoning_text.delta","output_index":0,"content_index":0,"delta":"Let me think about this carefully."}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","role":"assistant","content":[]}}

    event: response.output_text.delta
    data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"The answer is 42."}

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "o3-mini",
      messages: [Message(role: .user, content: "What is the meaning of life?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
      configuration: .init(reasoningEffortLevel: .medium),
    ))

    #expect(response.reasoningText?.contains("think") == true)
    #expect(response.responseText == "The answer is 42.")
    #expect(response.metadata?.responseId == "resp_eof_reasoning")
  }

  @Test
  func `Clean EOF preserves multiple reasoning summary items without completed event`() async throws {
    let sseData = """
    event: response.created
    data: {"type":"response.created","response":{"id":"resp_eof_summary","status":"in_progress","model":"o3-mini"}}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_summary_123","type":"reasoning","summary":[]}}

    event: response.reasoning_summary_text.done
    data: {"type":"response.reasoning_summary_text.done","item_id":"rs_summary_123","output_index":0,"summary_index":0,"text":"First summary item."}

    event: response.reasoning_summary_text.done
    data: {"type":"response.reasoning_summary_text.done","item_id":"rs_summary_123","output_index":0,"summary_index":1,"text":"Second summary item."}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","role":"assistant","content":[]}}

    event: response.output_text.delta
    data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"Done."}

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "o3-mini",
      messages: [Message(role: .user, content: "Summarize your reasoning")],
      maxTokens: 1024,
      apiKey: "test-api-key",
      configuration: .init(reasoningEffortLevel: .medium),
    ))

    #expect(response.reasoningText?.contains("First summary item.") == true)
    #expect(response.reasoningText?.contains("Second summary item.") == true)
    #expect(response.responseText == "Done.")
  }

  @Test
  func `Clean EOF reasoning summaries round-trip as summary_text without completed event`() async throws {
    let eofSSEData = """
    event: response.created
    data: {"type":"response.created","response":{"id":"resp_eof_summary_roundtrip","status":"in_progress","model":"o3-mini"}}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_summary_roundtrip_123","type":"reasoning","summary":[]}}

    event: response.reasoning_summary_text.done
    data: {"type":"response.reasoning_summary_text.done","item_id":"rs_summary_roundtrip_123","output_index":0,"summary_index":0,"text":"First summary item."}

    event: response.reasoning_summary_text.done
    data: {"type":"response.reasoning_summary_text.done","item_id":"rs_summary_roundtrip_123","output_index":0,"summary_index":1,"text":"Second summary item."}

    data: [DONE]

    """
    let (eofTestId, eofEndpoint) = setupMockHandler(sseData: eofSSEData)

    var capturedBodyData: Data?
    let roundTripTestId = UUID().uuidString
    let roundTripEndpoint = try #require(URL(string: "https://mock.test/\(roundTripTestId)"))
    MockURLProtocol.setHandler(for: roundTripTestId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"o3-mini"}}

      data: {"type":"response.output_text.delta","delta":"Ok"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"o3-mini","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }

    defer {
      MockURLProtocol.removeHandler(for: eofTestId)
      MockURLProtocol.removeHandler(for: roundTripTestId)
    }

    let eofClient = ResponsesClient(endpoint: eofEndpoint, session: makeMockSession())
    let response = try await consumeStream(eofClient.streamText(
      modelId: "o3-mini",
      messages: [Message(role: .user, content: "Summarize your reasoning")],
      maxTokens: 1024,
      apiKey: "test-api-key",
      configuration: .init(reasoningEffortLevel: .medium),
    ))

    let roundTripClient = ResponsesClient(endpoint: roundTripEndpoint, session: makeMockSession())
    _ = try await consumeStream(roundTripClient.streamText(
      modelId: "o3-mini",
      messages: [response.message, Message(role: .user, content: "Continue")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let reasoningItem = try #require(input.first(where: { $0["type"] as? String == "reasoning" }))
    let summary = try #require(reasoningItem["summary"] as? [[String: Any]])

    #expect(summary.count == 2)
    #expect(summary[0]["type"] as? String == "summary_text")
    #expect(summary[0]["text"] as? String == "First summary item.")
    #expect(summary[1]["type"] as? String == "summary_text")
    #expect(summary[1]["text"] as? String == "Second summary item.")
  }

  @Test
  func `Clean EOF preserves annotations and message metadata without completed event`() async throws {
    let sseData = """
    event: response.created
    data: {"type":"response.created","response":{"id":"resp_eof_annotations","status":"in_progress","model":"gpt-4o"}}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_eof_123","type":"message","role":"assistant","status":"completed","content":[]}}

    event: response.content_part.added
    data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":"See docs","annotations":[{"type":"url_citation","url":"https://example.com/docs","title":"Example Docs"}]}}

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: [Message(role: .user, content: "Share the docs")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.responseText == "See docs")
    #expect(response.endnotesText?.contains("Example Docs") == true)
    #expect(response.endnotesText?.contains("https://example.com/docs") == true)
    #expect(response.content.contains { content in
      guard case let .providerOpaque(block) = content else { return false }
      return block.provider == "openai-responses" && block.type == "message_metadata"
    })
  }

  @Test
  func `Clean EOF preserves empty valid response without completed event`() async throws {
    let sseData = """
    event: response.created
    data: {"type":"response.created","response":{"id":"resp_eof_empty","status":"in_progress","model":"gpt-4o"}}

    event: response.output_item.added
    data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}

    data: [DONE]

    """
    let (testId, endpoint) = setupMockHandler(sseData: sseData)
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    let response = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    #expect(response.responseText == nil)
    #expect(response.metadata?.responseId == "resp_eof_empty")
    #expect(response.metadata?.model == "gpt-4o")
  }

  @Test
  func `Clean EOF without any snapshot throws parsing error`() async throws {
    let (testId, endpoint) = setupMockHandler(sseData: "data: [DONE]\n\n")
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())
    var sawChunk = false

    do {
      for try await _ in client.streamText(
        modelId: "gpt-4o",
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-api-key",
      ) {
        sawChunk = true
      }
      Issue.record("Expected parsing error for stream without a response snapshot")
    } catch let error as AIError {
      if case .parsing = error {
        #expect(sawChunk == false)
      } else {
        Issue.record("Expected parsing error, got: \(error)")
      }
    }
  }

  // MARK: - Error Handling Tests

  @Test
  func `Throws error for 401 unauthorized`() async throws {
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
  func `Throws rate limit error for 429 status`() async throws {
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

    let client = ResponsesClient(endpoint: endpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
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

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gpt-4o",
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
      apiKey: "test-api-key",
    ), collecting: collector)

    // Verify we received multiple streaming updates (at least 3 text chunks)
    let updates = collector.updates
    #expect(updates.count >= 3)
  }

  // MARK: - Request Body Validation Tests

  @Test
  func `Request body includes instructions correctly`() async throws {
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
      apiKey: "test-key",
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
    #expect(try #require(input).isEmpty == false)
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

    // Verify tool structure (Responses API uses flat structure, not nested "function" object)
    let firstTool = tools?.first
    #expect(firstTool?["type"] as? String == "function")
    #expect(firstTool?["name"] as? String == "get_weather")
    #expect(firstTool?["description"] as? String == "Get current weather")
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

  @Test
  func `Document attachment encodes file_data as data URL`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Done"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Done"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let pdfData = Data("fake-pdf-content".utf8)
    let attachment = Attachment(kind: .document(data: pdfData, mimeType: "application/pdf"), filename: "test.pdf")
    let message = Message(role: .user, content: [.attachment(attachment)])

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: [message],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let firstInput = try #require(input.first)
    let content = try #require(firstInput["content"] as? [[String: Any]])
    let fileContent = try #require(content.first(where: { $0["type"] as? String == "input_file" }))
    let fileData = try #require(fileContent["file_data"] as? String)

    let expectedBase64 = pdfData.base64EncodedString()
    #expect(fileData == "data:application/pdf;base64,\(expectedBase64)")
    #expect(fileContent["filename"] as? String == "test.pdf")
  }

  @Test
  func `Audio attachment is omitted from Responses input`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Done"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Done"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let audioData = Data("fake-wav-content".utf8)
    let audioAttachment = Attachment(kind: .audio(data: audioData, mimeType: "audio/wav"))
    let message = Message(role: .user, content: [
      .text("Describe this audio"),
      .attachment(audioAttachment),
    ])

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: [message],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let firstInput = try #require(input.first)
    let content = try #require(firstInput["content"] as? [[String: Any]])

    // Audio is not supported by the Responses API, so only the text should be present
    #expect(content.count == 1)
    #expect(content[0]["type"] as? String == "input_text")
    #expect(content[0]["text"] as? String == "Describe this audio")
  }

  @Test
  func `Assistant message without metadata preserves image and document attachments`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Done"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Done"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let imageData = try #require(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR2QAAAAASUVORK5CYII="))
    let pdfData = Data("fake-pdf-content".utf8)
    let imageAttachment = Attachment(kind: .image(data: imageData, mimeType: "image/png"))
    let documentAttachment = Attachment(kind: .document(data: pdfData, mimeType: "application/pdf"), filename: "context.pdf")
    let messages = [
      Message(role: .assistant, content: [
        .text("Keep these results in mind."),
        .attachment(imageAttachment),
        .attachment(documentAttachment),
      ]),
      Message(role: .user, content: "What should I do next?"),
    ]

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let assistantMessage = try #require(input.first(where: {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    }))
    let content = try #require(assistantMessage["content"] as? [[String: Any]])

    #expect(content.contains(where: { $0["type"] as? String == "input_text" && $0["text"] as? String == "Keep these results in mind." }))
    let imageContent = try #require(content.first(where: { $0["type"] as? String == "input_image" }))
    let imageURL = try #require(imageContent["image_url"] as? String)
    #expect(imageURL.hasPrefix("data:image/"))
    let fileContent = try #require(content.first(where: { $0["type"] as? String == "input_file" }))
    #expect(fileContent["filename"] as? String == "context.pdf")
  }

  @Test
  func `Assistant metadata-backed replay downgrades attachment segment to EasyInputMessage`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Done"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Done"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let metadata = OpaqueBlock(
      provider: "openai-responses",
      type: "message_metadata",
      data: #"{"id":"msg_123","status":"completed","phase":"commentary"}"#,
    )
    let pdfData = Data("attachment-content".utf8)
    let documentAttachment = Attachment(
      kind: .document(data: pdfData, mimeType: "application/pdf"),
      filename: "notes.pdf",
    )
    let messages = [
      Message(role: .assistant, content: [
        .providerOpaque(metadata),
        .text("Earlier answer."),
        .attachment(documentAttachment),
        .text("Supplemental note."),
      ]),
      Message(role: .user, content: "Continue"),
    ]

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let assistantMessages = input.filter {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    }
    #expect(assistantMessages.count == 2)

    let replayedOutputMessage = assistantMessages[0]
    #expect(replayedOutputMessage["id"] as? String == "msg_123")
    #expect(replayedOutputMessage["status"] as? String == "completed")
    #expect(replayedOutputMessage["phase"] as? String == "commentary")
    let replayedOutputContent = try #require(replayedOutputMessage["content"] as? [[String: Any]])
    #expect(replayedOutputContent.count == 1)
    #expect(replayedOutputContent[0]["type"] as? String == "output_text")
    #expect(replayedOutputContent[0]["text"] as? String == "Earlier answer.")

    let downgradedAssistantMessage = assistantMessages[1]
    #expect(downgradedAssistantMessage["id"] == nil)
    let downgradedContent = try #require(downgradedAssistantMessage["content"] as? [[String: Any]])
    #expect(downgradedContent[0]["type"] as? String == "input_file")
    #expect(downgradedContent[0]["filename"] as? String == "notes.pdf")
    #expect(downgradedContent[1]["type"] as? String == "input_text")
    #expect(downgradedContent[1]["text"] as? String == "Supplemental note.")
  }

  @Test
  func `Mixed tool output preserves original content order`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Ok"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    // Build a conversation with a tool call followed by a mixed tool result: [image, text, file]
    let imageData = Data("fake-image".utf8)
    let pdfData = Data("fake-pdf".utf8)
    let toolCall = ToolCall(name: "analyze", id: "call_123", parameters: [:])
    let toolResult = ToolResult(
      name: "analyze",
      id: "call_123",
      content: [
        .image(imageData, mimeType: "image/png"),
        .text("Analysis complete"),
        .file(pdfData, mimeType: "application/pdf", filename: "report.pdf"),
      ],
    )
    let messages = [
      Message(role: .assistant, content: [.toolCall(toolCall)]),
      Message(role: .tool, content: [.toolResult(toolResult)]),
      Message(role: .user, content: "What did you find?"),
    ]

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])

    // Find the function_call_output item
    let toolOutput = try #require(input.first(where: { $0["type"] as? String == "function_call_output" }))
    let outputItems = try #require(toolOutput["output"] as? [[String: Any]])

    // Verify order: image, text, file — matching the original ToolResult.content order
    #expect(outputItems.count == 3)
    #expect(outputItems[0]["type"] as? String == "input_image")
    #expect(outputItems[1]["type"] as? String == "input_text")
    #expect(outputItems[1]["text"] as? String == "Analysis complete")
    #expect(outputItems[2]["type"] as? String == "input_file")
    #expect(outputItems[2]["filename"] as? String == "report.pdf")
  }

  @Test
  func `Reasoning items round-trip with correct wire shape`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"o3-mini"}}

      data: {"type":"response.output_text.delta","delta":"Ok"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"o3-mini","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    // Build history with a preserved reasoning opaque block
    let reasoningBlock = OpaqueBlock(
      provider: "openai-responses",
      type: "reasoning",
      content: "Let me think step by step.",
      signature: "reasoning_item_abc123",
      data: "encrypted_data_here",
    )
    let messages = [
      Message(role: .assistant, content: [
        .thinking(text: "Let me think step by step.", signature: nil),
        .providerOpaque(reasoningBlock),
        .text("The answer is 42."),
      ]),
      Message(role: .user, content: "Can you explain more?"),
    ]

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "o3-mini",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])

    // The reasoning opaque block should be serialized as a standalone reasoning item
    let reasoningItem = try #require(input.first(where: { $0["type"] as? String == "reasoning" }))
    #expect(reasoningItem["id"] as? String == "reasoning_item_abc123")
    #expect(reasoningItem["encrypted_content"] as? String == "encrypted_data_here")

    let summary = try #require(reasoningItem["summary"] as? [[String: Any]])
    #expect(summary.count == 1)
    #expect(summary[0]["type"] as? String == "summary_text")
    #expect(summary[0]["text"] as? String == "Let me think step by step.")

    // Manually constructed assistant messages (without message_metadata) use input_text
    let assistantMsg = try #require(input.first(where: {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    }))
    let content = try #require(assistantMsg["content"] as? [[String: Any]])
    #expect(content.contains(where: { $0["type"] as? String == "input_text" && $0["text"] as? String == "The answer is 42." }))
  }

  @Test
  func `Assistant message without metadata omits annotations and serializes refusal as text`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Ok"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    // Build history without message_metadata — simulates callers who
    // persisted Message values but dropped the opaque metadata block.
    let annotatedBlock = OpaqueBlock(
      provider: "openai-responses",
      type: "annotated_output_text",
      content: "See this source.",
      data: "[{\"type\":\"url_citation\",\"url\":\"https://example.com\"}]",
      isResponseContent: true,
    )
    let refusalBlock = OpaqueBlock(
      provider: "openai-responses",
      type: "refusal",
      content: "I cannot help with that.",
      isResponseContent: true,
    )
    let messages = [
      Message(role: .assistant, content: [
        .providerOpaque(annotatedBlock),
        .providerOpaque(refusalBlock),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let assistantMsg = try #require(input.first(where: {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    }))
    let content = try #require(assistantMsg["content"] as? [[String: Any]])

    // Without metadata, annotated text should become plain input_text without annotations
    let annotatedItem = try #require(content.first(where: { $0["text"] as? String == "See this source." }))
    #expect(annotatedItem["type"] as? String == "input_text")
    #expect(annotatedItem["annotations"] == nil)

    // Refusal should become plain input_text, not a "refusal" type
    let refusalItem = try #require(content.first(where: { $0["text"] as? String == "I cannot help with that." }))
    #expect(refusalItem["type"] as? String == "input_text")
  }

  @Test
  func `Assistant message with metadata emits output_text annotations array`() async throws {
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
      data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Ok"}

      data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"gpt-4o","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

      data: [DONE]

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let metadataJson = #"{"id":"msg_123","status":"completed"}"#
    let messages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: "openai-responses",
          type: "message_metadata",
          data: metadataJson,
        )),
        .text("The answer is 42."),
      ]),
      Message(role: .user, content: "Follow up"),
    ]

    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "gpt-4o",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let input = try #require(body?["input"] as? [[String: Any]])
    let assistantMsg = try #require(input.first(where: {
      $0["type"] as? String == "message" && $0["role"] as? String == "assistant"
    }))
    #expect(assistantMsg["id"] as? String == "msg_123")
    #expect(assistantMsg["status"] as? String == "completed")

    let content = try #require(assistantMsg["content"] as? [[String: Any]])
    let outputItem = try #require(content.first(where: { $0["text"] as? String == "The answer is 42." }))
    #expect(outputItem["type"] as? String == "output_text")
    let annotations = try #require(outputItem["annotations"] as? [[String: Any]])
    #expect(annotations.isEmpty)
  }

  @Test
  func `Stop sends authenticated cancel for background response`() async throws {
    var cancelRequest: URLRequest?
    let streamGate = AsyncStream<Data>.makeStream()

    // Use the global stream handler for controlled timing
    MockURLProtocol.streamHandler = { request in
      let url = request.url!

      // Handle cancel request
      if url.path.contains("/cancel") {
        cancelRequest = request
        let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
        return (response, AsyncStream { $0.yield(Data("{}".utf8)); $0.finish() })
      }

      // Return a background response with controlled timing
      let response = HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!

      // Send the response.created event immediately, then hold the stream open
      let createdEvent = """
      data: {"type":"response.created","response":{"id":"resp_bg_123","status":"in_progress","model":"gpt-4o"}}

      data: {"type":"response.output_text.delta","delta":"Working"}


      """
      streamGate.continuation.yield(Data(createdEvent.utf8))

      return (response, streamGate.stream)
    }
    defer { MockURLProtocol.streamHandler = nil }

    let testEndpoint = try #require(URL(string: "https://mock.test/bg-cancel-test"))
    let client = ResponsesClient(endpoint: testEndpoint, session: makeMockSession())

    // Start a background stream
    let task = Task {
      try await consumeStream(client.streamText(
        modelId: "gpt-4o",
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "secret-api-key",
        configuration: .init(backgroundMode: true),
      ))
    }

    // Wait for the background response ID to be set
    for _ in 0 ..< 20 {
      try await Task.sleep(for: .milliseconds(25))
      if await client.activeBackgroundResponseId != nil { break }
    }

    // Verify the background response ID was captured
    let backgroundId = await client.activeBackgroundResponseId
    #expect(backgroundId == "resp_bg_123")

    // Call stop, which should send an authenticated cancel
    await client.stop()

    // Wait for the cancel request to be sent
    try await Task.sleep(for: .milliseconds(100))

    // Finish the stream so the task can clean up
    streamGate.continuation.finish()
    task.cancel()
    _ = try? await task.value

    // Verify the cancel request included the API key
    let authHeader = try #require(cancelRequest).value(forHTTPHeaderField: "Authorization")
    #expect(authHeader == "Bearer secret-api-key")
  }
}
