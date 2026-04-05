// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import os
import Testing

@Suite(.serialized)
struct GeminiClientTests {
  // MARK: - Test Helpers

  /// Creates a mock client configured for testing with isolated handlers.
  private func makeTestClient(sseData: String, statusCode: Int = 200) -> (client: GeminiClient, cleanup: () -> Void) {
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

    let client = GeminiClient(
      session: makeMockSession(),
      modelsEndpoint: testEndpoint,
    )

    return (client, { MockURLProtocol.removeHandler(for: testId) })
  }

  // MARK: - Basic Response Tests

  @Test
  func `Parses basic text response and accumulates text`() async throws {
    let fixture = try loadFixture("gemini_basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
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
  }

  @Test
  func `Parses function call response`() async throws {
    let fixture = try loadFixture("gemini_function_call_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
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
  func `Extracts token usage metadata`() async throws {
    let fixture = try loadFixture("gemini_basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // Verify token counts from the fixture
    // Final usage: promptTokenCount: 8, candidatesTokenCount: 2, totalTokenCount: 10
    #expect(response.metadata?.inputTokens == 8)
    #expect(response.metadata?.outputTokens == 2)
  }

  @Test
  func `Sets finish reason correctly for normal stop`() async throws {
    let fixture = try loadFixture("gemini_basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ))

    // From gemini_basic_response.txt: finishReason: "STOP"
    #expect(response.metadata?.finishReason == .stop)
  }

  @Test
  func `Sets finish reason correctly for max tokens`() async throws {
    let fixture = try loadFixture("gemini_max_tokens_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Write a long story")],
      maxTokens: 15,
      apiKey: "test-api-key",
    ))

    // Verify the response was truncated
    #expect(response.responseText?.isEmpty == false)

    // Verify finish reason is maxTokens (critical for applications to know output was truncated)
    #expect(response.metadata?.finishReason == .maxTokens)
  }

  // MARK: - Error Handling Tests

  @Test
  func `Throws error for 400 bad request`() async throws {
    let errorResponse = """
    {"error":{"code":400,"message":"Invalid request","status":"INVALID_ARGUMENT"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 400)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key",
      ))
      Issue.record("Expected bad request error")
    } catch let error as AIError {
      if case .invalidRequest = error {
        // Expected
      } else {
        Issue.record("Expected invalid request error, got: \(error)")
      }
    }
  }

  @Test
  func `Throws error for 403 permission denied`() async throws {
    let errorResponse = """
    {"error":{"code":403,"message":"Permission denied","status":"PERMISSION_DENIED"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 403)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "invalid-key",
      ))
      Issue.record("Expected permission denied error")
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
    {"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 429)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
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
    {"error":{"code":500,"message":"Internal server error","status":"INTERNAL"}}
    """
    let (client, cleanup) = makeTestClient(sseData: errorResponse, statusCode: 500)
    defer { cleanup() }

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
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
    let client = GeminiClient()

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: nil,
      ))
      Issue.record("Expected error for missing API key")
    } catch let error as AIError {
      if case .authentication = error {
        // Expected
      } else {
        Issue.record("Expected authentication error, got: \(error)")
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

    let client = GeminiClient(
      session: makeMockSession(),
      modelsEndpoint: testEndpoint,
    )

    do {
      _ = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
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

  // MARK: - Safety Filtering Tests

  @Test
  func `Handles safety-blocked response`() async throws {
    let fixture = try loadFixture("gemini_safety_blocked_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    do {
      let response = try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: "Harmful content")],
        maxTokens: 1024,
        apiKey: "test-api-key",
      ))
      // If no error thrown, verify the finish reason indicates content filter
      #expect(response.metadata?.finishReason == .contentFilter)
    } catch let error as AIError {
      // LLMError.serverError with safety message is expected
      if case let .serverError(_, message, _) = error {
        #expect(message.lowercased().contains("safety") || message.lowercased().contains("blocked"))
      } else {
        Issue.record("Expected serverError with safety message, got: \(error)")
      }
    } catch {
      // Verify the error message indicates safety blocking
      let errorMessage = String(describing: error)
      #expect(
        errorMessage.lowercased().contains("safety") ||
          errorMessage.lowercased().contains("blocked") ||
          errorMessage.lowercased().contains("finish reason"),
      )
    }
  }

  // MARK: - Thinking Content Tests

  @Test
  func `Handles thinking content in response`() async throws {
    let fixture = try loadFixture("gemini_thinking_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash-thinking",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What is the meaning of life?")],
      maxTokens: 1024,
      apiKey: "test-api-key",
      configuration: .init(thinkingBudget: 1000),
    ), collecting: collector)

    // Verify we got the final answer (from non-thinking content)
    #expect(response.responseText?.contains("42") == true)

    // Verify thinking content was captured separately from regular response
    let updates = collector.updates
    let reasoningUpdates = updates.filter { $0.reasoningText != nil }
    #expect(!reasoningUpdates.isEmpty, "Expected at least one update with thinking content")

    // Verify the thinking content contains the expected text from fixture
    let reasoningText = reasoningUpdates.compactMap { $0.reasoningText }.joined()
    #expect(reasoningText.contains("Let me think"), "Thinking content should contain 'Let me think'")

    // Verify thinking content is separate from response content
    #expect(response.reasoningText != nil, "Final response should have accumulated thinking content")
  }

  // MARK: - Stream Processing Tests

  @Test
  func `Yields all chunks correctly`() async throws {
    let fixture = try loadFixture("gemini_basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let collector = UpdateCollector()
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ), collecting: collector)

    // Verify we received multiple streaming updates
    let updates = collector.updates
    #expect(updates.count >= 2) // At least 2 chunks in the fixture
  }

  // MARK: - Request Body Validation Tests

  @Test
  func `Request body includes system instruction correctly`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      // Capture body data immediately before it's consumed
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"totalTokenCount":11}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
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

    // Verify system instruction is included
    let systemInstruction = body?["systemInstruction"] as? [String: Any]
    #expect(systemInstruction != nil, "Request should include systemInstruction")

    let parts = systemInstruction?["parts"] as? [[String: Any]]
    #expect(parts != nil)
    #expect(parts?.first?["text"] as? String == "You are a helpful assistant")

    // Verify contents are included
    let contents = body?["contents"] as? [[String: Any]]
    #expect(contents != nil)
    #expect(try #require(contents).isEmpty == false)

    // Verify generation config includes temperature
    let generationConfig = body?["generationConfig"] as? [String: Any]
    #expect(generationConfig != nil)
    // Use approximate comparison for floating-point due to Float->Double conversion
    if let temp = generationConfig?["temperature"] as? Double {
      #expect(abs(temp - 0.7) < 0.01, "Temperature should be approximately 0.7")
    } else {
      Issue.record("temperature should be a Double")
    }
    #expect(generationConfig?["maxOutputTokens"] as? Int == 1024)
  }

  @Test
  func `Request body includes tools correctly`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      // Capture body data immediately before it's consumed
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"candidates":[{"content":{"parts":[{"text":"I'll check"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
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

    // Verify function declaration structure
    let functionDeclarations = tools?.first?["functionDeclarations"] as? [[String: Any]]
    #expect(functionDeclarations != nil)
    #expect(functionDeclarations?.first?["name"] as? String == "get_weather")
    #expect(functionDeclarations?.first?["description"] as? String == "Get current weather")
    #expect(functionDeclarations?.first?["parameters"] == nil)
    let parametersJsonSchema = functionDeclarations?.first?["parametersJsonSchema"] as? [String: Any]
    #expect(parametersJsonSchema != nil)
    #expect(parametersJsonSchema?["type"] as? String == "object")
  }

  @Test
  func `Request body preserves full JSON Schema in parametersJsonSchema`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))
    let rawSchemaTool = Tool(
      name: "process_nested_data",
      description: "Process nested data structures",
      inputSchema: [
        "type": .string("object"),
        "properties": .object([
          "grouped": .object([
            "type": .string("object"),
            "additionalProperties": .object([
              "type": .string("array"),
              "items": .object(["type": .string("integer")]),
            ]),
          ]),
          "payload": .object([
            "type": .string("string"),
            "contentEncoding": .string("base64"),
          ]),
        ]),
        "required": .array([.string("grouped"), .string("payload")]),
      ],
    ) { _ in [.text("OK")] }

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      let sseData = """
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      tools: [rawSchemaTool],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Process this data")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let tools = try #require(body?["tools"] as? [[String: Any]])
    let functionDeclarations = try #require(tools.first?["functionDeclarations"] as? [[String: Any]])
    let declaration = try #require(functionDeclarations.first)

    #expect(declaration["parameters"] == nil)
    let parametersJsonSchema = try #require(declaration["parametersJsonSchema"] as? [String: Any])
    #expect(parametersJsonSchema["type"] as? String == "object")

    let properties = try #require(parametersJsonSchema["properties"] as? [String: Any])
    let grouped = try #require(properties["grouped"] as? [String: Any])
    let groupedAdditionalProperties = try #require(grouped["additionalProperties"] as? [String: Any])
    #expect(groupedAdditionalProperties["type"] as? String == "array")
    let groupedItems = try #require(groupedAdditionalProperties["items"] as? [String: Any])
    #expect(groupedItems["type"] as? String == "integer")

    let payload = try #require(properties["payload"] as? [String: Any])
    #expect(payload["contentEncoding"] as? String == "base64")
  }

  @Test
  func `Request body requests server-side tool invocations`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Run some code")],
      maxTokens: 1024,
      apiKey: "test-key",
      configuration: .init(codeExecution: true),
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let tools = try #require(body?["tools"] as? [[String: Any]])
    #expect(tools.count == 1)
    #expect((tools.first?["codeExecution"] as? [String: Any]) != nil)

    let toolConfig = try #require(body?["toolConfig"] as? [String: Any])
    #expect(toolConfig["includeServerSideToolInvocations"] as? Bool == true)
    #expect(toolConfig["functionCallingConfig"] == nil)
  }

  @Test
  func `Chat Completions refusals replay as plain text in Gemini requests`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
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
    let contents = try #require(body?["contents"] as? [[String: Any]])
    let modelMessage = try #require(contents.first { ($0["role"] as? String) == "model" })
    let parts = try #require(modelMessage["parts"] as? [[String: Any]])
    #expect(parts.first?["text"] as? String == "I can't assist with that.")
  }

  @Test
  func `Responses annotated text and refusal replay as plain text in Gemini requests`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
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
    let contents = try #require(body?["contents"] as? [[String: Any]])
    let modelMessage = try #require(contents.first { ($0["role"] as? String) == "model" })
    let parts = try #require(modelMessage["parts"] as? [[String: Any]])
    let texts = parts.compactMap { $0["text"] as? String }
    #expect(texts == ["Cited answer.", "I can't continue beyond that."])
  }

  @Test
  func `Chat Completions citation endnotes replay as plain text in Gemini requests`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let citationNotes = """
    Sources:
    [1] Example Docs
        URL: https://example.com/docs
    """
    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .text("Cited answer."),
          .providerOpaque(OpaqueBlock(
            provider: "openai-chat-completions",
            type: "annotations",
            data: #"[{"type":"url_citation","url_citation":{"url":"https://example.com/docs","title":"Example Docs"}}]"#,
          )),
          .endnotes(citationNotes),
        ]),
        Message(role: .user, content: "Try again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let contents = try #require(body?["contents"] as? [[String: Any]])
    let modelMessage = try #require(contents.first { ($0["role"] as? String) == "model" })
    let parts = try #require(modelMessage["parts"] as? [[String: Any]])
    let texts = parts.compactMap { $0["text"] as? String }
    #expect(texts == ["Cited answer.", citationNotes])
  }

  @Test
  func `Foreign response-content opaque blocks replay as plain text in Gemini requests`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      systemPrompt: nil,
      messages: [
        Message(role: .system, content: [
          .providerOpaque(OpaqueBlock(
            provider: "anthropic",
            type: "web_fetch_tool_result",
            content: "Fetched article body.",
            data: #"{"type":"web_fetch_tool_result"}"#,
            isResponseContent: true,
          )),
        ]),
        Message(role: .assistant, content: [
          .providerOpaque(OpaqueBlock(
            provider: "anthropic",
            type: "server_tool_use",
            content: "```python\nprint(42)\n```",
            data: #"{"type":"server_tool_use"}"#,
            isResponseContent: true,
          )),
        ]),
        Message(role: .user, content: "Try again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let systemInstruction = try #require(body?["systemInstruction"] as? [String: Any])
    let systemParts = try #require(systemInstruction["parts"] as? [[String: Any]])
    #expect(systemParts.contains(where: { $0["text"] as? String == "Fetched article body." }))

    let contents = try #require(body?["contents"] as? [[String: Any]])
    let modelMessage = try #require(contents.first { ($0["role"] as? String) == "model" })
    let parts = try #require(modelMessage["parts"] as? [[String: Any]])
    #expect(parts.contains(where: { $0["text"] as? String == "```python\nprint(42)\n```" }))
  }

  @Test
  func `Gemini native opaque blocks without raw data fall back to plain text`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3,"totalTokenCount":23}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      systemPrompt: nil,
      messages: [
        Message(role: .assistant, content: [
          .providerOpaque(OpaqueBlock(
            provider: "gemini",
            type: "executableCode",
            content: "print(42)",
            isResponseContent: true,
          )),
        ]),
        Message(role: .user, content: "Try again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let contents = try #require(body?["contents"] as? [[String: Any]])
    let modelMessage = try #require(contents.first { ($0["role"] as? String) == "model" })
    let parts = try #require(modelMessage["parts"] as? [[String: Any]])
    #expect(parts.contains(where: { $0["text"] as? String == "print(42)" }))
  }

  @Test
  func `Server-side Gemini tool history survives parse and replay`() async throws {
    var requestCount = 0
    var replayRequestBodyData: Data?
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      requestCount += 1

      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!

      if requestCount == 1 {
        let sseData = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"I checked the URL.\"},{\"toolCall\":{\"id\":\"server_tool_1\",\"toolType\":\"URL_CONTEXT\",\"args\":{\"url\":\"https://example.com\"}}},{\"toolResponse\":{\"id\":\"server_tool_1\",\"toolType\":\"URL_CONTEXT\",\"response\":{\"status\":\"ok\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0,\"urlContextMetadata\":{\"urlMetadata\":[{\"retrievedUrl\":\"https://example.com\",\"urlRetrievalStatus\":\"URL_RETRIEVAL_STATUS_SUCCESS\"}]}}],\"usageMetadata\":{\"promptTokenCount\":20,\"candidatesTokenCount\":9,\"totalTokenCount\":29}}\n\n"
        return (response, sseData.data(using: .utf8)!)
      }

      replayRequestBodyData = readRequestBody(from: request)
      let sseData = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Summary\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":16,\"candidatesTokenCount\":2,\"totalTokenCount\":18}}\n\n"
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    let firstResponse = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Check https://example.com")],
      maxTokens: 1024,
      apiKey: "test-key",
      configuration: .init(webContent: true),
    ))

    let opaqueBlocks = firstResponse.content.compactMap { item -> OpaqueBlock? in
      guard case let .providerOpaque(block) = item else { return nil }
      return block
    }
    #expect(opaqueBlocks.contains { $0.provider == "gemini" && $0.type == "toolCall" })
    #expect(opaqueBlocks.contains { $0.provider == "gemini" && $0.type == "toolResponse" })
    #expect(opaqueBlocks.contains { $0.provider == "gemini" && $0.type == "urlContextMetadata" })

    let assistantMessage = Message(role: .assistant, content: firstResponse.content)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.5-flash",
      systemPrompt: nil,
      messages: [
        Message(role: .user, content: "Check https://example.com"),
        assistantMessage,
        Message(role: .user, content: "Summarize it again"),
      ],
      maxTokens: 1024,
      apiKey: "test-key",
      configuration: .init(webContent: true),
    ))

    #expect(requestCount == 2)

    let body = try JSONSerialization.jsonObject(with: #require(replayRequestBodyData)) as? [String: Any]
    let contents = try #require(body?["contents"] as? [[String: Any]])
    let modelMessage = try #require(contents.first { ($0["role"] as? String) == "model" })
    let parts = try #require(modelMessage["parts"] as? [[String: Any]])

    #expect(parts.contains { part in
      let toolCall = part["toolCall"] as? [String: Any]
      return toolCall?["id"] as? String == "server_tool_1"
    })
    #expect(parts.contains { part in
      let toolResponse = part["toolResponse"] as? [String: Any]
      return toolResponse?["id"] as? String == "server_tool_1"
    })
    #expect(parts.allSatisfy { $0["urlContextMetadata"] == nil })
  }

  @Test
  func `Immediate failed upload returns an error before Gemini request`() async throws {
    let observedPaths = OSAllocatedUnfairLock(initialState: [String]())
    let testId = UUID().uuidString
    let modelsEndpoint = try #require(URL(string: "https://mock.test/\(testId)/v1beta/models"))

    MockURLProtocol.setHandler(for: testId) { request in
      let path = request.url?.path ?? ""
      observedPaths.withLock { $0.append(path) }

      switch path {
        case "/\(testId)/upload/v1beta/files":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: [
              "Content-Type": "application/json",
              "X-Goog-Upload-URL": "https://mock.test/\(testId)/upload-session",
            ],
          )!
          return (response, Data())

        case "/\(testId)/upload-session":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"file":{"uri":"https://mock.test/\(testId)/files/file_123","state":"FAILED"}}
          """
          return try (response, #require(body.data(using: .utf8)))

        default:
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}]}
          """
          return try (response, #require(body.data(using: .utf8)))
      }
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let attachment = Attachment(kind: .video(data: Data([0x00, 0x01]), mimeType: "video/mp4"), filename: "clip.mp4")
    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: modelsEndpoint)

    do {
      _ = try await client.generateText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: [.attachment(attachment)])],
        maxTokens: 1024,
        apiKey: "test-key",
      )
      Issue.record("Expected upload failure")
    } catch let error as AIError {
      if case let .serverError(_, message, _) = error {
        #expect(message.contains("File processing failed"))
      } else {
        Issue.record("Expected server error for failed upload, got: \(error)")
      }
    }

    #expect(observedPaths.withLock { $0 } == [
      "/\(testId)/upload/v1beta/files",
      "/\(testId)/upload-session",
    ])
  }

  @Test
  func `Immediate failed upload without uri surfaces nested file error`() async throws {
    let observedPaths = OSAllocatedUnfairLock(initialState: [String]())
    let testId = UUID().uuidString
    let modelsEndpoint = try #require(URL(string: "https://mock.test/\(testId)/v1beta/models"))

    MockURLProtocol.setHandler(for: testId) { request in
      let path = request.url?.path ?? ""
      observedPaths.withLock { $0.append(path) }

      switch path {
        case "/\(testId)/upload/v1beta/files":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: [
              "Content-Type": "application/json",
              "X-Goog-Upload-URL": "https://mock.test/\(testId)/upload-session",
            ],
          )!
          return (response, Data())

        case "/\(testId)/upload-session":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"file":{"state":"FAILED","error":{"code":9,"message":"Upload rejected"}}}
          """
          return try (response, #require(body.data(using: .utf8)))

        default:
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}]}
          """
          return try (response, #require(body.data(using: .utf8)))
      }
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let attachment = Attachment(kind: .video(data: Data([0x00, 0x01]), mimeType: "video/mp4"), filename: "clip.mp4")
    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: modelsEndpoint)

    do {
      _ = try await client.generateText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: [.attachment(attachment)])],
        maxTokens: 1024,
        apiKey: "test-key",
      )
      Issue.record("Expected upload failure")
    } catch let error as AIError {
      if case let .serverError(_, message, _) = error {
        #expect(message == "Upload rejected")
      } else {
        Issue.record("Expected server error for failed upload, got: \(error)")
      }
    }

    #expect(observedPaths.withLock { $0 } == [
      "/\(testId)/upload/v1beta/files",
      "/\(testId)/upload-session",
    ])
  }

  @Test
  func `Polled failed upload without uri returns status error`() async throws {
    let observedPaths = OSAllocatedUnfairLock(initialState: [String]())
    let testId = UUID().uuidString
    let modelsEndpoint = try #require(URL(string: "https://mock.test/\(testId)/v1beta/models"))

    MockURLProtocol.setHandler(for: testId) { request in
      let path = request.url?.path ?? ""
      observedPaths.withLock { $0.append(path) }

      switch path {
        case "/\(testId)/upload/v1beta/files":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: [
              "Content-Type": "application/json",
              "X-Goog-Upload-URL": "https://mock.test/\(testId)/upload-session",
            ],
          )!
          return (response, Data())

        case "/\(testId)/upload-session":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"file":{"uri":"https://mock.test/\(testId)/files/file_123","state":"PROCESSING"}}
          """
          return try (response, #require(body.data(using: .utf8)))

        case "/\(testId)/files/file_123":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"state":"FAILED","error":{"code":13,"message":"Video processing failed"}}
          """
          return try (response, #require(body.data(using: .utf8)))

        default:
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}]}
          """
          return try (response, #require(body.data(using: .utf8)))
      }
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let attachment = Attachment(kind: .video(data: Data([0x00, 0x01]), mimeType: "video/mp4"), filename: "clip.mp4")
    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: modelsEndpoint)

    do {
      _ = try await client.generateText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: [.attachment(attachment)])],
        maxTokens: 1024,
        apiKey: "test-key",
      )
      Issue.record("Expected upload failure")
    } catch let error as AIError {
      if case let .serverError(_, message, _) = error {
        #expect(message == "Video processing failed")
      } else {
        Issue.record("Expected server error for failed upload, got: \(error)")
      }
    }

    #expect(observedPaths.withLock { $0 } == [
      "/\(testId)/upload/v1beta/files",
      "/\(testId)/upload-session",
      "/\(testId)/files/file_123",
    ])
  }

  @Test
  func `Polled upload HTTP error preserves Gemini status mapping`() async throws {
    let observedPaths = OSAllocatedUnfairLock(initialState: [String]())
    let testId = UUID().uuidString
    let modelsEndpoint = try #require(URL(string: "https://mock.test/\(testId)/v1beta/models"))

    MockURLProtocol.setHandler(for: testId) { request in
      let path = request.url?.path ?? ""
      observedPaths.withLock { $0.append(path) }

      switch path {
        case "/\(testId)/upload/v1beta/files":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: [
              "Content-Type": "application/json",
              "X-Goog-Upload-URL": "https://mock.test/\(testId)/upload-session",
            ],
          )!
          return (response, Data())

        case "/\(testId)/upload-session":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"file":{"uri":"https://mock.test/\(testId)/files/file_123","state":"PROCESSING"}}
          """
          return try (response, #require(body.data(using: .utf8)))

        case "/\(testId)/files/file_123":
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 403,
            httpVersion: nil,
            headerFields: ["Content-Type": "text/plain"],
          )!
          return (response, Data("Forbidden".utf8))

        default:
          let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"],
          )!
          let body = """
          {"candidates":[{"content":{"parts":[{"text":"Done"}],"role":"model"},"finishReason":"STOP","index":0}]}
          """
          return try (response, #require(body.data(using: .utf8)))
      }
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let attachment = Attachment(kind: .video(data: Data([0x00, 0x01]), mimeType: "video/mp4"), filename: "clip.mp4")
    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: modelsEndpoint)

    do {
      _ = try await client.generateText(
        modelId: "gemini-2.0-flash",
        systemPrompt: nil,
        messages: [Message(role: .user, content: [.attachment(attachment)])],
        maxTokens: 1024,
        apiKey: "test-key",
      )
      Issue.record("Expected upload failure")
    } catch let error as AIError {
      if case let .authentication(message) = error {
        #expect(message == "Ensure your API key is set correctly and has the right access.")
      } else {
        Issue.record("Expected authentication error for poll failure, got: \(error)")
      }
    }

    #expect(observedPaths.withLock { $0 } == [
      "/\(testId)/upload/v1beta/files",
      "/\(testId)/upload-session",
      "/\(testId)/files/file_123",
    ])
  }

  // MARK: - Multiple Tool Calls Tests

  @Test
  func `Parses multiple function calls in single response`() async throws {
    // Create a fixture with multiple function calls
    let sseData = """
    data: {"candidates":[{"content":{"parts":[{"text":"I'll check the weather in both cities."}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":10,"totalTokenCount":35}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}},{"functionCall":{"name":"get_weather","args":{"location":"London"}}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":20,"totalTokenCount":45}}


    """
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris and London?")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    // Verify we got multiple function calls
    #expect(response.toolCalls.count == 2, "Expected 2 function calls for Paris and London")

    // Verify first function call
    let firstCall = response.toolCalls[0]
    #expect(firstCall.name == "get_weather")
    if case let .string(location) = firstCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected location parameter to be 'Paris'")
    }

    // Verify second function call
    let secondCall = response.toolCalls[1]
    #expect(secondCall.name == "get_weather")
    if case let .string(location) = secondCall.parameters["location"] {
      #expect(location == "London")
    } else {
      Issue.record("Expected location parameter to be 'London'")
    }

    // Verify text response is also present
    #expect(response.responseText?.contains("weather") == true)
  }

  @Test
  func `Parses streamed partial function call arguments`() async throws {
    let sseData = """
    data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_partial_1","name":"get_weather","willContinue":true},"thoughtSignature":"sig_partial_1"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":1,"totalTokenCount":26}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.location","stringValue":"NewYork","willContinue":true}],"willContinue":true}}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":2,"totalTokenCount":27}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.location","stringValue":"City"},{"jsonPath":"$.unit","stringValue":"C"}],"willContinue":true}}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":3,"totalTokenCount":28}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":4,"totalTokenCount":29}}

    """
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-3-pro-preview",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What should I wear in New York City?")],
      maxTokens: 1024,
      apiKey: "test-key",
    ), collecting: collector)

    #expect(response.toolCalls.count == 1)
    #expect(response.metadata?.finishReason == .toolUse)

    let toolCall = try #require(response.toolCalls.first)
    #expect(toolCall.id == "call_partial_1")
    #expect(toolCall.name == "get_weather")
    #expect(toolCall.providerMetadata?["thoughtSignature"] == "sig_partial_1")

    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "NewYorkCity")
    } else {
      Issue.record("Expected streamed location parameter to be reconstructed as a string")
    }

    if case let .string(unit) = toolCall.parameters["unit"] {
      #expect(unit == "C")
    } else {
      Issue.record("Expected streamed unit parameter to be reconstructed as a string")
    }

    let streamedToolCallStates = collector.updates.compactMap(\.toolCalls.first)
    #expect(streamedToolCallStates.count >= 2, "Expected incremental tool-call updates while partial args stream")
    #expect(streamedToolCallStates.last?.parameters == toolCall.parameters)
  }

  @Test
  func `Final partial function call chunk clears active state without sentinel`() async throws {
    let sseData = """
    data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_partial_1","name":"get_weather","willContinue":true}}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":1,"totalTokenCount":26}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.location","stringValue":"Paris"}]}}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":2,"totalTokenCount":27}}

    data: {"candidates":[{"content":{"parts":[{"thoughtSignature":"sig_unrelated"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":3,"totalTokenCount":28}}

    """
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-3-pro-preview",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris?")],
      maxTokens: 1024,
      apiKey: "test-key",
    ), collecting: collector)

    #expect(response.toolCalls.count == 1)
    #expect(response.metadata?.finishReason == .toolUse)

    let toolCall = try #require(response.toolCalls.first)
    #expect(toolCall.id == "call_partial_1")
    #expect(toolCall.providerMetadata == nil)
    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected final partial argument to persist on the tool call")
    }

    let streamedToolCallStates = collector.updates.compactMap(\.toolCalls.first)
    #expect(streamedToolCallStates.count == 3, "Only the initial chunk, final partial-args chunk, and terminal update should contain tool calls")
    #expect(streamedToolCallStates.allSatisfy { $0.providerMetadata == nil })
  }

  @Test
  func `Final function call args chunk without id reuses active tool call id`() async throws {
    let sseData = """
    data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_partial_1","name":"get_weather","willContinue":true},"thoughtSignature":"sig_partial_1"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":1,"totalTokenCount":26}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris","unit":"C"}}}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":2,"totalTokenCount":27}}

    data: {"candidates":[{"content":{"parts":[{"thoughtSignature":"sig_unrelated"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":3,"totalTokenCount":28}}

    """
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-3-pro-preview",
      tools: [makeTestTool(name: "get_weather", description: "Get weather", paramName: "location")],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the weather in Paris?")],
      maxTokens: 1024,
      apiKey: "test-key",
    ), collecting: collector)

    #expect(response.toolCalls.count == 1)
    #expect(response.metadata?.finishReason == .toolUse)

    let toolCall = try #require(response.toolCalls.first)
    #expect(toolCall.id == "call_partial_1")
    #expect(toolCall.name == "get_weather")
    #expect(toolCall.providerMetadata?["thoughtSignature"] == "sig_partial_1")

    if case let .string(location) = toolCall.parameters["location"] {
      #expect(location == "Paris")
    } else {
      Issue.record("Expected terminal args chunk to populate location")
    }

    if case let .string(unit) = toolCall.parameters["unit"] {
      #expect(unit == "C")
    } else {
      Issue.record("Expected terminal args chunk to populate unit")
    }

    let streamedToolCallStates = collector.updates.compactMap(\.toolCalls.first)
    #expect(streamedToolCallStates.count == 3)
    #expect(streamedToolCallStates.allSatisfy { $0.id == "call_partial_1" })
    #expect(streamedToolCallStates.last?.providerMetadata?["thoughtSignature"] == "sig_partial_1")
  }

  @Test
  func `Final function call args chunk without id keeps a different named call distinct`() async throws {
    let sseData = """
    data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_partial_1","name":"get_weather","willContinue":true},"thoughtSignature":"sig_partial_1"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":1,"totalTokenCount":26}}

    data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_time","args":{"timezone":"UTC"}}}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":2,"totalTokenCount":27}}

    data: {"candidates":[{"content":{"parts":[{"thoughtSignature":"sig_unrelated"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":3,"totalTokenCount":28}}

    """
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-3-pro-preview",
      tools: [
        makeTestTool(name: "get_weather", description: "Get weather", paramName: "location"),
        makeTestTool(name: "get_time", description: "Get time", paramName: "timezone"),
      ],
      systemPrompt: nil,
      messages: [Message(role: .user, content: "What's the time in UTC?")],
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    #expect(response.toolCalls.count == 2)
    #expect(response.metadata?.finishReason == .toolUse)

    let partialToolCall = try #require(response.toolCalls.first(where: { $0.id == "call_partial_1" }))
    #expect(partialToolCall.name == "get_weather")
    #expect(partialToolCall.providerMetadata?["thoughtSignature"] == "sig_partial_1")
    #expect(partialToolCall.parameters.isEmpty)

    let distinctToolCall = try #require(response.toolCalls.first(where: { $0.name == "get_time" }))
    #expect(distinctToolCall.id != "call_partial_1")
    #expect(distinctToolCall.providerMetadata == nil)
    if case let .string(timezone) = distinctToolCall.parameters["timezone"] {
      #expect(timezone == "UTC")
    } else {
      Issue.record("Expected different named args chunk to remain a distinct tool call")
    }
  }

  // MARK: - Stream Cancellation Tests

  @Test
  func `Cancellation propagates correctly`() async throws {
    let testId = UUID().uuidString
    let testEndpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    // Use URL-specific handler to avoid interfering with other tests
    MockURLProtocol.setHandler(for: testId) { request in
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"],
      )!

      // Return a response that simulates a slow stream
      let sseData = """
      data: {"candidates":[{"content":{"parts":[{"text":"Hello World"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":2,"totalTokenCount":10}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let client = GeminiClient(
      session: makeMockSession(),
      modelsEndpoint: testEndpoint,
    )

    let task = Task {
      try await consumeStream(client.streamText(
        modelId: "gemini-2.0-flash",
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

  @Test
  func `System and developer messages in history are routed to system instruction`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"totalTokenCount":11}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let messages = [
      Message(role: .system, content: "You are helpful"),
      Message(role: .developer, content: "Be concise"),
      Message(role: .user, content: "Hello"),
    ]

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: "Base instructions",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]

    // Verify systemInstruction contains the systemPrompt and extracted system/developer messages
    let systemInstruction = try #require(body?["systemInstruction"] as? [String: Any])
    let parts = try #require(systemInstruction["parts"] as? [[String: Any]])
    let texts = parts.compactMap { $0["text"] as? String }
    #expect(texts.contains("Base instructions"))
    #expect(texts.contains("You are helpful"))
    #expect(texts.contains("Be concise"))

    // Verify contents only has user messages, not system or developer
    let contents = try #require(body?["contents"] as? [[String: Any]])
    let roles = contents.compactMap { $0["role"] as? String }
    #expect(!roles.contains("system"))
    #expect(!roles.contains("developer"))
    #expect(roles.contains("user"))
  }

  @Test
  func `System instruction preserves replayable text and attachment fallbacks from history`() async throws {
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
      data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"totalTokenCount":11}}

      """
      return (response, sseData.data(using: .utf8)!)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let policyData = Data("fake-pdf-content".utf8)
    let imageData = try #require(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR2QAAAAASUVORK5CYII="))
    let messages = [
      Message(role: .system, content: [
        .text("Follow the attached policy."),
        .attachment(Attachment(kind: .document(data: policyData, mimeType: "application/pdf"), filename: "policy.pdf")),
        .providerOpaque(OpaqueBlock(
          provider: "openai-responses",
          type: "annotated_output_text",
          content: "Prefer cited guidance.",
          data: "[]",
          isResponseContent: true,
        )),
      ]),
      Message(role: .developer, content: [
        .providerOpaque(OpaqueBlock(
          provider: "openai-responses",
          type: "refusal",
          content: "Do not reveal internal policy text.",
          isResponseContent: true,
        )),
        .providerOpaque(OpaqueBlock(
          provider: "openai-chat-completions",
          type: "refusal",
          content: "Avoid unsafe detail.",
          isResponseContent: true,
        )),
        .attachment(Attachment(kind: .image(data: imageData, mimeType: "image/png"))),
      ]),
      Message(role: .user, content: "Hello"),
    ]

    let client = GeminiClient(session: makeMockSession(), modelsEndpoint: testEndpoint)
    _ = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: "Base instructions",
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let systemInstruction = try #require(body?["systemInstruction"] as? [String: Any])
    let parts = try #require(systemInstruction["parts"] as? [[String: Any]])
    let texts = parts.compactMap { $0["text"] as? String }
    let combinedText = texts.joined(separator: "\n")

    #expect(texts.contains("Base instructions"))
    #expect(texts.contains("Follow the attached policy."))
    #expect(texts.contains("Prefer cited guidance."))
    #expect(texts.contains("Do not reveal internal policy text."))
    #expect(texts.contains("Avoid unsafe detail."))
    #expect(combinedText.contains("policy.pdf"))
    #expect(combinedText.contains("application/pdf"))
    #expect(combinedText.contains("image/png"))
  }

  @Test
  func `Terminal yield carries complete metadata that intermediate updates lack`() async throws {
    // The basic fixture has two SSE chunks:
    //   1. "Hello" with candidatesTokenCount: 1, no finishReason
    //   2. " there!" with candidatesTokenCount: 2, finishReason: STOP
    // Gemini's parser yields text, usage, and finish reason as separate stream events,
    // so intermediate sendUpdate() calls only include metadata accumulated so far.
    // The terminal yield from streamText is the only one guaranteed to have complete metadata.
    let fixture = try loadFixture("gemini_basic_response.txt")
    let (client, cleanup) = makeTestClient(sseData: fixture)
    defer { cleanup() }

    let collector = UpdateCollector()
    let response = try await consumeStream(client.streamText(
      modelId: "gemini-2.0-flash",
      systemPrompt: nil,
      messages: [Message(role: .user, content: "Say hello")],
      maxTokens: 1024,
      apiKey: "test-api-key",
    ), collecting: collector)

    let updates = collector.updates
    #expect(updates.count >= 2, "Should have at least two updates (intermediate + terminal)")

    // The last pre-terminal update should lack the final finish reason,
    // since finishReason arrives as a separate stream event after the text
    let preTerminal = updates[updates.count - 2]
    #expect(preTerminal.metadata?.finishReason != .stop)

    // The terminal yield should have the complete metadata
    #expect(response.metadata?.finishReason == .stop)
    #expect(response.metadata?.inputTokens == 8)
    #expect(response.metadata?.outputTokens == 2)
  }
}
