// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import os
import Testing

@Suite(.serialized)
struct TopLevelResponsesHelperTests {
  @Test
  func `Top level Responses helper carries explicit provider to custom replay capture`() async throws {
    var capturedBodyData: Data?
    let testId = UUID().uuidString
    let endpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    setTopLevelResponsesSessionOverride(makeMockSession())
    defer { setTopLevelResponsesSessionOverride(nil) }

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      return successSSE(for: request)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    _ = try await consumeStream(streamText(
      model: .responses("o3", endpoint: endpoint),
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-key",
      responsesProvider: .openAI,
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    let include = try #require(body?["include"] as? [String])
    #expect(include.contains("reasoning.encrypted_content"))
  }

  @Test
  func `Top level Responses helper matches direct client replay capture for equivalent custom endpoint config`() async throws {
    var topLevelBodyData: Data?
    var directClientBodyData: Data?

    let topLevelId = UUID().uuidString
    let directClientId = UUID().uuidString
    let topLevelEndpoint = try #require(URL(string: "https://mock.test/\(topLevelId)"))
    let directClientEndpoint = try #require(URL(string: "https://mock.test/\(directClientId)"))

    setTopLevelResponsesSessionOverride(makeMockSession())
    defer { setTopLevelResponsesSessionOverride(nil) }

    MockURLProtocol.setHandler(for: topLevelId) { request in
      topLevelBodyData = readRequestBody(from: request)
      return successSSE(for: request)
    }
    defer { MockURLProtocol.removeHandler(for: topLevelId) }

    MockURLProtocol.setHandler(for: directClientId) { request in
      directClientBodyData = readRequestBody(from: request)
      return successSSE(for: request)
    }
    defer { MockURLProtocol.removeHandler(for: directClientId) }

    _ = try await consumeStream(streamText(
      model: .responses("o3", endpoint: topLevelEndpoint),
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-key",
      responsesProvider: .openAI,
    ))

    let client = ResponsesClient(endpoint: directClientEndpoint, session: makeMockSession())
    _ = try await consumeStream(client.streamText(
      modelId: "o3",
      messages: [Message(role: .user, content: "Hello")],
      maxTokens: 1024,
      apiKey: "test-key",
      configuration: .init(provider: .openAI),
    ))

    let topLevelBody = try JSONSerialization.jsonObject(with: #require(topLevelBodyData)) as? [String: Any]
    let directClientBody = try JSONSerialization.jsonObject(with: #require(directClientBodyData)) as? [String: Any]

    #expect(topLevelBody?["include"] as? [String] == directClientBody?["include"] as? [String])
  }

  @Test
  func `Top level Responses helper warns and degrades replay capture for ambiguous custom endpoints with reasoning history`() async throws {
    var capturedBodyData: Data?
    let warnings = LockedWarnings()
    let testId = UUID().uuidString
    let endpoint = try #require(URL(string: "https://mock.test/\(testId)"))

    setResponsesReplayWarningObserver { warnings.append($0) }
    defer { setResponsesReplayWarningObserver(nil) }
    setTopLevelResponsesSessionOverride(makeMockSession())
    defer { setTopLevelResponsesSessionOverride(nil) }

    MockURLProtocol.setHandler(for: testId) { request in
      capturedBodyData = readRequestBody(from: request)
      return successSSE(for: request)
    }
    defer { MockURLProtocol.removeHandler(for: testId) }

    let messages = [
      Message(role: .assistant, content: [.thinking(text: "Let me think.", signature: nil), .text("Earlier answer.")]),
      Message(role: .user, content: "Continue."),
    ]

    _ = try await consumeStream(streamText(
      model: .responses("o3", endpoint: endpoint),
      messages: messages,
      maxTokens: 1024,
      apiKey: "test-key",
    ))

    let body = try JSONSerialization.jsonObject(with: #require(capturedBodyData)) as? [String: Any]
    #expect(body?["include"] == nil)
    #expect(warnings.messages.contains { $0.contains("missing responsesProvider") })
  }

  @Test
  func `Top level Responses helper rejects conflicting provider for built in endpoints`() async {
    do {
      _ = try await consumeStream(streamText(
        model: .responses("o3", endpoint: ResponsesClient.Endpoint.openAI.url),
        messages: [Message(role: .user, content: "Hello")],
        maxTokens: 1024,
        apiKey: "test-key",
        responsesProvider: .xAI,
      ))
      Issue.record("Expected conflicting provider for built-in Responses endpoint to throw")
    } catch let error as AIError {
      guard case let .invalidRequest(message) = error else {
        Issue.record("Expected invalidRequest but got \(error)")
        return
      }
      #expect(message.contains("conflicts"))
      #expect(message.contains("`.openAI`"))
    } catch {
      Issue.record("Expected AIError.invalidRequest but got \(error)")
    }
  }
}

private func successSSE(for request: URLRequest) -> (HTTPURLResponse, Data) {
  let response = HTTPURLResponse(
    url: request.url!,
    statusCode: 200,
    httpVersion: nil,
    headerFields: ["Content-Type": "text/event-stream"],
  )!
  let sseData = """
  data: {"type":"response.created","response":{"id":"test","status":"in_progress","model":"o3"}}

  data: {"type":"response.output_text.delta","delta":"Ok"}

  data: {"type":"response.completed","response":{"id":"test","status":"completed","model":"o3","created_at":1700000000,"output":[{"type":"message","content":[{"type":"output_text","text":"Ok"}]}],"usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}

  data: [DONE]

  """
  return (response, sseData.data(using: .utf8)!)
}

private final class LockedWarnings: Sendable {
  private let storage = OSAllocatedUnfairLock(initialState: [String]())

  func append(_ message: String) {
    storage.withLock { $0.append(message) }
  }

  var messages: [String] {
    storage.withLock { Array($0) }
  }
}
