// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

@Suite("SSE Parser", .serialized)
struct SSEParserTests {
  // MARK: - Test Helpers

  /// Collects all payloads from the SSE parser using mock data.
  /// Uses URL-based isolation to prevent conflicts with concurrent tests.
  private func collectPayloads(from sseData: String, terminateOnDone: Bool = true) async throws -> [String] {
    // Use a unique test ID for isolation
    let testId = UUID().uuidString

    MockURLProtocol.setHandler(for: testId) { _ in
      let response = HTTPURLResponse(
        url: URL(string: "https://mock.test/\(testId)")!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
      )!
      return (response, sseData.data(using: .utf8)!)
    }

    defer {
      MockURLProtocol.removeHandler(for: testId)
    }

    let session = makeMockSession()
    let url = URL(string: "https://mock.test/\(testId)")!
    let (bytes, _) = try await session.bytes(from: url)

    var payloads: [String] = []
    for try await payload in SSEParser.dataPayloads(from: bytes, terminateOnDone: terminateOnDone) {
      payloads.append(payload)
    }
    return payloads
  }

  // MARK: - Basic Parsing Tests

  @Test("Parses basic data payload")
  func basicDataPayload() async throws {
    let sseData = """
    data: {"foo":true}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "{\"foo\":true}")
  }

  @Test("Parses event with data")
  func eventWithData() async throws {
    let sseData = """
    event: completion
    data: {"foo":true}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "{\"foo\":true}")
  }

  @Test("Skips event without data")
  func eventWithoutData() async throws {
    let sseData = """
    event: ping

    data: {"foo":true}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "{\"foo\":true}")
  }

  @Test("Parses multiple events with data")
  func multipleEventsWithData() async throws {
    let sseData = """
    event: foo
    data: {"foo":true}

    event: bar
    data: {"bar":false}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 2)
    #expect(payloads[0] == "{\"foo\":true}")
    #expect(payloads[1] == "{\"bar\":false}")
  }

  @Test("Skips comment lines")
  func skipsComments() async throws {
    let sseData = """
    : this is a comment
    data: {"foo":true}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "{\"foo\":true}")
  }

  @Test("Handles empty data payload")
  func emptyDataPayload() async throws {
    // Our parser requires "data: " prefix (with space)
    // An empty payload would be "data: " followed by nothing
    let sseData = "data: \n\n"

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "")
  }

  // MARK: - [DONE] Termination Tests

  @Test("Terminates on [DONE] by default")
  func terminatesOnDone() async throws {
    let sseData = """
    data: {"first":true}

    data: [DONE]

    data: {"after_done":true}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "{\"first\":true}")
  }

  @Test("Continues past [DONE] when terminateOnDone is false")
  func continuesPastDone() async throws {
    let sseData = """
    data: {"first":true}

    data: [DONE]

    data: {"after_done":true}

    """

    let payloads = try await collectPayloads(from: sseData, terminateOnDone: false)

    #expect(payloads.count == 3)
    #expect(payloads[0] == "{\"first\":true}")
    #expect(payloads[1] == "[DONE]")
    #expect(payloads[2] == "{\"after_done\":true}")
  }

  // MARK: - JSON Content Tests

  @Test("Handles JSON with escaped newlines")
  func jsonWithEscapedNewlines() async throws {
    let sseData = """
    data: {"content":"line1\\nline2"}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)

    // Verify the JSON can be parsed and contains escaped newlines
    let data = try #require(payloads[0].data(using: .utf8))
    let json = try JSONDecoder().decode([String: String].self, from: data)
    #expect(json["content"] == "line1\nline2")
  }

  @Test("Handles JSON with unicode characters")
  func jsonWithUnicode() async throws {
    let sseData = """
    data: {"content":"Hello 👋 World 🌍"}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)

    let data = try #require(payloads[0].data(using: .utf8))
    let json = try JSONDecoder().decode([String: String].self, from: data)
    #expect(json["content"] == "Hello 👋 World 🌍")
  }

  @Test("Handles JSON with Cyrillic characters")
  func jsonWithCyrillic() async throws {
    let sseData = """
    data: {"content":"известни"}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)

    let data = try #require(payloads[0].data(using: .utf8))
    let json = try JSONDecoder().decode([String: String].self, from: data)
    #expect(json["content"] == "известни")
  }

  // MARK: - Anthropic-Style Events Tests

  @Test("Parses Anthropic message_start event")
  func anthropicMessageStart() async throws {
    let sseData = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_123","role":"assistant"}}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)

    let data = try #require(payloads[0].data(using: .utf8))
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(json["type"] as? String == "message_start")
  }

  @Test("Parses Anthropic content_block_delta event")
  func anthropicContentBlockDelta() async throws {
    let sseData = """
    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)

    let data = try #require(payloads[0].data(using: .utf8))
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(json["type"] as? String == "content_block_delta")

    let delta = try #require(json["delta"] as? [String: Any])
    #expect(delta["text"] as? String == "Hello")
  }

  @Test("Skips ping events")
  func skipsPingEvents() async throws {
    let sseData = """
    event: content_block_delta
    data: {"type":"content_block_delta","text":"Hello"}

    event: ping
    data: {"type":"ping"}

    event: content_block_delta
    data: {"type":"content_block_delta","text":"World"}

    """

    let payloads = try await collectPayloads(from: sseData)

    // All data payloads are returned, including ping
    // The caller is responsible for filtering event types
    #expect(payloads.count == 3)
  }

  // MARK: - Edge Cases

  @Test("Handles data with colon in value")
  func dataWithColonInValue() async throws {
    let sseData = """
    data: {"url":"https://example.com"}

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.count == 1)
    #expect(payloads[0] == "{\"url\":\"https://example.com\"}")
  }

  @Test("Handles empty stream")
  func emptyStream() async throws {
    let sseData = ""

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.isEmpty)
  }

  @Test("Handles stream with only comments and events")
  func onlyCommentsAndEvents() async throws {
    let sseData = """
    : comment
    event: ping

    : another comment

    """

    let payloads = try await collectPayloads(from: sseData)

    #expect(payloads.isEmpty)
  }
}
