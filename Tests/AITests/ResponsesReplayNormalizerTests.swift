// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ResponsesReplayNormalizerTests {
  @Test
  func `Responses normalizer emits request ready items for mixed tool history`() async throws {
    let messages = ReplayFixtures.mixedToolTurnHistory()

    let plan = try await ResponsesReplayNormalizer.normalize(messages)
    let functionCallOutputs = plan.inputItems.filter { $0["type"] as? String == "function_call_output" }
    #expect(functionCallOutputs.count == 2)

    let matchedOutput = try #require(functionCallOutputs.first { $0["call_id"] as? String == "call_1" })
    #expect(matchedOutput["output"] as? String == ReplayFixtures.matchedToolResultText)

    let syntheticOutput = try #require(functionCallOutputs.first { $0["call_id"] as? String == "call_2" })
    let syntheticError = try #require(syntheticOutput["output"] as? String)
    #expect(syntheticError.contains(ToolReplaySupport.syntheticToolResultErrorText))

    let collapsedStrayMessage = try #require(plan.inputItems.first(where: { item in
      guard item["type"] as? String == "message", item["role"] as? String == "user" else { return false }
      let content = item["content"] as? [[String: Any]]
      return content?.contains(where: { ($0["text"] as? String)?.contains(ReplayFixtures.strayToolResultText) == true }) == true
    }))
    #expect(collapsedStrayMessage["id"] == nil)
  }

  @Test
  func `Responses normalizer emits rich error as multi-part output with sentinel`() async throws {
    // Per spec §G: errors with non-text content emit the multi-part shape
    // with a "Tool call failed:" sentinel prepended, since Responses has no
    // out-of-band isError field on function_call_output.
    let imageData = Data([0x89, 0x50, 0x4E, 0x47]) // PNG magic
    let errorResult = ToolResult(
      name: "render",
      id: "call_err",
      content: [
        .text("rendering failed"),
        .image(imageData, mimeType: "image/png"),
      ],
      isError: true,
    )
    let messages = [
      Message(role: .user, content: "Render a chart"),
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "render", id: "call_err", parameters: [:])),
      ]),
      Message(role: .tool, content: [.toolResult(errorResult)]),
    ]

    let plan = try await ResponsesReplayNormalizer.normalize(messages)
    let output = try #require(plan.inputItems.first { $0["type"] as? String == "function_call_output" })
    let parts = try #require(output["output"] as? [[String: Any]])
    #expect(parts.count == 3) // sentinel + text + image
    #expect(parts[0]["type"] as? String == "input_text")
    #expect(parts[0]["text"] as? String == "Tool call failed:")
    #expect(parts[1]["type"] as? String == "input_text")
    #expect(parts[1]["text"] as? String == "rendering failed")
    #expect(parts[2]["type"] as? String == "input_image")
  }

  @Test
  func `Responses normalizer emits text only error as JSON string wrap without sentinel`() async throws {
    // Per spec §G: pure-text errors keep today's {"error": text} JSON-string
    // wrap so OpenAI's models recognize the failure signal. The sentinel
    // multi-part shape is reserved for rich errors.
    let errorResult = ToolResult(
      name: "lookup",
      id: "call_text",
      content: [.text("not found")],
      isError: true,
    )
    let messages = [
      Message(role: .user, content: "Look something up"),
      Message(role: .assistant, content: [
        .toolCall(ToolCall(name: "lookup", id: "call_text", parameters: [:])),
      ]),
      Message(role: .tool, content: [.toolResult(errorResult)]),
    ]

    let plan = try await ResponsesReplayNormalizer.normalize(messages)
    let output = try #require(plan.inputItems.first { $0["type"] as? String == "function_call_output" })
    let outputString = try #require(output["output"] as? String)
    #expect(outputString == "{\"error\":\"not found\"}")
  }

  @Test
  func `Responses normalizer preserves assistant role when output metadata falls back to input content`() async throws {
    let attachment = Attachment(
      kind: .document(data: Data("notes".utf8), mimeType: "application/pdf"),
      filename: "notes.pdf",
    )
    let messages = [
      Message(role: .assistant, content: [
        .providerOpaque(OpaqueBlock(
          provider: "openai-responses",
          type: "message_metadata",
          data: #"{"id":"msg_123","status":"completed","phase":"commentary"}"#,
          isResponseContent: true,
        )),
        .attachment(attachment),
        .text("Supplemental note."),
      ]),
    ]

    let plan = try await ResponsesReplayNormalizer.normalize(messages)
    let messageItem = try #require(plan.inputItems.first)
    #expect(messageItem["role"] as? String == "assistant")
    #expect(messageItem["id"] == nil)
    #expect(messageItem["phase"] == nil)
  }
}
