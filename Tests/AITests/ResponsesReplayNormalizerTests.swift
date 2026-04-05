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
  func `Responses normalizer downgrades assistant segments that require input only content`() async throws {
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
    #expect(messageItem["role"] as? String == "user")
    #expect(messageItem["id"] == nil)
    #expect(messageItem["phase"] == nil)
  }
}
