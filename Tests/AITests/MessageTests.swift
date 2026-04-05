// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct MessageTests {
  @Test
  func `Collapsing tool calls preserves visible replayable text and attachments`() throws {
    let toolCall = ToolCall(
      name: "search",
      id: "call_1",
      parameters: ["query": .string("history")],
    )
    let attachment = Attachment(
      kind: .document(data: Data("notes".utf8), mimeType: "text/plain"),
      filename: "notes.txt",
    )
    let message = Message(role: .assistant, content: [
      .text("Intro"),
      .providerOpaque(OpaqueBlock(
        provider: "openai-responses",
        type: "annotated_output_text",
        content: "Opaque output",
        isResponseContent: true,
      )),
      .endnotes("Footnote"),
      .toolCall(toolCall),
      .attachment(attachment),
    ])

    let collapsed = message.collapsingToolCalls()

    #expect(collapsed.role == Message.Role.assistant)
    #expect(collapsed.content.count == 2)

    let firstContent = try #require(collapsed.content.first)
    guard case let .text(text) = firstContent else {
      Issue.record("Expected collapsed text content")
      return
    }

    #expect(text.contains("Intro"))
    #expect(text.contains("Opaque output"))
    #expect(text.contains("Footnote"))
    #expect(text.contains(#"[Called tool "search" with: {"query":"history"}]"#))

    let lastContent = try #require(collapsed.content.last)
    guard case let .attachment(collapsedAttachment) = lastContent else {
      Issue.record("Expected attachment to be preserved")
      return
    }

    #expect(collapsedAttachment == attachment)
  }

  @Test
  func `Collapsing tool results preserves visible replayable text and attachments and becomes user message`() throws {
    let attachment = Attachment(
      kind: .image(data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png"),
      filename: "chart.png",
    )
    let toolResult = ToolResult(
      name: "search",
      id: "call_1",
      content: [
        .text("Found a match"),
        .file(Data("report".utf8), mimeType: "text/plain", filename: "report.txt"),
      ],
      isError: false,
    )
    let message = Message(role: .tool, content: [
      .text("Visible preface"),
      .providerOpaque(OpaqueBlock(
        provider: "gemini",
        type: "toolResponse",
        content: "Opaque transcript text",
        isResponseContent: true,
      )),
      .toolResult(toolResult),
      .attachment(attachment),
    ])

    let collapsed = message.collapsingToolResults()

    #expect(collapsed.role == .user)
    #expect(collapsed.content.count == 2)

    let firstContent = try #require(collapsed.content.first)
    guard case let .text(text) = firstContent else {
      Issue.record("Expected collapsed text content")
      return
    }

    #expect(text.contains("Visible preface"))
    #expect(text.contains("Opaque transcript text"))
    #expect(text.contains(#"[Result from tool "search": Found a match [Unsupported result: report.txt text/plain"#))

    let lastContent = try #require(collapsed.content.last)
    guard case let .attachment(collapsedAttachment) = lastContent else {
      Issue.record("Expected attachment to be preserved")
      return
    }

    #expect(collapsedAttachment == attachment)
  }
}

private extension Message.Content {
  var toolResult: ToolResult? {
    guard case let .toolResult(toolResult) = self else { return nil }
    return toolResult
  }
}
