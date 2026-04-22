// Copyright © Anthony DePasquale

import AIMCP
import Foundation
import Testing

struct ResultConversionTests {
  @Test
  func `ToolResult.Content to ContentBlock - text`() {
    let value = AI.ToolResult.Content.text("Hello, world!")
    let content = MCP.ContentBlock(value)

    if case let .text(text, _, _) = content {
      #expect(text == "Hello, world!")
    } else {
      Issue.record("Expected text content")
    }
  }

  @Test
  func `ToolResult.Content to ContentBlock -image without mimeType`() {
    let imageData = Data([0x89, 0x50, 0x4E, 0x47]) // PNG magic bytes
    let value = AI.ToolResult.Content.image(imageData, mimeType: nil)
    let content = MCP.ContentBlock(value)

    if case let .image(data, mimeType, _, _) = content {
      #expect(mimeType == "image/png")
      #expect(Data(base64Encoded: data) == imageData)
    } else {
      Issue.record("Expected image content")
    }
  }

  @Test
  func `ToolResult.Content to ContentBlock -image with mimeType`() {
    let imageData = Data([0xFF, 0xD8, 0xFF])
    let value = AI.ToolResult.Content.image(imageData, mimeType: "image/jpeg")
    let content = MCP.ContentBlock(value)

    if case let .image(data, mimeType, _, _) = content {
      #expect(mimeType == "image/jpeg")
      #expect(Data(base64Encoded: data) == imageData)
    } else {
      Issue.record("Expected image content")
    }
  }

  @Test
  func `ToolResult.Content to ContentBlock -audio`() {
    let audioData = Data([0x49, 0x44, 0x33]) // ID3 tag for MP3
    let value = AI.ToolResult.Content.audio(audioData, mimeType: "audio/mpeg")
    let content = MCP.ContentBlock(value)

    if case let .audio(data, mimeType, _, _) = content {
      #expect(mimeType == "audio/mpeg")
      #expect(Data(base64Encoded: data) == audioData)
    } else {
      Issue.record("Expected audio content")
    }
  }

  @Test
  func `ToolResult.Content to ContentBlock -file (image type) round-trips through resource channel`() {
    // Per spec §H: anonymous .file bytes always round-trip via the resource channel
    // with a synthesized URI. Authors with image bytes should reach for ImageResult
    // or .embeddedResource; .file is for anonymous bytes only.
    let fileData = Data([0x89, 0x50, 0x4E, 0x47])
    let value = AI.ToolResult.Content.file(fileData, mimeType: "image/png", filename: "test.png")
    let content = MCP.ContentBlock(value)

    if case let .resource(resource, _, _) = content {
      #expect(resource.uri == "file:///test.png")
      #expect(resource.mimeType == "image/png")
      #expect(Data(base64Encoded: resource.blob ?? "") == fileData)
    } else {
      Issue.record("Expected embedded resource content for file with image MIME")
    }
  }

  @Test
  func `ToolResult.Content to ContentBlock -file (audio type) round-trips through resource channel`() {
    let fileData = Data([0x49, 0x44, 0x33])
    let value = AI.ToolResult.Content.file(fileData, mimeType: "audio/wav", filename: "test.wav")
    let content = MCP.ContentBlock(value)

    if case let .resource(resource, _, _) = content {
      #expect(resource.uri == "file:///test.wav")
      #expect(resource.mimeType == "audio/wav")
      #expect(Data(base64Encoded: resource.blob ?? "") == fileData)
    } else {
      Issue.record("Expected embedded resource content for file with audio MIME")
    }
  }

  @Test
  func `ToolResult.Content to ContentBlock -file (generic type)`() {
    let fileData = Data([0x25, 0x50, 0x44, 0x46])
    let value = AI.ToolResult.Content.file(fileData, mimeType: "application/pdf", filename: "report.pdf")
    let content = MCP.ContentBlock(value)

    if case let .resource(resource, _, _) = content {
      #expect(resource.uri == "file:///report.pdf")
      #expect(resource.mimeType == "application/pdf")
      #expect(resource.text == nil)
      #expect(Data(base64Encoded: resource.blob ?? "") == fileData)
    } else {
      Issue.record("Expected embedded resource content for generic file")
    }
  }

  @Test
  func `ToolResult to CallTool.Result`() {
    let result = AI.ToolResult(
      name: "test_tool",
      id: "call-123",
      content: .text("Success!"),
    )

    let mcpResult = MCP.CallTool.Result(result)
    #expect(mcpResult.isError == nil)
    #expect(mcpResult.content.count == 1)

    if case let .text(text, _, _) = mcpResult.content.first {
      #expect(text == "Success!")
    }
  }

  @Test
  func `ToolResult error to CallTool.Result with isError flag`() {
    let result = AI.ToolResult(
      name: "test_tool",
      id: "call-123",
      content: [.text("Failed!")],
      isError: true,
    )

    let mcpResult = MCP.CallTool.Result(result)
    #expect(mcpResult.isError == true)
    #expect(mcpResult.content.count == 1)

    if case let .text(text, _, _) = mcpResult.content.first {
      #expect(text == "Failed!")
    }
  }

  @Test
  func `ContentBlock to ToolResult.Content - text`() {
    let content = MCP.ContentBlock.text("Test response")
    let value = AI.ToolResult.Content(content)

    if case let .text(text) = value {
      #expect(text == "Test response")
    } else {
      Issue.record("Expected text value")
    }
  }

  @Test
  func `ContentBlock to ToolResult.Content - image`() {
    let imageData = Data([0xFF, 0xD8, 0xFF]) // JPEG magic bytes
    let base64 = imageData.base64EncodedString()
    let content = MCP.ContentBlock.image(data: base64, mimeType: "image/jpeg")
    let value = AI.ToolResult.Content(content)

    if case let .image(data, mimeType) = value {
      #expect(data == imageData)
      #expect(mimeType == "image/jpeg")
    } else {
      Issue.record("Expected image value")
    }
  }

  @Test
  func `ContentBlock to ToolResult.Content - audio`() {
    let audioData = Data([0x49, 0x44, 0x33])
    let base64 = audioData.base64EncodedString()
    let content = MCP.ContentBlock.audio(data: base64, mimeType: "audio/mpeg")
    let value = AI.ToolResult.Content(content)

    if case let .audio(data, mimeType) = value {
      #expect(data == audioData)
      #expect(mimeType == "audio/mpeg")
    } else {
      Issue.record("Expected audio value")
    }
  }

  @Test
  func `CallTool.Result to ToolResult`() {
    let mcpResult = MCP.CallTool.Result(
      content: [.text("Tool output")],
      isError: nil,
    )

    let result = AI.ToolResult(mcpResult, name: "my_tool", id: "call-456")

    #expect(result.name == "my_tool")
    #expect(result.id == "call-456")
    #expect(result.isError == nil)
    #expect(result.content.count == 1)

    if case let .text(text) = result.content.first {
      #expect(text == "Tool output")
    } else {
      Issue.record("Expected text value")
    }
  }

  @Test
  func `CallTool.Result with error to ToolResult`() {
    let mcpResult = MCP.CallTool.Result(
      content: [.text("Error: Permission denied")],
      isError: true,
    )

    let result = AI.ToolResult(mcpResult, name: "my_tool", id: "call-789")

    #expect(result.isError == true)
    #expect(result.content.count == 1)

    if case let .text(text) = result.content.first {
      #expect(text == "Error: Permission denied")
    } else {
      Issue.record("Expected text content in error result")
    }
  }

  @Test
  func `ToolCall to CallTool.Parameters`() {
    let toolCall = AI.ToolCall(
      name: "search",
      id: "call-001",
      parameters: [
        "query": .string("swift concurrency"),
        "limit": .int(10),
      ],
    )

    let params = MCP.CallTool.Parameters(toolCall)

    #expect(params.name == "search")
    #expect(params.arguments?["query"]?.stringValue == "swift concurrency")
    #expect(params.arguments?["limit"]?.intValue == 10)
  }

  @Test
  func `CallTool.Parameters to ToolCall`() {
    let params = MCP.CallTool.Parameters(
      name: "get_weather",
      arguments: [
        "city": .string("San Francisco"),
        "units": .string("celsius"),
      ],
    )

    let toolCall = AI.ToolCall(params, id: "call-002")

    #expect(toolCall.name == "get_weather")
    #expect(toolCall.id == "call-002")
    #expect(toolCall.parameters["city"]?.stringRepresentation == "San Francisco")
    #expect(toolCall.parameters["units"]?.stringRepresentation == "celsius")
  }

  // MARK: - Resource conversions (§H new content cases)

  @Test
  func `ContentBlock binary resource round-trips through embeddedResource`() {
    let blob = Data([0xDE, 0xAD, 0xBE, 0xEF])
    let block = MCP.ContentBlock.resource(uri: "doc://abc", mimeType: "application/pdf", blob: blob)
    let aiContent = AI.ToolResult.Content(block)

    if case let .embeddedResource(data, uri, mimeType) = aiContent {
      #expect(data == blob)
      #expect(uri == "doc://abc")
      #expect(mimeType == "application/pdf")
    } else {
      Issue.record("Expected .embeddedResource, got \(aiContent)")
    }

    // Round-trip back to MCP preserves URI + MIME + bytes.
    let roundTripped = MCP.ContentBlock(aiContent)
    if case let .resource(contents, _, _) = roundTripped {
      #expect(contents.uri == "doc://abc")
      #expect(contents.mimeType == "application/pdf")
      #expect(Data(base64Encoded: contents.blob ?? "") == blob)
    } else {
      Issue.record("Expected .resource, got \(roundTripped)")
    }
  }

  @Test
  func `ContentBlock text resource round-trips through embeddedText`() {
    let block = MCP.ContentBlock.resource(uri: "report://q1", mimeType: "text/markdown", text: "# Q1 Report")
    let aiContent = AI.ToolResult.Content(block)

    if case let .embeddedText(text, uri, mimeType) = aiContent {
      #expect(text == "# Q1 Report")
      #expect(uri == "report://q1")
      #expect(mimeType == "text/markdown")
    } else {
      Issue.record("Expected .embeddedText, got \(aiContent)")
    }

    let roundTripped = MCP.ContentBlock(aiContent)
    if case let .resource(contents, _, _) = roundTripped {
      #expect(contents.uri == "report://q1")
      #expect(contents.mimeType == "text/markdown")
      #expect(contents.text == "# Q1 Report")
      #expect(contents.blob == nil)
    } else {
      Issue.record("Expected .resource, got \(roundTripped)")
    }
  }

  @Test
  func `ContentBlock resourceLink round-trips preserving full link metadata`() {
    let link = MCP.ResourceLink(
      name: "design-doc",
      title: "Design Doc",
      uri: "https://example.com/design.pdf",
      description: "The system design",
      mimeType: "application/pdf",
      size: 102_400,
    )
    let block = MCP.ContentBlock.resourceLink(link)
    let aiContent = AI.ToolResult.Content(block)

    if case let .resourceLink(uri, name, title, description, mimeType, size) = aiContent {
      #expect(uri == "https://example.com/design.pdf")
      #expect(name == "design-doc")
      #expect(title == "Design Doc")
      #expect(description == "The system design")
      #expect(mimeType == "application/pdf")
      #expect(size == 102_400)
    } else {
      Issue.record("Expected .resourceLink, got \(aiContent)")
    }

    let roundTripped = MCP.ContentBlock(aiContent)
    if case let .resourceLink(linkBack) = roundTripped {
      #expect(linkBack.name == "design-doc")
      #expect(linkBack.title == "Design Doc")
      #expect(linkBack.uri == "https://example.com/design.pdf")
      #expect(linkBack.description == "The system design")
      #expect(linkBack.mimeType == "application/pdf")
      #expect(linkBack.size == 102_400)
    } else {
      Issue.record("Expected .resourceLink, got \(roundTripped)")
    }
  }

  @Test
  func `Resource with invalid base64 blob falls back to text marker`() throws {
    // Resource.Contents is JSON-decodable; construct a malformed-blob case
    // through the decoder. The fallback must preserve the URI as a marker
    // and never fabricate a zero-byte payload.
    let json = """
    {
      "type": "resource",
      "resource": {
        "uri": "broken://fixture",
        "mimeType": "application/octet-stream",
        "blob": "not-valid-base64!@#$%^&*"
      }
    }
    """.data(using: .utf8)!
    let block = try JSONDecoder().decode(MCP.ContentBlock.self, from: json)

    let aiContent = AI.ToolResult.Content(block)
    if case let .text(text) = aiContent {
      #expect(text == "[Invalid resource data: broken://fixture]")
    } else {
      Issue.record("Expected text fallback, got \(aiContent)")
    }
  }

  @Test
  func `CallTool.Result with structuredContent round-trips through ToolResult`() {
    let mcpStructured: MCP.Value = .object([
      "score": .int(95),
      "label": .string("excellent"),
    ])
    let mcpResult = MCP.CallTool.Result(
      content: [.text("done")],
      structuredContent: mcpStructured,
      isError: nil,
    )

    let result = AI.ToolResult(mcpResult, name: "rate", id: "call-1")
    #expect(result.structuredContent == .object([
      "score": .int(95),
      "label": .string("excellent"),
    ]))

    // Round-trip back: structured channel survives through both directions.
    let mcpAgain = MCP.CallTool.Result(result)
    #expect(mcpAgain.structuredContent == mcpStructured)
  }

  @Test
  func `CallTool.Result rich error round-trips with content + structuredContent + isError`() {
    // Per spec §H: an MCP server returning isError:true with rich content must
    // round-trip with all blocks preserved, instead of being collapsed to a
    // thrown opaque error.
    let imageBytes = Data([0xFF, 0xD8, 0xFF, 0xE0]) // JPEG magic
    let mcpResult = MCP.CallTool.Result(
      content: [
        .text("operation failed"),
        .image(data: imageBytes.base64EncodedString(), mimeType: "image/jpeg"),
      ],
      structuredContent: .object([
        "code": .int(42),
        "message": .string("operation failed"),
      ]),
      isError: true,
    )

    let result = AI.ToolResult(mcpResult, name: "broken", id: "call-9")
    #expect(result.isError == true)
    #expect(result.content.count == 2)
    if case let .text(text) = result.content[0] {
      #expect(text == "operation failed")
    } else {
      Issue.record("Expected first content block to be .text")
    }
    if case let .image(data, mimeType) = result.content[1] {
      #expect(data == imageBytes)
      #expect(mimeType == "image/jpeg")
    } else {
      Issue.record("Expected second content block to be .image")
    }
    #expect(result.structuredContent == .object([
      "code": .int(42),
      "message": .string("operation failed"),
    ]))
  }
}
