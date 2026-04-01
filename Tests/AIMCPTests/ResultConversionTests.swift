// Copyright © Anthony DePasquale

import AIMCP
import Foundation
import Testing

struct ResultConversionTests {
  @Test
  func `ToolResult.Content to Tool.Content - text`() {
    let value = AI.ToolResult.Content.text("Hello, world!")
    let content = MCP.Tool.Content(value)

    if case let .text(text, _, _) = content {
      #expect(text == "Hello, world!")
    } else {
      Issue.record("Expected text content")
    }
  }

  @Test
  func `ToolResult.Content to Tool.Content - image without mimeType`() {
    let imageData = Data([0x89, 0x50, 0x4E, 0x47]) // PNG magic bytes
    let value = AI.ToolResult.Content.image(imageData, mimeType: nil)
    let content = MCP.Tool.Content(value)

    if case let .image(data, mimeType, _, _) = content {
      #expect(mimeType == "image/png")
      #expect(Data(base64Encoded: data) == imageData)
    } else {
      Issue.record("Expected image content")
    }
  }

  @Test
  func `ToolResult.Content to Tool.Content - image with mimeType`() {
    let imageData = Data([0xFF, 0xD8, 0xFF])
    let value = AI.ToolResult.Content.image(imageData, mimeType: "image/jpeg")
    let content = MCP.Tool.Content(value)

    if case let .image(data, mimeType, _, _) = content {
      #expect(mimeType == "image/jpeg")
      #expect(Data(base64Encoded: data) == imageData)
    } else {
      Issue.record("Expected image content")
    }
  }

  @Test
  func `ToolResult.Content to Tool.Content - audio`() {
    let audioData = Data([0x49, 0x44, 0x33]) // ID3 tag for MP3
    let value = AI.ToolResult.Content.audio(audioData, mimeType: "audio/mpeg")
    let content = MCP.Tool.Content(value)

    if case let .audio(data, mimeType, _, _) = content {
      #expect(mimeType == "audio/mpeg")
      #expect(Data(base64Encoded: data) == audioData)
    } else {
      Issue.record("Expected audio content")
    }
  }

  @Test
  func `ToolResult.Content to Tool.Content - file (image type)`() {
    let fileData = Data([0x89, 0x50, 0x4E, 0x47])
    let value = AI.ToolResult.Content.file(fileData, mimeType: "image/png", filename: "test.png")
    let content = MCP.Tool.Content(value)

    // File with image mimeType should become image content
    if case let .image(data, mimeType, _, _) = content {
      #expect(mimeType == "image/png")
      #expect(Data(base64Encoded: data) == fileData)
    } else {
      Issue.record("Expected image content for image file")
    }
  }

  @Test
  func `ToolResult.Content to Tool.Content - file (audio type)`() {
    let fileData = Data([0x49, 0x44, 0x33])
    let value = AI.ToolResult.Content.file(fileData, mimeType: "audio/wav", filename: "test.wav")
    let content = MCP.Tool.Content(value)

    // File with audio mimeType should become audio content
    if case let .audio(data, mimeType, _, _) = content {
      #expect(mimeType == "audio/wav")
      #expect(Data(base64Encoded: data) == fileData)
    } else {
      Issue.record("Expected audio content for audio file")
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
    let result = AI.ToolResult.error("Failed!", name: "test_tool", id: "call-123")

    let mcpResult = MCP.CallTool.Result(result)
    #expect(mcpResult.isError == true)
    #expect(mcpResult.content.count == 1)

    if case let .text(text, _, _) = mcpResult.content.first {
      #expect(text == "Failed!")
    }
  }

  @Test
  func `Tool.Content to ToolResult.Content - text`() {
    let content = MCP.Tool.Content.text("Test response")
    let value = AI.ToolResult.Content(content)

    if case let .text(text) = value {
      #expect(text == "Test response")
    } else {
      Issue.record("Expected text value")
    }
  }

  @Test
  func `Tool.Content to ToolResult.Content - image`() {
    let imageData = Data([0xFF, 0xD8, 0xFF]) // JPEG magic bytes
    let base64 = imageData.base64EncodedString()
    let content = MCP.Tool.Content.image(data: base64, mimeType: "image/jpeg")
    let value = AI.ToolResult.Content(content)

    if case let .image(data, mimeType) = value {
      #expect(data == imageData)
      #expect(mimeType == "image/jpeg")
    } else {
      Issue.record("Expected image value")
    }
  }

  @Test
  func `Tool.Content to ToolResult.Content - audio`() {
    let audioData = Data([0x49, 0x44, 0x33])
    let base64 = audioData.base64EncodedString()
    let content = MCP.Tool.Content.audio(data: base64, mimeType: "audio/mpeg")
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
}
