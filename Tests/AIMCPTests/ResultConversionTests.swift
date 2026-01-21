// Copyright © Anthony DePasquale

import AIMCP
import Foundation
import Testing

@Suite("Result Conversions")
struct ResultConversionTests {
  @Test("ToolResult.Content to Tool.Content - text")
  func toolResultValueToToolContentText() {
    let value = AI.ToolResult.Content.text("Hello, world!")
    let content = MCP.Tool.Content(value)

    if case let .text(text, _, _) = content {
      #expect(text == "Hello, world!")
    } else {
      Issue.record("Expected text content")
    }
  }

  @Test("ToolResult.Content to Tool.Content - image without mimeType")
  func toolResultValueToToolContentImage() {
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

  @Test("ToolResult.Content to Tool.Content - image with mimeType")
  func toolResultValueToToolContentImageWithMimeType() {
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

  @Test("ToolResult.Content to Tool.Content - audio")
  func toolResultValueToToolContentAudio() {
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

  @Test("ToolResult.Content to Tool.Content - file (image type)")
  func toolResultValueToToolContentFileImage() {
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

  @Test("ToolResult.Content to Tool.Content - file (audio type)")
  func toolResultValueToToolContentFileAudio() {
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

  @Test("ToolResult to CallTool.Result")
  func toolResultToCallToolResult() {
    let result = AI.ToolResult(
      name: "test_tool",
      id: "call-123",
      content: .text("Success!")
    )

    let mcpResult = MCP.CallTool.Result(result)
    #expect(mcpResult.isError == nil)
    #expect(mcpResult.content.count == 1)

    if case let .text(text, _, _) = mcpResult.content.first {
      #expect(text == "Success!")
    }
  }

  @Test("ToolResult error to CallTool.Result with isError flag")
  func toolResultErrorToCallToolResult() {
    let result = AI.ToolResult.error("Failed!", name: "test_tool", id: "call-123")

    let mcpResult = MCP.CallTool.Result(result)
    #expect(mcpResult.isError == true)
    #expect(mcpResult.content.count == 1)

    if case let .text(text, _, _) = mcpResult.content.first {
      #expect(text == "Failed!")
    }
  }

  @Test("Tool.Content to ToolResult.Content - text")
  func toolContentToToolResultValueText() {
    let content = MCP.Tool.Content.text("Test response")
    let value = AI.ToolResult.Content(content)

    if case let .text(text) = value {
      #expect(text == "Test response")
    } else {
      Issue.record("Expected text value")
    }
  }

  @Test("Tool.Content to ToolResult.Content - image")
  func toolContentToToolResultValueImage() {
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

  @Test("Tool.Content to ToolResult.Content - audio")
  func toolContentToToolResultValueAudio() {
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

  @Test("CallTool.Result to ToolResult")
  func callToolResultToToolResult() {
    let mcpResult = MCP.CallTool.Result(
      content: [.text("Tool output")],
      isError: nil
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

  @Test("CallTool.Result with error to ToolResult")
  func callToolResultErrorToToolResult() {
    let mcpResult = MCP.CallTool.Result(
      content: [.text("Error: Permission denied")],
      isError: true
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

  @Test("GenerationResponse.ToolCall to CallTool.Parameters")
  func toolCallToCallToolParameters() {
    let toolCall = AI.GenerationResponse.ToolCall(
      name: "search",
      id: "call-001",
      parameters: [
        "query": .string("swift concurrency"),
        "limit": .int(10),
      ]
    )

    let params = MCP.CallTool.Parameters(toolCall)

    #expect(params.name == "search")
    #expect(params.arguments?["query"]?.stringValue == "swift concurrency")
    #expect(params.arguments?["limit"]?.intValue == 10)
  }

  @Test("CallTool.Parameters to GenerationResponse.ToolCall")
  func callToolParametersToToolCall() {
    let params = MCP.CallTool.Parameters(
      name: "get_weather",
      arguments: [
        "city": .string("San Francisco"),
        "units": .string("celsius"),
      ]
    )

    let toolCall = AI.GenerationResponse.ToolCall(params, id: "call-002")

    #expect(toolCall.name == "get_weather")
    #expect(toolCall.id == "call-002")
    #expect(toolCall.parameters["city"]?.stringRepresentation == "San Francisco")
    #expect(toolCall.parameters["units"]?.stringRepresentation == "celsius")
  }
}
