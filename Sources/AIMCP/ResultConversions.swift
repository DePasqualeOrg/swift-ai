// Copyright © Anthony DePasquale

import AI
import Foundation
import ImageIO
import MCP
import UniformTypeIdentifiers

// MARK: - ToolResult.Content → Tool.Content Conversion

extension MCP.Tool.Content {
  /// Creates MCP Tool Content from an AI ToolResult Content.
  public init(_ value: AI.ToolResult.Content) {
    switch value {
      case let .text(text):
        self = .text(text)
      case let .image(data, providedMimeType):
        let mimeType = providedMimeType ?? Self.detectImageMimeType(data) ?? "image/png"
        self = .image(data: data.base64EncodedString(), mimeType: mimeType)
      case let .audio(data, mimeType):
        self = .audio(data: data.base64EncodedString(), mimeType: mimeType)
      case let .file(data, mimeType, _):
        // MCP doesn't have a direct file content type, so we use the appropriate media type
        // or fall back to a resource representation
        if mimeType.hasPrefix("image/") {
          self = .image(data: data.base64EncodedString(), mimeType: mimeType)
        } else if mimeType.hasPrefix("audio/") {
          self = .audio(data: data.base64EncodedString(), mimeType: mimeType)
        } else {
          // For other file types, encode as text with base64 data
          self = .text("[File data: \(mimeType), \(data.count) bytes]")
        }
    }
  }

  private static func detectImageMimeType(_ imageData: Data) -> String? {
    guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, nil),
          let uniformTypeIdentifier = CGImageSourceGetType(imageSource) as String?,
          let utType = UTType(uniformTypeIdentifier),
          let mimeType = utType.preferredMIMEType
    else {
      return nil
    }
    return mimeType
  }
}

// MARK: - ToolResult → CallTool.Result Conversion

public extension MCP.CallTool.Result {
  /// Creates an MCP CallTool Result from an AI ToolResult.
  init(_ result: AI.ToolResult) {
    let content = result.content.map { MCP.Tool.Content($0) }
    self.init(content: content, isError: result.isError)
  }
}

// MARK: - Tool.Content → ToolResult.Content Conversion

public extension AI.ToolResult.Content {
  /// Creates an AI ToolResult Value from MCP Tool Content.
  init(_ content: MCP.Tool.Content) {
    switch content {
      case let .text(text, _, _):
        self = .text(text)
      case let .image(data, mimeType, _, _):
        if let imageData = Data(base64Encoded: data) {
          self = .image(imageData, mimeType: mimeType)
        } else {
          self = .text("[Invalid image data]")
        }
      case let .audio(data, mimeType, _, _):
        if let audioData = Data(base64Encoded: data) {
          self = .audio(audioData, mimeType: mimeType)
        } else {
          self = .text("[Invalid audio data]")
        }
      case let .resource(resource, _, _):
        // Resource.Content is a struct with text and/or blob fields
        if let text = resource.text {
          self = .text(text)
        } else if let blob = resource.blob, let data = Data(base64Encoded: blob) {
          let mimeType = resource.mimeType ?? "application/octet-stream"
          if mimeType.hasPrefix("image/") {
            self = .image(data, mimeType: mimeType)
          } else if mimeType.hasPrefix("audio/") {
            self = .audio(data, mimeType: mimeType)
          } else {
            // Use file type for other binary resources
            self = .file(data, mimeType: mimeType, filename: nil)
          }
        } else {
          self = .text("[Resource: \(resource.uri)]")
        }
      case let .resourceLink(link):
        self = .text("[Resource link: \(link.uri)]")
    }
  }
}

// MARK: - CallTool.Result → ToolResult Conversion

public extension AI.ToolResult {
  /// Creates an AI ToolResult from an MCP CallTool Result.
  ///
  /// - Parameters:
  ///   - result: The MCP CallTool Result
  ///   - name: The tool name (required for ToolResult)
  ///   - id: The call ID (required for ToolResult)
  init(_ result: MCP.CallTool.Result, name: String, id: String) {
    let content = result.content.map { AI.ToolResult.Content($0) }
    self.init(name: name, id: id, content: content, isError: result.isError)
  }
}

// MARK: - GenerationResponse.ToolCall ↔ CallTool.Parameters

public extension MCP.CallTool.Parameters {
  /// Creates MCP CallTool Parameters from an AI ToolCall.
  init(_ toolCall: AI.GenerationResponse.ToolCall) {
    self.init(
      name: toolCall.name,
      arguments: toolCall.parameters.mcpValues
    )
  }
}

public extension AI.GenerationResponse.ToolCall {
  /// Creates an AI ToolCall from MCP CallTool Parameters.
  ///
  /// - Parameters:
  ///   - params: The MCP CallTool Parameters
  ///   - id: The call ID (MCP doesn't include this in Parameters)
  init(_ params: MCP.CallTool.Parameters, id: String) {
    self.init(
      name: params.name,
      id: id,
      parameters: params.arguments?.aiValues ?? [:]
    )
  }
}
