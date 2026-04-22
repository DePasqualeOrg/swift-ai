// Copyright © Anthony DePasquale

import AI
import Foundation
import ImageIO
import MCP
import UniformTypeIdentifiers

// MARK: - ToolResult.Content → ContentBlock Conversion

extension MCP.ContentBlock {
  /// Creates an MCP ContentBlock from an AI ToolResult Content.
  public init(_ value: AI.ToolResult.Content) {
    switch value {
      case let .text(text):
        self = .text(text)
      case let .json(jsonValue):
        self = .text(jsonValue.jsonString)
      case let .image(data, providedMimeType):
        let mimeType = providedMimeType ?? Self.detectImageMimeType(data) ?? "image/png"
        self = .image(data: data.base64EncodedString(), mimeType: mimeType)
      case let .audio(data, mimeType):
        self = .audio(data: data.base64EncodedString(), mimeType: mimeType)
      case let .file(data, mimeType, filename):
        // Anonymous bytes get a synthesized URI so they round-trip through the resource channel.
        self = .resource(uri: Self.fileResourceURI(filename: filename), mimeType: mimeType, blob: data)
      case let .embeddedResource(data, uri, mimeType):
        self = .resource(uri: uri, mimeType: mimeType, blob: data)
      case let .embeddedText(text, uri, mimeType):
        self = .resource(uri: uri, mimeType: mimeType, text: text)
      case let .resourceLink(uri, name, title, description, mimeType, size):
        self = .resourceLink(MCP.ResourceLink(
          name: name,
          title: title,
          uri: uri,
          description: description,
          mimeType: mimeType,
          size: size,
        ))
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

  private static func fileResourceURI(filename: String?) -> String {
    let path = filename.flatMap { $0.isEmpty ? nil : $0 } ?? "tool-result"
    return URL(fileURLWithPath: "/\(path)").absoluteString
  }
}

// MARK: - ToolResult → CallTool.Result Conversion

public extension MCP.CallTool.Result {
  /// Creates an MCP CallTool Result from an AI ToolResult.
  init(_ result: AI.ToolResult) {
    let content = result.content.map { MCP.ContentBlock($0) }
    let structured = result.structuredContent?.mcpValue
    self.init(content: content, structuredContent: structured, isError: result.isError)
  }
}

// MARK: - ContentBlock → ToolResult.Content Conversion

public extension AI.ToolResult.Content {
  /// Creates an AI ToolResult Value from an MCP ContentBlock.
  init(_ content: MCP.ContentBlock) {
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
        // Resource.Contents has either text or blob; preserve URI on both branches.
        if let text = resource.text {
          self = .embeddedText(text, uri: resource.uri, mimeType: resource.mimeType)
        } else if let blob = resource.blob {
          if let data = Data(base64Encoded: blob) {
            self = .embeddedResource(data, uri: resource.uri, mimeType: resource.mimeType)
          } else {
            self = .text("[Invalid resource data: \(resource.uri)]")
          }
        } else {
          self = .text("[Resource: \(resource.uri)]")
        }
      case let .resourceLink(link):
        self = .resourceLink(
          uri: link.uri,
          name: link.name,
          title: link.title,
          description: link.description,
          mimeType: link.mimeType,
          size: link.size,
        )
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
    let structured = result.structuredContent?.aiValue
    self.init(
      name: name,
      id: id,
      content: content,
      structuredContent: structured,
      isError: result.isError,
    )
  }
}

// MARK: - ToolCall ↔ CallTool.Parameters

public extension MCP.CallTool.Parameters {
  /// Creates MCP CallTool Parameters from an AI ToolCall.
  init(_ toolCall: AI.ToolCall) {
    self.init(
      name: toolCall.name,
      arguments: toolCall.parameters.mcpValues,
    )
  }
}

public extension AI.ToolCall {
  /// Creates an AI ToolCall from MCP CallTool Parameters.
  ///
  /// - Parameters:
  ///   - params: The MCP CallTool Parameters
  ///   - id: The call ID (MCP doesn't include this in Parameters)
  init(_ params: MCP.CallTool.Parameters, id: String) {
    self.init(
      name: params.name,
      id: id,
      parameters: params.arguments?.aiValues ?? [:],
    )
  }
}
