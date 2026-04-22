// Copyright © Anthony DePasquale

import Foundation

/// URI-referenced resources (generated PDFs / ZIPs / videos, generated text,
/// or URL references the client fetches lazily).
///
/// Mirrors swift-mcp's `Asset` shape modulo the `annotations` and `[Icon]`
/// fields that swift-ai's content cases don't carry.
public struct Asset: Sendable, Hashable {
  public enum Block: Sendable, Hashable {
    case binary(_ data: Data, uri: String, mimeType: String? = nil)
    case text(_ text: String, uri: String, mimeType: String? = nil)
    case link(
      _ uri: String,
      name: String,
      title: String? = nil,
      description: String? = nil,
      mimeType: String? = nil,
      size: Int? = nil,
    )
  }

  public let blocks: [Block]

  public init(_ blocks: [Block]) {
    self.blocks = blocks
  }

  /// Single-block convenience for the common one-block case.
  public init(_ block: Block) {
    blocks = [block]
  }
}

extension Asset: ToolOutput {
  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.resource]
  }

  public func toToolResult() -> ToolOutputResult {
    ToolOutputResult(content: blocks.map { block in
      switch block {
        case let .binary(data, uri, mime):
          .embeddedResource(data, uri: uri, mimeType: mime)
        case let .text(text, uri, mime):
          .embeddedText(text, uri: uri, mimeType: mime)
        case let .link(uri, name, title, desc, mime, size):
          .resourceLink(
            uri: uri, name: name, title: title,
            description: desc, mimeType: mime, size: size,
          )
      }
    })
  }
}

// MARK: - AssetWithMetadata

/// URI-bearing resources + typed metadata.
///
/// The metadata text reaches text-stringifying providers via `content[]`;
/// `structuredContent` reaches Gemini and MCP.
public struct AssetWithMetadata<Metadata: StructuredOutput>: ToolOutput, Sendable {
  public let blocks: [Asset.Block]
  public let metadata: Metadata

  public init(_ blocks: [Asset.Block], metadata: Metadata) {
    self.blocks = blocks
    self.metadata = metadata
  }

  public init(_ block: Asset.Block, metadata: Metadata) {
    blocks = [block]
    self.metadata = metadata
  }

  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.text, .json, .resource]
  }

  public func toToolResult() throws -> ToolOutputResult {
    let data = try Metadata.encoder.encode(metadata)
    guard let json = String(data: data, encoding: .utf8) else {
      throw AIError.invalidRequest(
        message: "AssetWithMetadata<\(Metadata.self)>: metadata is not valid UTF-8",
      )
    }
    let structured = try JSONDecoder().decode(Value.self, from: data)
    let assetBlocks = Asset(blocks).toToolResult().content
    return ToolOutputResult(
      content: [.text(json)] + assetBlocks,
      structuredContent: structured,
    )
  }
}

extension AssetWithMetadata: StructuredMetadataCarrier {
  static var metadataSchema: Value {
    Metadata.outputJSONSchema
  }
}
