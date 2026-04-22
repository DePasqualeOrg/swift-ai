// Copyright © Anthony DePasquale

import Foundation

/// Mixed-modality bare bytes (image + audio in one return).
///
/// Rare authoring case (video-frame-plus-audio, lip-sync analysis). For
/// single-modality returns, prefer `ImageResult` / `AudioResult` — narrower
/// `resultTypes`, no compatibility-filter regression.
public struct Media: Sendable, Hashable {
  public enum Block: Sendable, Hashable {
    case image(data: Data, mimeType: String)
    case audio(data: Data, mimeType: String)
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

extension Media: ToolOutput {
  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.image, .audio]
  }

  public func toToolResult() -> ToolOutputResult {
    ToolOutputResult(content: blocks.map { block in
      switch block {
        case let .image(data, mime): .image(data, mimeType: mime)
        case let .audio(data, mime): .audio(data, mimeType: mime)
      }
    })
  }
}

// MARK: - Metadata wrappers

/// Single-image bytes + typed metadata.
///
/// Recommended for screenshot, generated-image, OCR-result tools — the dominant
/// metadata-bearing case. `resultTypes: [.text, .json, .image]` is a subset of
/// every `.image`-supporting provider, so the tool appears on every applicable
/// provider with no per-tool override needed.
public struct ImageWithMetadata<Metadata: StructuredOutput>: ToolOutput, Sendable {
  public let data: Data
  public let mimeType: String
  public let metadata: Metadata

  public init(_ data: Data, mimeType: String, metadata: Metadata) {
    self.data = data
    self.mimeType = mimeType
    self.metadata = metadata
  }

  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.text, .json, .image]
  }

  public func toToolResult() throws -> ToolOutputResult {
    let (json, structured) = try encodeMetadataPair(metadata, label: "ImageWithMetadata")
    return ToolOutputResult(
      content: [.text(json), .image(data, mimeType: mimeType)],
      structuredContent: structured,
    )
  }
}

/// Single-audio bytes + typed metadata.
///
/// Recommended for transcription-with-metadata, TTS-with-metadata,
/// audio-analysis tools.
public struct AudioWithMetadata<Metadata: StructuredOutput>: ToolOutput, Sendable {
  public let data: Data
  public let mimeType: String
  public let metadata: Metadata

  public init(_ data: Data, mimeType: String, metadata: Metadata) {
    self.data = data
    self.mimeType = mimeType
    self.metadata = metadata
  }

  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.text, .json, .audio]
  }

  public func toToolResult() throws -> ToolOutputResult {
    let (json, structured) = try encodeMetadataPair(metadata, label: "AudioWithMetadata")
    return ToolOutputResult(
      content: [.text(json), .audio(data, mimeType: mimeType)],
      structuredContent: structured,
    )
  }
}

/// Mixed-modality bytes + typed metadata.
///
/// Reserved for the rare case where a single tool return needs both modalities
/// together (lip-sync analysis, video-frame + audio-clip). For the common
/// single-modality case, prefer `ImageWithMetadata<T>` / `AudioWithMetadata<T>`.
public struct MediaWithMetadata<Metadata: StructuredOutput>: ToolOutput, Sendable {
  public let blocks: [Media.Block]
  public let metadata: Metadata

  public init(_ blocks: [Media.Block], metadata: Metadata) {
    self.blocks = blocks
    self.metadata = metadata
  }

  /// Single-block convenience for the common one-block case.
  public init(_ block: Media.Block, metadata: Metadata) {
    blocks = [block]
    self.metadata = metadata
  }

  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.text, .json, .image, .audio]
  }

  public func toToolResult() throws -> ToolOutputResult {
    let (json, structured) = try encodeMetadataPair(metadata, label: "MediaWithMetadata")
    let mediaBlocks: [ToolResult.Content] = blocks.map { block in
      switch block {
        case let .image(data, mime): .image(data, mimeType: mime)
        case let .audio(data, mime): .audio(data, mimeType: mime)
      }
    }
    return ToolOutputResult(
      content: [.text(json)] + mediaBlocks,
      structuredContent: structured,
    )
  }
}

// MARK: - StructuredMetadataCarrier conformances

extension ImageWithMetadata: StructuredMetadataCarrier {
  static var metadataSchema: Value {
    Metadata.outputJSONSchema
  }
}

extension AudioWithMetadata: StructuredMetadataCarrier {
  static var metadataSchema: Value {
    Metadata.outputJSONSchema
  }
}

extension MediaWithMetadata: StructuredMetadataCarrier {
  static var metadataSchema: Value {
    Metadata.outputJSONSchema
  }
}

// MARK: - Shared encoding helper

/// Encodes the metadata struct through `Metadata.encoder` and round-trips the
/// bytes into `Value` via `JSONDecoder`. Returns the JSON string (for
/// `content[].text`) and the decoded `Value` (for `structuredContent`).
private func encodeMetadataPair<M: StructuredOutput>(
  _ metadata: M,
  label: String,
) throws -> (json: String, structured: Value) {
  let data = try M.encoder.encode(metadata)
  guard let json = String(data: data, encoding: .utf8) else {
    throw AIError.invalidRequest(message: "\(label)<\(M.self)>: metadata is not valid UTF-8")
  }
  let structured = try JSONDecoder().decode(Value.self, from: data)
  return (json, structured)
}
