// Copyright © Anthony DePasquale

import Foundation

/// The dual-channel return value of `ToolOutput.toToolResult()`.
///
/// `content` is the model-facing wire — what the LLM sees as part of its input.
/// `structuredContent` is the parallel programmatic channel that becomes
/// `functionResponse.response` on Gemini and round-trips through MCP's
/// `CallTool.Result.structuredContent`. Text-stringifying providers
/// (Anthropic / Responses / ChatCompletions) read `content[]` only.
public struct ToolOutputResult: Sendable {
  /// The content blocks the model sees.
  public let content: [ToolResult.Content]
  /// Optional programmatic structured channel.
  public let structuredContent: Value?

  public init(content: [ToolResult.Content], structuredContent: Value? = nil) {
    self.content = content
    self.structuredContent = structuredContent
  }
}

/// A type that can be returned from a tool's `perform()` method.
///
/// Each conforming type declares what result types it produces via `resultTypes`.
/// This enables compile-time derivation of tool capabilities - the `@Tool` macro
/// automatically sets `Tool.resultTypes` based on the `perform()` return type,
/// eliminating the possibility of mismatched declarations.
///
/// Built-in conformances (see also `PrimitiveToolOutput`, `StructuredOutput`,
/// `Asset`, `Media`):
/// - `String`, `Int`, `Double`, `Bool`, `Date`, `Array<WrappableValue>`,
///   `Optional<WrappableValue>` → `[.text, .json]` (via `PrimitiveToolOutput`,
///   wrapped under `"result"` in `structuredContent`)
/// - `Dictionary<String, WrappableValue>` → `[.text, .json]` (unwrapped
///   top-level object in `structuredContent`)
/// - `Void` / `VoidOutput` → `[.text, .json]` (`{"result": null}`)
/// - `@StructuredOutput` struct → `[.text, .json]` (unwrapped struct shape)
/// - `ImageResult` → `[.image]`; `ImageWithMetadata<T>` → `[.text, .json, .image]`
/// - `AudioResult` → `[.audio]`; `AudioWithMetadata<T>` → `[.text, .json, .audio]`
/// - `Media` → `[.image, .audio]`; `MediaWithMetadata<T>` → `[.text, .json, .image, .audio]`
/// - `Asset` → `[.resource]`; `AssetWithMetadata<T>` → `[.text, .json, .resource]`
/// - `FileResult` → `[.file]`
/// - `MultiContent` → `nil` (contents determined at runtime)
///
/// Example:
/// ```swift
/// func perform() async throws -> String {
///     "Hello, world!"
/// }
/// ```
public protocol ToolOutput: Sendable {
  /// The result types this output produces.
  ///
  /// Used by the `@Tool` macro to automatically derive `Tool.resultTypes`,
  /// enabling capability-based filtering (e.g., `tools.compatible(with: ChatCompletionsClient.self)`).
  ///
  /// Return `nil` for types like `MultiContent` where the actual content types
  /// are determined at runtime.
  static var resultTypes: Set<ToolResult.ValueType>? { get }

  /// Convert to the dual-channel `ToolOutputResult` for the response.
  /// - Throws: On encoding failure - the dispatcher catches and surfaces as `isError`.
  func toToolResult() throws -> ToolOutputResult
}

// `String` conforms to `ToolOutput` transitively through `PrimitiveToolOutput`
// (see `PrimitiveToolOutput.swift`). A tool returning `String` emits
// `content = [.text(value)]` *and* `structuredContent = {"result": value}` —
// both display and wire channels populated.

// MARK: - Image Output

/// Output type for tools that return images.
///
/// Example:
/// ```swift
/// func perform() async throws -> ImageResult {
///     let imageData = try await captureScreen()
///     return ImageResult(pngData: imageData)
/// }
/// ```
public struct ImageResult: ToolOutput, Sendable {
  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.image]
  }

  /// The raw image data.
  public let data: Data

  /// The MIME type of the image (e.g., "image/png", "image/jpeg").
  public let mimeType: String

  /// Creates an image output with the specified data and MIME type.
  /// - Parameters:
  ///   - data: The raw image data.
  ///   - mimeType: The MIME type of the image.
  public init(data: Data, mimeType: String) {
    self.data = data
    self.mimeType = mimeType
  }

  /// Creates an image output from PNG data.
  /// - Parameter pngData: The PNG image data.
  public init(pngData: Data) {
    self.init(data: pngData, mimeType: "image/png")
  }

  /// Creates an image output from JPEG data.
  /// - Parameter jpegData: The JPEG image data.
  public init(jpegData: Data) {
    self.init(data: jpegData, mimeType: "image/jpeg")
  }

  public func toToolResult() -> ToolOutputResult {
    ToolOutputResult(content: [.image(data, mimeType: mimeType)])
  }
}

// MARK: - Audio Output

/// Output type for tools that return audio.
///
/// Example:
/// ```swift
/// func perform() async throws -> AudioResult {
///     let audioData = try await synthesizeSpeech(text: text)
///     return AudioResult(data: audioData, mimeType: "audio/mpeg")
/// }
/// ```
public struct AudioResult: ToolOutput, Sendable {
  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.audio]
  }

  /// The raw audio data.
  public let data: Data

  /// The MIME type of the audio (e.g., "audio/mpeg", "audio/wav").
  public let mimeType: String

  /// Creates an audio output with the specified data and MIME type.
  /// - Parameters:
  ///   - data: The raw audio data.
  ///   - mimeType: The MIME type of the audio.
  public init(data: Data, mimeType: String) {
    self.data = data
    self.mimeType = mimeType
  }

  public func toToolResult() -> ToolOutputResult {
    ToolOutputResult(content: [.audio(data, mimeType: mimeType)])
  }
}

// MARK: - File Output

/// Output type for tools that return files.
///
/// Example:
/// ```swift
/// func perform() async throws -> FileResult {
///     let fileData = try await generateReport()
///     return FileResult(data: fileData, mimeType: "application/pdf", filename: "report.pdf")
/// }
/// ```
public struct FileResult: ToolOutput, Sendable {
  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.file]
  }

  /// The raw file data.
  public let data: Data

  /// The MIME type of the file.
  public let mimeType: String

  /// Optional filename.
  public let filename: String?

  /// Creates a file output with the specified data, MIME type, and optional filename.
  /// - Parameters:
  ///   - data: The raw file data.
  ///   - mimeType: The MIME type of the file.
  ///   - filename: Optional filename.
  public init(data: Data, mimeType: String, filename: String? = nil) {
    self.data = data
    self.mimeType = mimeType
    self.filename = filename
  }

  public func toToolResult() -> ToolOutputResult {
    ToolOutputResult(content: [.file(data, mimeType: mimeType, filename: filename)])
  }
}

// MARK: - Multi-Content Output

/// Output type for tools that return multiple content items.
///
/// Example:
/// ```swift
/// func perform() async throws -> MultiContent {
///     MultiContent([
///         .text("Analysis complete"),
///         .image(chartData, mimeType: "image/png")
///     ])
/// }
/// ```
///
/// `MultiContent` populates `content[]` only — `structuredContent` is always
/// `nil`. Authors who need the structured channel (Gemini's
/// `functionResponse.response`, MCP's `structuredContent`) should construct
/// a `ToolOutputResult(content:, structuredContent:)` directly from a custom
/// `ToolOutput` conformer.
public struct MultiContent: ToolOutput, Sendable {
  /// Returns `nil` because the actual content types are determined at runtime.
  public static var resultTypes: Set<ToolResult.ValueType>? {
    nil
  }

  /// The content items to return.
  public let items: [ToolResult.Content]

  /// Creates a multi-content output with the specified items.
  /// - Parameter items: The content items.
  public init(_ items: [ToolResult.Content]) {
    self.items = items
  }

  public func toToolResult() -> ToolOutputResult {
    ToolOutputResult(content: items)
  }
}
