// Copyright © Anthony DePasquale

import Foundation

/// A type that can be returned from a tool's `perform()` method.
///
/// Each conforming type declares what result types it produces via `resultTypes`.
/// This enables compile-time derivation of tool capabilities - the `@Tool` macro
/// automatically sets `Tool.resultTypes` based on the `perform()` return type,
/// eliminating the possibility of mismatched declarations.
///
/// Built-in conformances:
/// - `String` → `[.text]`
/// - `ImageResult` → `[.image]`
/// - `AudioResult` → `[.audio]`
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

  /// Convert to `ToolResult.Content` for the response.
  func toToolResult() -> [ToolResult.Content]
}

// MARK: - String Conformance

extension String: ToolOutput {
  public static var resultTypes: Set<ToolResult.ValueType>? {
    [.text]
  }

  public func toToolResult() -> [ToolResult.Content] {
    [.text(self)]
  }
}

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

  public func toToolResult() -> [ToolResult.Content] {
    [.image(data, mimeType: mimeType)]
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

  public func toToolResult() -> [ToolResult.Content] {
    [.audio(data, mimeType: mimeType)]
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

  public func toToolResult() -> [ToolResult.Content] {
    [.file(data, mimeType: mimeType, filename: filename)]
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

  public func toToolResult() -> [ToolResult.Content] {
    items
  }
}
