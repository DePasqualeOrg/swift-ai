// Copyright © Anthony DePasquale

import Foundation

/// A tool output type that provides a JSON Schema for validation.
///
/// Conforming types pair `@Schemable` (from JSONSchemaBuilder) for schema
/// generation with `@StructuredOutput` (from AITool) for a stable wire
/// encoding and automatic `StructuredOutput` conformance.
///
/// Example:
/// ```swift
/// @Schemable
/// @StructuredOutput
/// struct EventList: Sendable {
///     let events: [String]
///     let totalCount: Int
/// }
///
/// @Tool
/// struct GetEvents {
///     static let name = "get_events"
///     static let description = "Get events"
///
///     func perform() async throws -> EventList {
///         EventList(events: ["Event 1", "Event 2"], totalCount: 2)
///     }
/// }
/// ```
///
/// `@StructuredOutput` generates the `outputJSONSchema` implementation by
/// bridging to `Self.schema` (from `@Schemable`) through `SchemableAdapter`.
/// One source of truth for schema generation across inputs and outputs.
public protocol StructuredOutput: ToolOutput, Encodable {
  /// The JSON Schema for this output type, in AI wire (`Value`) form.
  ///
  /// `@StructuredOutput` synthesizes this from the type's `@Schemable`
  /// component and post-processes the result so every property appears in
  /// `required` — matching the wire contract where optional Swift properties
  /// are always emitted as `null` rather than absent.
  static var outputJSONSchema: Value { get }

  /// The `JSONEncoder` used when encoding this type for the wire.
  ///
  /// Defaults to `AIEncoding.defaultEncoder()` — sorted keys plus ISO8601
  /// date encoding.
  static var encoder: JSONEncoder { get }
}

public extension StructuredOutput {
  static var encoder: JSONEncoder {
    AIEncoding.defaultEncoder()
  }

  /// `[.text, .json]`: `.text` for `content[].text(jsonString)`; `.json` for
  /// the structured channel (universal across providers per §A capability table).
  static var resultTypes: Set<ToolResult.ValueType>? {
    [.text, .json]
  }

  /// Default implementation that encodes to JSON (using `Self.encoder`) and
  /// includes both `content[0].text` (stringified payload, model-facing) and
  /// `structuredContent` (decoded `Value`, programmatic / Gemini / MCP).
  /// Tools needing a different wire shape override this method.
  func toToolResult() throws -> ToolOutputResult {
    let data = try Self.encoder.encode(self)

    guard let json = String(data: data, encoding: .utf8) else {
      throw AIError.invalidRequest(
        message: "Failed to encode \(Self.self) output as UTF-8 string",
      )
    }

    let structured = try JSONDecoder().decode(Value.self, from: data)

    return ToolOutputResult(
      content: [.text(json)],
      structuredContent: structured,
    )
  }
}

// MARK: - Default Encoder Factory

/// Namespace for the library's encoder defaults. Keeping this behind a
/// dedicated `AIEncoding` name avoids ambiguity with the `AI` module itself.
public enum AIEncoding {
  /// The encoder used by `StructuredOutput.toToolResult()` unless a
  /// conforming type overrides `encoder`.
  ///
  /// - `outputFormatting: .sortedKeys` keeps the byte form stable so consumers
  ///   produce byte-equivalent output across invocations and platforms.
  /// - `dateEncodingStrategy: .iso8601` aligns with what code-mode consumers
  ///   expect.
  ///
  /// Returns a fresh encoder each call so callers can mutate it locally
  /// without affecting the library-wide default.
  public static func defaultEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    encoder.dateEncodingStrategy = .iso8601
    return encoder
  }
}
