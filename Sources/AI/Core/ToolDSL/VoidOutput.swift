// Copyright © Anthony DePasquale

import Foundation

/// Internal sentinel output type used by the `@Tool` macro and the
/// closure-based register paths to normalize `Void`-returning handlers into
/// the existing `StructuredOutput` / `ToolOutput` machinery.
///
/// Swift's `Void` is `()` — an empty tuple — and tuples can't adopt protocol
/// conformance. The library substitutes this sentinel for `Void` at the
/// `_perform` bridge. Authors never reference `VoidOutput` directly.
///
/// Wire shape:
/// - `structuredContent = {"result": null}`
/// - `content = [.text("null")]`
///
/// Identical to `Optional<T>.none` so agents can treat "no value" uniformly
/// regardless of whether the tool returns `T?` or `Void`.
public struct VoidOutput: StructuredOutput, Sendable, Encodable {
  public init() {}

  /// Schema for a Void-returning tool. Shape matches the wrap convention used
  /// by primitives — a top-level object with a single `"result"` property —
  /// but the inner type is `"null"`.
  public static let outputJSONSchema: Value = .object([
    "type": .string("object"),
    "properties": .object(["result": .object(["type": .string("null")])]),
    "required": .array([.string("result")]),
    "additionalProperties": .bool(false),
  ])

  /// Encodes as `{"result": null}` so the default `StructuredOutput.toToolResult()`
  /// path produces a `structuredContent` that matches `outputJSONSchema`. The
  /// override below short-circuits the round-trip for the display-text channel,
  /// but the encoder still has to stay honest in case it's called directly.
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encodeNil(forKey: .result)
  }

  private enum CodingKeys: String, CodingKey {
    case result
  }

  public func toToolResult() throws -> ToolOutputResult {
    ToolOutputResult(
      content: [.text("null")],
      structuredContent: .object(["result": .null]),
    )
  }
}
