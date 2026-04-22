// Copyright © Anthony DePasquale

import Foundation

/// Tool-output-level marker for values that should be wrapped under `"result"`
/// in `structuredContent`.
///
/// The companion `WrappableValue` protocol describes the *shape* of the value;
/// `PrimitiveToolOutput` adds the "when returned from a tool, wrap under
/// `result`" behavior. Primitives (`Int`, `Double`, `Bool`, `String`, `Date`),
/// arrays whose elements are `WrappableValue`, and optionals whose wrapped
/// type is `WrappableValue` all conform.
///
/// Two deliberate exclusions:
/// - `@StructuredOutput` structs keep their existing `StructuredOutput`
///   default, which emits the struct's own shape unwrapped.
/// - `Dictionary<String, V>` declares its own `ToolOutput` conformance that
///   emits the map as a top-level object (no `"result"` wrap), matching
///   Pydantic's `RootModel[dict[str, T]]`.
public protocol PrimitiveToolOutput: WrappableValue, ToolOutput {}

public extension PrimitiveToolOutput {
  /// `[.text, .json]`: `.text` for `content[].text(displayText)`; `.json` for
  /// the structured wrap (universal across providers per §A capability table).
  static var resultTypes: Set<ToolResult.ValueType>? {
    [.text, .json]
  }

  func toToolResult() throws -> ToolOutputResult {
    try ToolOutputResult(
      content: [.text(asDisplayText())],
      structuredContent: .object(["result": asJSONValue()]),
    )
  }
}

// MARK: - Conformances

extension Int: PrimitiveToolOutput {}
extension Double: PrimitiveToolOutput {}
extension Bool: PrimitiveToolOutput {}
extension String: PrimitiveToolOutput {}
extension Date: PrimitiveToolOutput {}

// Swift's conditional-conformance rules require the inherited `ToolOutput`
// to be restated explicitly even though `PrimitiveToolOutput: ToolOutput`.
extension Array: PrimitiveToolOutput, ToolOutput where Element: WrappableValue {}
extension Optional: PrimitiveToolOutput, ToolOutput where Wrapped: WrappableValue {}
